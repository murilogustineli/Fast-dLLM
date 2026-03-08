import torch
import types

# Constants for Fast_dLLM model
FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645

MASK_COLOR = 0.5
TOKEN_COLOR = -0.5


def auto_docstring(x):
    return x


# --- HELPER FUNCTIONS (Define these OUTSIDE the class) ---
def _patch_layers_helper(model, reuse_k, subset_name, reuse_state):
    """
    Replaces the .forward() method of specific layers with a wrapper
    that caches outputs and skips computation based on reuse_state.
    """
    if not subset_name or reuse_k <= 1:
        return {}

    # 1. Identify Target Layers
    layers = model.layers
    n = len(layers)
    subset_size = 12
    target_indices = []

    if subset_name == "first":
        # IMPORTANT: Skip Layer 0. It allocates memory for the cache.
        target_indices = list(range(1, subset_size + 1))
    elif subset_name == "middle":
        start = max(0, n // 2 - subset_size // 2)
        target_indices = list(range(start, start + subset_size))
    elif subset_name == "last":
        target_indices = list(range(n - subset_size, n))

    original_forwards = {}

    # Registry so batch_sample's trim logic can access per-layer caches
    reuse_state["caches"] = {}

    # 2. Define the Wrapper Function (The Closure)
    def create_wrapper(original_forward, layer_idx):
        # Local cache for this specific layer, also registered in reuse_state
        layer_cache = {}
        reuse_state["caches"][layer_idx] = layer_cache

        def wrapper(self_layer, *args, **kwargs):
            # If reuse is globally disabled, run normal
            if not reuse_state["enabled"]:
                return original_forward(*args, **kwargs)

            step = reuse_state["count"]

            # LOGIC:
            # 1. If mod k == 0: Recompute & Cache
            # 2. If mod k != 0: Reuse Cache
            should_recompute = step % reuse_k == 0

            if kwargs.get("update_past_key_values", False):
                should_recompute = True

            if should_recompute:
                output = original_forward(*args, **kwargs)
                tensor = output[0] if isinstance(output, tuple) else output

                # Only update the reuse cache during full-block forwards.
                # The generation loop forces count=0 exclusively for full-block
                # forwards (which process all 32 tokens). Small-block recomputes
                # (count=k, 2k, 3k...) must NOT overwrite the full-block cache —
                # that causes cross-small-block contamination (Bug 6).
                if step == 0:
                    layer_cache["full_block_output"] = tensor
                    layer_cache["is_tuple"] = isinstance(output, tuple)

                return output
            else:
                # Reuse: always slice from the full-block cache (32 tokens).
                # This avoids cross-small-block contamination — the 32-token
                # cache covers all positions, and replace_position selects
                # the correct 8-token slice.
                if "full_block_output" in layer_cache:
                    cached_tensor = layer_cache["full_block_output"]
                    current_input = args[0]
                    current_len = current_input.shape[1]
                    replace_pos = kwargs.get("replace_position") or 0

                    if replace_pos + current_len <= cached_tensor.shape[1]:
                        output_tensor = cached_tensor[
                            :, replace_pos : replace_pos + current_len, :
                        ]
                    else:
                        return original_forward(*args, **kwargs)

                    if layer_cache.get("is_tuple", False):
                        return (output_tensor,)
                    else:
                        return output_tensor
                else:
                    return original_forward(*args, **kwargs)
        return wrapper

    # 3. Apply the Monkey Patch
    for idx in target_indices:
        original_forwards[idx] = layers[idx].forward
        layers[idx].forward = types.MethodType(
            create_wrapper(layers[idx].forward, idx), layers[idx]
        )

    return original_forwards


def _unpatch_layers_helper(model, original_forwards):
    """Restores the original .forward() methods."""
    layers = model.layers
    for idx, orig_func in original_forwards.items():
        layers[idx].forward = orig_func


@auto_docstring
class Fast_dLLM_QwenForCausalLM:
    @torch.no_grad()
    def batch_sample(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        reuse_k,
        layer_subset=None,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
    ):
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]

        # --- SETUP LAYER REUSE ---
        # reuse_state is a mutable dictionary shared between the loop and the layers
        reuse_state = {"count": 0, "enabled": False}

        # Apply the patches (Call the global helper function)
        original_forwards = _patch_layers_helper(
            self.model, reuse_k, layer_subset, reuse_state
        )
        # -------------------------

        if min_len > block_size:
            output = self.forward(
                input_ids=input_ids[:, : (min_len // block_size * block_size)],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
            logits, past_key_values = output.logits, output.past_key_values
            if min_len % block_size == 0:
                predict_sample_idx = seq_len == min_len
                predict_logits = logits[predict_sample_idx, -1:, :]
                next_token = predict_logits.argmax(dim=-1)
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)

        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}

        for block_idx in range(start_block_idx, num_blocks):
            if finished_flag.all():
                break
            if (seq_block_idx == block_idx).all():
                x_init = mask_id * torch.ones(
                    (input_ids.shape[0], block_size - input_ids.shape[1] % block_size),
                    device=self.device,
                    dtype=torch.long,
                )
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, : (block_idx + 1) * block_size]

            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()
            step = 0
            block_past_key_values = None

            # --- First Pass Loop (Initial Block Prediction) ---
            while True:
                mask_idx = x_t[:, -block_size:] == mask_id
                if mask_idx.sum() == 0:
                    for sample_idx in range(x_t.shape[0]):
                        if (
                            finished_flag[sample_idx]
                            and seq_len[sample_idx] < (block_idx + 1) * block_size
                        ):
                            stop_token_idx = (
                                x_t[sample_idx, seq_len[sample_idx] :] == stop_token
                            ).nonzero()[0][0]
                            x_t[
                                sample_idx, seq_len[sample_idx] + stop_token_idx + 1 :
                            ] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break

                    # Disable reuse for prefill/correction steps
                    reuse_state["enabled"] = False

                    output = self.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True,
                        block_size=block_size,
                    )
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    step += 1
                    break

                # --- SMALL BLOCK LOOP ---
                # Reset reuse step at the start of small block processing
                reuse_step = 0
                # Enable reuse for the speculative generation loop
                reuse_state["enabled"] = True

                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = (
                        None
                        if block_size == small_block_end_idx
                        else -block_size + small_block_end_idx
                    )
                    while True:
                        mask_idx = x_t[:, -block_size:] == mask_id
                        if mask_idx[:, start:end].sum() == 0:
                            break

                        # --- REUSE BLOCK CACHE ACTIVATIONS ---
                        if use_block_cache:
                            # SYNC THE LAYERS WITH THE CURRENT STEP
                            reuse_state["count"] = reuse_step

                            # Block cache decision: full block vs small block.
                            # This is the ORIGINAL Fast-dLLM logic — independent
                            # of layer reuse (which is handled by the monkey-patched
                            # wrappers via reuse_state["count"]).
                            should_recompute = (
                                block_past_key_values is None
                                or (
                                    x_t[:, -block_size + small_block_start_idx]
                                    == mask_id
                                ).any()
                            )

                            if should_recompute:
                                # Full Compute — BUILDS block_past_key_values.
                                # Force all layer wrappers to recompute (not skip)
                                # by setting count to 0 (0 % k == 0 for all k).
                                # This ensures:
                                # 1. All layers run original_forward → block cache
                                #    entries populated (no None crashes)
                                # 2. Wrappers cache the full 32-token output, so
                                #    subsequent small-block reuse steps can slice
                                #    the correct positions via replace_position
                                reuse_state["count"] = 0
                                output = self.forward(
                                    input_ids=x_t[:, -block_size:],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_size=block_size,
                                )
                                logits, block_past_key_values = (
                                    output.logits,
                                    output.block_past_key_values,
                                )
                                logits = torch.cat(
                                    [logits[:, :1, :], logits[:, :-1, :]], dim=1
                                )
                                logits = logits[:, start:end]
                            else:
                                # Reuse Compute (Fast Path) — READS block_past_key_values.
                                # Layer reuse is safe here: skipped layers return
                                # cached hidden states; block cache is only read, not built.
                                logits = self.forward(
                                    input_ids=x_t[:, start:end],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=small_block_start_idx,
                                    block_size=block_size,
                                ).logits
                                logits = torch.cat(
                                    [logits[:, :1, :], logits[:, :-1, :]], dim=1
                                )
                            # Increment the step counter for this small block sequence
                            reuse_step += 1
                            # --- REUSE LOGIC END ---

                        else:
                            # Disable layer reuse if block cache is off
                            reuse_state["enabled"] = False
                            # Standard path without block cache
                            logits = self.forward(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                            ).logits
                            logits = torch.cat(
                                [logits[:, :1, :], logits[:, :-1, :]], dim=1
                            )
                            logits = logits[:, start:end]
                        # --- END REUSE BLOCK CACHE ACTIVATIONS ---

                        # sampling and unmasking
                        x_1, p_1t = self.sample_with_top_p(
                            logits, top_p=top_p, temperature=temperature
                        )
                        x1_p = torch.squeeze(
                            torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)),
                            -1,
                        )
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)
                        unmask_idx = x1_p > threshold
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]
                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
                        finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(
                            dim=1
                        )  # shape: [B]
                        finished_flag = finished_flag | finished_row_flags
                        step += 1
            # --- END SMALL BLOCK LOOP ---

            if input_ids.shape[1] == x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, : (block_idx + 1) * block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1) * block_size:
                        input_ids = x_t
                    else:
                        input_ids[
                            seq_block_idx == block_idx, (block_idx + 1) * block_size
                        ] = x_t[
                            seq_block_idx == block_idx, (block_idx + 1) * block_size
                        ]
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

            # --- Trim Logic ---
            # This handles cases where len(past_key_values) != len(key_cache)
            if finished_flag.any():
                # 1. SAVE FINISHED SAMPLES
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = (
                            x_t[sample_idx : sample_idx + 1].clone().squeeze(dim=0)
                        )

                # 2. TRIM TENSORS
                sample_indices = sample_indices[~finished_flag]
                input_ids = input_ids[~finished_flag]
                seq_block_idx = seq_block_idx[~finished_flag]
                seq_len = seq_len[~finished_flag]
                x_t = x_t[~finished_flag]

                # --- TRIM LOCAL LAYER CACHES ---
                if "caches" in reuse_state:
                    for layer_id, layer_cache in reuse_state["caches"].items():
                        if "full_block_output" in layer_cache:
                            layer_cache["full_block_output"] = layer_cache[
                                "full_block_output"
                            ][~finished_flag]

                # 3. TRIM KV CACHE
                if past_key_values is not None:
                    # Case A: Standard Tuple of Tuples (Legacy HF)
                    # This was missing and caused crash
                    if isinstance(past_key_values, (tuple, list)):
                        new_past = []
                        for layer_past in past_key_values:
                            k, v = layer_past
                            new_k = k[~finished_flag]
                            new_v = v[~finished_flag]
                            new_past.append((new_k, new_v))
                        past_key_values = type(past_key_values)(new_past)
                    # Case B: DynamicCache (list of key_cache tensors)
                    if hasattr(past_key_values, "key_cache") and hasattr(
                        past_key_values, "value_cache"
                    ):
                        L = len(past_key_values.key_cache)
                        for layer_id in range(L):
                            past_key_values.key_cache[layer_id] = (
                                past_key_values.key_cache[layer_id][~finished_flag]
                            )
                            past_key_values.value_cache[layer_id] = (
                                past_key_values.value_cache[layer_id][~finished_flag]
                            )
                    # Case C: Caches list (some modern HF models)
                    elif hasattr(past_key_values, "caches"):
                        for layer_cache in past_key_values.caches:
                            if hasattr(layer_cache, "k_cache"):
                                layer_cache.k_cache = layer_cache.k_cache[
                                    ~finished_flag
                                ]
                                layer_cache.v_cache = layer_cache.v_cache[
                                    ~finished_flag
                                ]
                            elif hasattr(layer_cache, "kv_cache"):
                                k, v = layer_cache.kv_cache
                                layer_cache.kv_cache = (
                                    k[~finished_flag],
                                    v[~finished_flag],
                                )

                # 4. UPDATE FLAG
                finished_flag = finished_flag[~finished_flag]

        if len(finished_samples) < batch_size:
            for sample_idx in range(x_t.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = (
                    x_t[sample_idx : sample_idx + 1].clone().squeeze(dim=0)
                )

        # --- TEARDOWN LAYER REUSE ---
        _unpatch_layers_helper(self.model, original_forwards)
        # ----------------------------

        assert len(finished_samples) == batch_size
        return finished_samples

    @torch.no_grad()
    def mdm_sample_with_visualization(
        self,
        input_ids,
        tokenizer,
        block_size=32,
        max_new_tokens=1024,
        mask_id=FAST_DLLM_MASK_ID,
        threshold=0.95,
        small_block_size=32,
        stop_token=FAST_DLLM_STOP_TOKEN,
        temperature=0.0,
        top_p=0.95,
    ):
        """
        MDM sampling function with visualization
        with intermediate state output for Gradio visualization
        """
        nfe = 0
        self.model.bd_size = block_size
        num_blocks = max_new_tokens // block_size

        # Initialize state - show all positions as mask
        initial_state = []
        current_state = []

        if input_ids.shape[1] > block_size:
            output = self.forward(
                input_ids=input_ids[
                    :, : (input_ids.shape[1] // block_size * block_size)
                ],
                use_cache=True,
                update_past_key_values=True,
            )
            logits, past_key_values = output.logits, output.past_key_values
            nfe += 1
            if input_ids.shape[1] % block_size == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        else:
            past_key_values = None

        num_small_blocks = block_size // small_block_size
        original_input_length = input_ids.shape[1]

        for block_idx in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                break
            prompt_length = input_ids.shape[1]

            # Use the length of the first block to initialize state
            first_block_length = block_size - (input_ids.shape[1] % block_size)

            if len(initial_state) == 0:
                for i in range(first_block_length):
                    initial_state.append(("[MASK]", MASK_COLOR))
                yield initial_state
            else:
                for i in range(first_block_length):
                    current_state.append(("[MASK]", MASK_COLOR))
                yield current_state

            # Initialize x_init as mask_id
            x_init = mask_id * torch.ones(
                (input_ids.shape[0], block_size - prompt_length % block_size),
                device=self.device,
                dtype=torch.long,
            )
            x_init = torch.cat([input_ids, x_init], dim=1)

            x_t = x_init.clone()
            step = 0

            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[
                        0
                    ][1]
                    if (
                        x_t[:, prompt_length : prompt_length + stop_token_idx]
                        == mask_id
                    ).sum() == 0:
                        break
                mask_idx = x_t[:, -block_size:] == mask_id
                # Decode a complete block, update cache, and generate next token
                if mask_idx.sum() == 0:
                    nfe += 1
                    output = self.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True,
                    )
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    token_text = tokenizer.decode(
                        [next_token[0].item()], skip_special_tokens=True
                    )
                    # Handle special characters
                    token_text = token_text
                    current_state.append((token_text, TOKEN_COLOR))
                    yield current_state
                    break

                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = (
                        None
                        if block_size == small_block_end_idx
                        else -block_size + small_block_end_idx
                    )
                    while True:
                        mask_idx = x_t[:, -block_size:] == mask_id
                        if mask_idx[:, start:end].sum() == 0:
                            break
                        if stop_token in x_t[:, prompt_length:]:
                            stop_token_idx = (
                                x_t[:, prompt_length:] == stop_token
                            ).nonzero()[0][1]
                            if (
                                x_t[:, prompt_length : prompt_length + stop_token_idx]
                                == mask_id
                            ).sum() == 0:
                                break

                        logits = self.forward(
                            input_ids=x_t[:, -block_size:],
                            use_cache=True,
                            past_key_values=past_key_values,
                            update_past_key_values=False,
                        ).logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]

                        step += 1
                        x_1, p_1t = self.sample_with_top_p(
                            logits, top_p=top_p, temperature=temperature
                        )

                        # Select tokens with probability greater than threshold in p_1t
                        x1_p = torch.squeeze(
                            torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)),
                            -1,
                        )
                        x1_p = torch.where(
                            mask_idx[:, small_block_start_idx:small_block_end_idx],
                            x1_p,
                            -torch.inf,
                        )
                        unmask_idx = x1_p > threshold
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        # Generate visualization state
                        current_state = []
                        generated_tokens = x_t[0, original_input_length:]

                        # Display generated tokens
                        for i, token_id in enumerate(generated_tokens):
                            if token_id == mask_id:
                                current_state.append(("[MASK]", MASK_COLOR))
                            else:
                                token_text = tokenizer.decode(
                                    [token_id.item()], skip_special_tokens=True
                                )
                                # Handle special characters
                                token_text = token_text
                                current_state.append((token_text, TOKEN_COLOR))

                        yield current_state

            input_ids = x_t

        # Truncate stop_token
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (
                input_ids[:, original_input_length:] == stop_token
            ).nonzero()[0][1]
            input_ids = input_ids[:, : stop_token_idx + original_input_length + 1]

        # Final state - display complete text
        final_state = []
        generated_tokens = input_ids[0, original_input_length:]
        for token_id in generated_tokens:
            token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
            token_text = token_text
            final_state.append((token_text, TOKEN_COLOR))

        # Final state doesn't need mask padding, only show actually generated tokens

        yield final_state

        # Return final text
        final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        yield final_text


def setup_model_with_custom_generation(model):
    """
    Set up custom generation functions for the model
    """
    # Add mdm_sample method with visualization
    model.mdm_sample_with_visualization = types.MethodType(
        Fast_dLLM_QwenForCausalLM.mdm_sample_with_visualization, model
    )
    return model
