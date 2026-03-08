#!/usr/bin/env python3
"""
Phase 10 Verification: Steps B + C
===================================
Runs a single GSM8K sample with instrumented layer reuse to verify:

  B. Layer indices: Confirms which layers are patched and that the wrapper
     is actually called with correct recompute/reuse decisions.

  C. Cache staleness: On reuse steps, computes BOTH the cached result and
     the real original_forward() result, logging L2 distance between them.
     If distance is ~0, cached hidden states match fresh ones (layer reuse
     has no effect regardless of which layers are targeted).

Usage (from v2/ directory, with GPU):
  python experiments/00_baseline/verify_wrapper.py --reuse_k 2 --layer_subset first
  python experiments/00_baseline/verify_wrapper.py --reuse_k 2 --layer_subset middle
  python experiments/00_baseline/verify_wrapper.py --reuse_k 3 --layer_subset first
"""

import argparse
import json
import os
import sys
import types

import torch

# Add v2/ to path so we can import eval utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from transformers import AutoTokenizer, AutoModelForCausalLM


def get_optimal_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor >= 80:
        return torch.bfloat16
    elif major * 10 + minor >= 70:
        return torch.float16
    return torch.float32


def build_layer_indices(n_layers, subset_name, subset_size=12):
    """Reproduce the exact logic from generation_functions.py."""
    if subset_name == "first":
        return list(range(1, subset_size + 1))
    elif subset_name == "middle":
        start = max(0, n_layers // 2 - subset_size // 2)
        return list(range(start, start + subset_size))
    elif subset_name == "last":
        return list(range(n_layers - subset_size, n_layers))
    return []


def main():
    parser = argparse.ArgumentParser(description="Verify layer reuse wrapper")
    parser.add_argument("--reuse_k", type=int, default=2)
    parser.add_argument("--layer_subset", type=str, default="first",
                        choices=["first", "middle", "last"])
    parser.add_argument("--model_path", type=str,
                        default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    args = parser.parse_args()

    print("=" * 70)
    print(f"VERIFICATION: k={args.reuse_k}, subset={args.layer_subset}")
    print("=" * 70)

    # --- Load model ---
    dtype = get_optimal_dtype()
    print(f"[INFO] Loading model ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=dtype
    ).eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    n_layers = len(model.model.layers)
    target_indices = build_layer_indices(n_layers, args.layer_subset)

    print(f"\n[STEP B] Layer index verification")
    print(f"  Model has {n_layers} layers")
    print(f"  Subset '{args.layer_subset}' targets: {target_indices}")
    print(f"  ({len(target_indices)} layers)")

    # Verify no overlap between subsets
    first_idx = build_layer_indices(n_layers, "first")
    middle_idx = build_layer_indices(n_layers, "middle")
    last_idx = build_layer_indices(n_layers, "last")
    print(f"\n  All subsets:")
    print(f"    first:  {first_idx}")
    print(f"    middle: {middle_idx}")
    print(f"    last:   {last_idx}")
    print(f"    first ∩ middle: {sorted(set(first_idx) & set(middle_idx))}")
    print(f"    first ∩ last:   {sorted(set(first_idx) & set(last_idx))}")
    print(f"    middle ∩ last:  {sorted(set(middle_idx) & set(last_idx))}")

    # --- Load generation functions and patch ---
    import importlib.util
    gen_path = os.path.join(os.path.dirname(__file__), "generation_functions.py")
    spec = importlib.util.spec_from_file_location("gen_funcs", gen_path)
    gen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_module)

    # Bind batch_sample to the model
    model.mdm_sample = types.MethodType(
        gen_module.Fast_dLLM_QwenForCausalLM.batch_sample, model
    )

    # --- Instrument the wrapper for Step C ---
    # We'll monkey-patch _patch_layers_helper to add staleness measurement
    original_patch = gen_module._patch_layers_helper
    staleness_log = []  # Collect (layer_idx, step, l2_dist, rel_dist) tuples

    def instrumented_patch(model_inner, reuse_k, subset_name, reuse_state):
        """Wraps the normal patching to add staleness measurement."""
        if not subset_name or reuse_k <= 1:
            return {}

        # First, apply normal patching
        original_forwards = original_patch(model_inner, reuse_k, subset_name, reuse_state)

        # Now wrap each patched layer AGAIN to measure staleness on reuse
        layers = model_inner.layers
        call_log = []  # (layer_idx, step, decision, seq_len)

        for idx in list(original_forwards.keys()):
            # Get the wrapper that _patch_layers_helper installed
            patched_forward = layers[idx].forward
            # Get the TRUE original forward (before any patching)
            true_original = original_forwards[idx]

            def create_staleness_wrapper(patched_fwd, true_orig, layer_idx):
                def staleness_wrapper(self_layer, *args, **kwargs):
                    step = reuse_state.get("count", -1)
                    enabled = reuse_state.get("enabled", False)
                    seq_len = args[0].shape[1] if args else -1

                    if not enabled:
                        call_log.append((layer_idx, step, "disabled", seq_len))
                        return true_orig(*args, **kwargs)

                    should_recompute = step % reuse_k == 0
                    if kwargs.get("update_past_key_values", False):
                        should_recompute = True

                    if should_recompute:
                        call_log.append((layer_idx, step, "recompute", seq_len))
                        # Run through the patched wrapper (which caches)
                        return patched_fwd(*args, **kwargs)
                    else:
                        # REUSE path — this is where we measure staleness
                        # 1. Get what the wrapper returns (cached value)
                        cached_result = patched_fwd(*args, **kwargs)
                        cached_tensor = cached_result[0] if isinstance(cached_result, tuple) else cached_result

                        # 2. Get what original_forward would return (fresh)
                        fresh_result = true_orig(*args, **kwargs)
                        fresh_tensor = fresh_result[0] if isinstance(fresh_result, tuple) else fresh_result

                        # 3. Measure L2 distance
                        l2_dist = torch.norm(cached_tensor.float() - fresh_tensor.float()).item()
                        fresh_norm = torch.norm(fresh_tensor.float()).item()
                        rel_dist = l2_dist / (fresh_norm + 1e-8)

                        staleness_log.append((layer_idx, step, l2_dist, rel_dist, seq_len))
                        call_log.append((layer_idx, step, f"reuse(L2={l2_dist:.2f},rel={rel_dist:.4f})", seq_len))

                        # Return the cached result (normal behavior)
                        return cached_result

                return staleness_wrapper

            layers[idx].forward = types.MethodType(
                create_staleness_wrapper(patched_forward, true_original, idx),
                layers[idx]
            )

        # Store logs in reuse_state for later access
        reuse_state["_call_log"] = call_log
        return original_forwards

    # Replace the module's _patch_layers_helper
    gen_module._patch_layers_helper = instrumented_patch

    # --- Run a single sample ---
    print(f"\n[STEP C] Running single GSM8K sample with staleness measurement...")
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Question: A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?\n"
        "Answer:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    seq_len = torch.tensor([input_ids.shape[1]], device="cuda")

    reuse_state_ref = {"count": 0, "enabled": False}

    result = model.mdm_sample(
        input_ids=input_ids,
        tokenizer=tokenizer,
        block_size=32,
        max_new_tokens=256,
        small_block_size=8,
        min_len=0,
        seq_len=seq_len,
        reuse_k=args.reuse_k,
        layer_subset=args.layer_subset,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=True,
    )

    # --- Report ---
    output_ids = result[0]
    output_text = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True)

    print(f"\n{'=' * 70}")
    print(f"GENERATED OUTPUT ({len(output_ids) - input_ids.shape[1]} tokens):")
    print(f"{'=' * 70}")
    print(output_text[:500])
    if len(output_text) > 500:
        print(f"... ({len(output_text)} chars total)")

    # Step B report: wrapper call summary
    print(f"\n{'=' * 70}")
    print(f"[STEP B] WRAPPER CALL SUMMARY")
    print(f"{'=' * 70}")

    # Count calls per decision type per layer
    from collections import defaultdict
    layer_decisions = defaultdict(lambda: defaultdict(int))
    for layer_idx, step, decision, seq_len in staleness_log and [] or []:
        pass  # staleness_log has different format

    # Use reuse_state to get call_log - but we need to access it from gen_module
    # Actually the call_log was stored in reuse_state but that's local. Let's use staleness_log.
    if staleness_log:
        print(f"\nReuse calls with staleness measurement: {len(staleness_log)}")
        print(f"\n{'Layer':>6} {'Step':>5} {'L2 Dist':>10} {'Rel Dist':>10} {'SeqLen':>7}")
        print("-" * 45)

        # Show first 20 and summary
        for i, (layer_idx, step, l2, rel, sl) in enumerate(staleness_log[:20]):
            print(f"{layer_idx:>6} {step:>5} {l2:>10.2f} {rel:>10.4f} {sl:>7}")
        if len(staleness_log) > 20:
            print(f"  ... ({len(staleness_log)} total reuse calls)")

        # Per-layer summary
        print(f"\n{'=' * 70}")
        print(f"[STEP C] CACHE STALENESS SUMMARY (per layer)")
        print(f"{'=' * 70}")

        layer_stats = defaultdict(list)
        for layer_idx, step, l2, rel, sl in staleness_log:
            layer_stats[layer_idx].append(rel)

        print(f"\n{'Layer':>6} {'Count':>6} {'Mean RelDist':>13} {'Min':>8} {'Max':>8}")
        print("-" * 50)
        for layer_idx in sorted(layer_stats.keys()):
            dists = layer_stats[layer_idx]
            print(f"{layer_idx:>6} {len(dists):>6} {sum(dists)/len(dists):>13.6f} "
                  f"{min(dists):>8.6f} {max(dists):>8.6f}")

        # Overall
        all_rel = [rel for _, _, _, rel, _ in staleness_log]
        print(f"\n  Overall: {len(all_rel)} reuse calls, "
              f"mean relative L2 = {sum(all_rel)/len(all_rel):.6f}, "
              f"max = {max(all_rel):.6f}")

        if sum(all_rel)/len(all_rel) < 0.001:
            print(f"\n  ⚠ WARNING: Mean relative L2 < 0.001 — cached values nearly identical")
            print(f"  to fresh computation. Layer reuse may have negligible effect.")
        elif sum(all_rel)/len(all_rel) > 0.1:
            print(f"\n  ✓ Cache staleness is significant (mean rel L2 > 0.1).")
            print(f"  Layer reuse IS changing the hidden states meaningfully.")
    else:
        print("\n  No reuse calls recorded (k=1 or no reuse steps taken)")

    print(f"\n{'=' * 70}")
    print("VERIFICATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
