# Debugging: 00_baseline - Layer Reuse

This file tracks all debugging work for the 00_baseline experiment, including bug
discoveries, fixes, design constraints, and the ongoing Phase 6 investigation.

Referenced from [tasks.md](tasks.md) (Phases 4-6).

---

## Bug Details

### Bug 1 (CRITICAL) [FIXED - Feb 15]: Reuse path calls `original_forward()` to check return type

**File**: `generation_functions.py:104-108`
**Impact**: Throughput measurements are INVALID - no computation is actually saved

On the reuse path (where we skip computation and return cached values), the code
calls `original_forward(*args, **kwargs)` just to check `isinstance(..., tuple)`.
This means every "reuse" step runs the full layer computation anyway, then throws
the result away. There is zero actual speedup from layer reuse.

```python
# BUGGY (line 104-108):
return (
    (output_tensor,)
    if isinstance(original_forward(*args, **kwargs), tuple)  # runs full forward!
    else output_tensor
)
```

**Fix**: Cache the return type alongside the output on recompute steps:
```python
# On recompute:
layer_cache["is_tuple"] = isinstance(output, tuple)

# On reuse:
if layer_cache.get("is_tuple", False):
    return (output_tensor,)
else:
    return output_tensor
```

### Bug 2 (MODERATE) [FIXED - Feb 15]: k=1 never uses block cache fast path

**File**: `generation_functions.py:280-288`
**Impact**: k=1 runs are slower than the original code, making it a bad baseline

The `should_recompute` condition conflated block cache and layer reuse into one
flag. Conditions `reuse_k <= 1` and `reuse_step % reuse_k == 0` forced full-block
forwards even when the block cache could be used.

```python
# BUGGY:
should_recompute = (
    block_past_key_values is None
    or reuse_k <= 1                    # always True for k=1!
    or (reuse_step % reuse_k == 0)
    or (x_t[:, -block_size + small_block_start_idx] == mask_id).any()
)
```

**Fix**: Restore the original block cache condition (remove conditions 2 and 3).
Layer reuse is handled by the monkey-patched wrappers via `reuse_state["count"]`.
Note: block cache and layer reuse are NOT fully independent — see
[Design Constraint](#design-constraint-block-cache-vs-layer-reuse) below.

### Bug 3 (MODERATE) [FIXED - Feb 15]: "first" subset patches 11 layers, not 12

**File**: `generation_functions.py:33`
**Impact**: Uneven comparison - "first" applies reuse to 11 layers while
"middle" and "last" apply to 12

```python
# BUGGY:
target_indices = list(range(1, min(n, subset_size)))  # range(1,12) = 11 layers

# FIX:
target_indices = list(range(1, subset_size + 1))  # range(1,13) = 12 layers
```

### Bug 4 (MINOR) [FIXED - Feb 15]: Dead code - layer caches never trimmed

**File**: `generation_functions.py:400-408`
**Impact**: After batch trimming (finished samples removed), layer caches have
stale batch entries.

The trim code checks `reuse_state["caches"]` but layer caches lived in
closure-local dicts inside `create_wrapper()`, never registered in `reuse_state`.

**Fix**: Register each `layer_cache` dict into `reuse_state["caches"]` during
`create_wrapper()`, so the existing trim code works.

### Bug 5 (MODERATE) [FIXED - Feb 16]: Stale cache from disabled wrappers during full-block forwards

**Impact**: Middle/last subsets produced runaway gibberish (18-20K tokens instead of ~3,500)

When `reuse_state["enabled"] = False` was set during full-block forwards, wrappers
passed through to `original_forward()` without caching. This meant subsequent
small-block reuse steps had stale hidden states from a previous block.

**Fix**: Use `reuse_state["count"] = 0` instead of `reuse_state["enabled"] = False`
during full-block forwards. Since `0 % k == 0` for all k, wrappers recompute AND
cache fresh 32-token outputs.

**What this fixed**: Token counts normalized (18-20K -> ~3,500), throughput normalized
(83-97 tok/s -> ~38-42 tok/s).

**What this did NOT fix**: Accuracy for middle/last still 0-10% — but the failure mode
changed from gibberish to coherent-but-wrong answers.

### Bug 7 (MINOR) [FIXED - Mar 7]: `find_results_file()` picks arbitrary results file

**File**: `v2/log_utils.py:95-111`
**Impact**: `summary.json` reports stale accuracy from old runs instead of latest run

`os.listdir()` returns entries in arbitrary order. When multiple `results_*.json` files
exist (from re-runs with `--force`), the function picked whichever file came first —
often a stale result from a previous buggy run. This made it appear that the Bug 6 fix
hadn't worked (summary.json showed 0-10% accuracy) when the actual latest results all
showed 70%.

**Fix**: Collect all `results_*.json` candidates and sort reverse-lexicographically.
Since lm_eval uses ISO timestamps in filenames (`results_2026-03-07T16-16-31.json`),
this returns the latest file.

### Bug 8 (CRITICAL) [FIXED - Mar 7]: `full_block_output` cache never populated

**File**: `experiments/00_baseline/generation_functions.py`, wrapper recompute path
**Impact**: Layer reuse has ZERO effect — every "reuse" call runs `original_forward()`

The two-tier caching fix (Bug 6) used this condition to detect full-block forwards:

```python
if tensor.shape[1] > current_input.shape[1]:  # ALWAYS False!
    layer_cache["full_block_output"] = tensor
```

A decoder layer's output ALWAYS has the same sequence length as its input
(`[B, seq_len, D]` → `[B, seq_len, D]`). This condition was designed for the
model-level forward (where 32 input tokens → 32 output tokens), but it was placed
inside the LAYER wrapper where input and output shapes match. Result: `full_block_output`
was never stored, every reuse call fell through to `original_forward()`, and layer
reuse had literally zero effect on generation.

**Evidence** (debug stats per layer for k=3, 1 sample):
- Before fix: `cache_stored=0, reuse_fallback=43` (all reuse calls run full forward)
- After fix: `cache_stored=270, reuse_hit=108, reuse_fallback=0` (actual cache hits)

**Fix**: Replace shape comparison with `step == 0`. The generation loop forces
`reuse_state["count"] = 0` exclusively during full-block forwards. Small-block
recomputes have `count = k, 2k, 3k...` (never 0). So `step == 0` reliably identifies
full-block forwards at the layer level.

```python
# OLD (Bug 8 — always False):
if tensor.shape[1] > current_input.shape[1]:

# NEW (correctly identifies full-block forwards):
if step == 0:
```

**Confirmed working**: k1_first (538 tokens, 34.3 tok/s, 100% acc) vs k3_middle
(2267 tokens, 58.4 tok/s, 50% acc) on limit_2 — different outputs, higher throughput,
lower accuracy. Layer reuse is now actually active.

---

## Design Constraint: Block Cache vs Layer Reuse

Layer reuse and block cache are **not fully independent**. Full-block forwards
(32 tokens) BUILD `block_past_key_values` — each layer's attention must run to
populate its block cache entry via `block_past_key_values.update()`. If a patched
layer is skipped (returns cached hidden states), its block cache entry is never
created, causing a `NoneType` crash when the subsequent small-block forward
tries to read it.

**Constraint**: All layer wrappers must recompute (not skip) during full-block forwards.

~~**Old approach (broken)**: Set `reuse_state["enabled"] = False` during full-block
forwards. This prevented wrappers from caching, leaving them with stale outputs
for subsequent small-block reuse — causing middle/last collapse.~~

**Current approach**: Set `reuse_state["count"] = 0` during full-block forwards.
Since `0 % k == 0` for all k, every wrapper's `should_recompute` is True -> they
call `original_forward()` (building block cache) AND cache the 32-token output
(available for subsequent small-block reuse via `replace_position` slicing).

```
should_recompute (block cache logic)
  |
  +-- YES: full block (32 tokens) — BUILDS block_past_key_values
  |   Wrappers FORCED TO RECOMPUTE (count=0): all layers run + cache 32-token output
  |
  +-- NO: small block (8 tokens) — READS block_past_key_values
      Layer reuse ENABLED (skipped layers return cached hidden states)
```

**Implication for throughput**: The speedup opportunity from layer reuse is
limited to small-block iterations. The full-block forward (the most expensive
step in each denoising cycle) cannot benefit from layer skipping. This partly
explains why k=2 and k=3 show minimal throughput improvement over k=1.

---

## Old Code's Load-Bearing Bugs

The old code (commit `8e18baac`) had 4 known bugs. Two created behavioral side-effects
that may have been essential for correctness:

1. **Bug 1 side-effect**: On the reuse path, `isinstance(original_forward(*args, **kwargs), tuple)`
   ran the full layer computation as a side-effect. The result was discarded (cached value
   returned), but the computation ran attention + feedforward. This means the attention
   mechanism still accessed `block_past_key_values`, potentially keeping block cache entries
   consistent. **With Bug 1 fixed, the reuse path truly skips all computation.**

2. **Bug 2 side-effect**: `reuse_step % reuse_k == 0` in `should_recompute` forced
   full-block forwards every k-th step (not just when block cache was missing or masks
   present). This provided more frequent block cache AND layer cache refreshes. During
   these forced full-block forwards, wrappers stayed enabled and saw `step % k == 0`
   (same counter) -> they recomputed and cached fresh 32-token outputs. **With Bug 2
   fixed, full-block forwards only happen when `block_past_key_values is None` or when
   there's a mask at the small block start position.**

---

## Phase 6: Investigation [RESOLVED — Mar 7, 2026]

### Root Cause: Cross-Small-Block Cache Contamination (Bug 6)

**Status**: Root cause identified. Fix planned (two-tier caching).

**The problem**: Middle/last subsets collapse to 0-10% accuracy (k>=2) while first
subset works. The wrapper's single-slot cache (`layer_cache["last_output"]`) conflates
32-token full-block outputs with 8-token small-block outputs, causing wrong-position
hidden states to be returned when reuse spans across different small blocks.

### Investigation Results

#### A. `replace_position` propagation: NOT the bug

Reviewed the custom `modeling.py` from `Efficient-Large-Model/Fast_dLLM_v2_7B`
(HuggingFace, `trust_remote_code=True`). The model is Qwen2.5-based with custom
block diffusion code.

`replace_position` IS explicitly propagated through the full chain:

```
Fast_dLLM_QwenForCausalLM.forward(replace_position=X)
  -> Fast_dLLM_QwenModel.forward(replace_position=X)
    -> for decoder_layer in self.layers:
         decoder_layer(hidden_states, ..., replace_position=X)  # explicit kwarg
           -> Fast_dLLM_QwenDecoderLayer.forward(replace_position=X)
             -> self.self_attn(..., replace_position=X)
```

Both `Fast_dLLM_QwenDecoderLayer.forward()` and `Fast_dLLM_QwenAttention.forward()`
have `replace_position: Optional[int] = None` in their signatures. Our wrapper receives
it via `**kwargs`. The slicing math (`kwargs.get("replace_position") or 0`) is correct.

**Model source**: `modeling.py` and `configuration.py` at
`https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B`

#### B-F. Superseded by Bug 6 discovery

The root cause was found through code analysis of the wrapper + model interaction,
making the experimental tests (B-F) unnecessary.

### Bug 6: Cross-Small-Block Cache Contamination

**File**: `generation_functions.py`, wrapper reuse path (lines 77-113)
**Impact**: Middle/last subsets produce wrong-position hidden states → 0-10% accuracy

#### Mechanism

The wrapper uses a single cache slot (`layer_cache["last_output"]`) that gets
overwritten on every recompute, whether full-block (32 tokens) or small-block (8 tokens).
When reuse spans across different small blocks, the cached 8-token tensor from one
small block is returned for a different small block's positions.

The shape-match check masks the position mismatch:

```python
if cached_tensor.shape[1] == current_len:  # 8 == 8 → True!
    output_tensor = cached_tensor  # Returns positions 0-7 for positions 8-15!
```

#### Step-by-step trace

Processing a 32-token block with k=2, small_block_size=8:

1. **Iter 0, sb_idx=0**: `should_recompute=True` (no block cache). Full-block forward.
   `count=0` → wrappers recompute. Cache becomes **[B, 32, D]**. `reuse_step=1`.

2. **Iter 1, sb_idx=0**: Position 0 unmasked → small-block (8 tokens, `replace_pos=0`).
   `count=1`, `1%2≠0` → **REUSE**. Cache is [B,32,D], slice [:, 0:8, :] → **CORRECT**.

3. **Iter 2, sb_idx=0**: Small-block. `count=2`, `2%2==0` → **RECOMPUTE**. Runs 8-token
   forward. **Cache overwritten to [B, 8, D]** (positions 0-7). `reuse_step=3`.

4. **(Move to sb_idx=1)**

5. **Iter 3, sb_idx=1**: Position 8 already unmasked → small-block (8 tokens,
   `replace_pos=8`). `count=3`, `3%2≠0` → **REUSE**.

   ```
   cached_tensor.shape[1] = 8   (from step 3, positions 0-7)
   current_len = 8              (for positions 8-15)
   8 == 8 → shape match! → returns cached as-is
   → Returns hidden states for positions 0-7 when positions 8-15 are needed!
   ```

#### Why `first` works but `middle`/`last` collapse

When layers 1-12 (`first`) return wrong-position hidden states, **16 subsequent correct
layers** (13-27) compute fresh attention against the correct block cache K,V and can
compensate for the bad input.

When layers 16-27 (`last`) return wrong-position hidden states, **zero layers** follow
to correct. The LM head receives garbage directly → catastrophic failure.

When layers 8-19 (`middle`) return wrong hidden states, only **8 layers** (20-27) try
to correct → insufficient.

#### Why old code was immune

Both load-bearing bugs prevented this scenario:

- **Bug 1**: `original_forward()` ran as side-effect → attention updated block cache
  entries → consistent state regardless of what hidden states were returned.

- **Bug 2**: `reuse_step % reuse_k == 0` in `should_recompute` → forced full-block
  forwards every k-th step → wrappers frequently got fresh 32-token caches →
  the 8-token cache rarely persisted long enough to contaminate a different small block.

#### Block cache mutation during small-block forwards

The model's `Fast_dLLM_QwenAttention.forward()` WRITES into `block_past_key_values`
during small-block forwards (not just reads):

```python
# When block cache already exists for this layer:
block_cache_key_states[:, :, replace_position:replace_position+key_states.shape[2]] = key_states
block_cache_value_states[:, :, replace_position:replace_position+value_states.shape[2]] = value_states
```

When a layer is skipped (wrapper returns cached hidden states), its attention does not
run → its K,V in `block_past_key_values` are NOT updated at `replace_position` for the
current tokens. This is a secondary inconsistency (stale K,V for skipped layers) but is
tolerable by design — the K,V from the full-block forward are a reasonable approximation.

### Fix: Two-Tier Caching

Replace the single-slot cache with a dedicated full-block cache that small-block
recomputes cannot overwrite. On reuse, always slice from the full-block cache.

**Recompute path**:
```python
output = original_forward(*args, **kwargs)
tensor = output[0] if isinstance(output, tuple) else output
current_input = args[0]

# Only update reuse cache during full-block forwards
if tensor.shape[1] > current_input.shape[1]:
    layer_cache["full_block_output"] = tensor
    layer_cache["is_tuple"] = isinstance(output, tuple)

return output
```

**Reuse path**:
```python
if "full_block_output" in layer_cache:
    cached = layer_cache["full_block_output"]   # Always 32 tokens
    replace_pos = kwargs.get("replace_position") or 0
    current_len = args[0].shape[1]
    output_tensor = cached[:, replace_pos:replace_pos+current_len, :]
    return (output_tensor,) if layer_cache.get("is_tuple") else output_tensor
else:
    return original_forward(*args, **kwargs)     # No cache yet
```

**Why this works**:
- Full-block forwards (count=0) populate a 32-token cache
- Small-block recomputes run normally but DO NOT overwrite the 32-token cache
- Reuse ALWAYS slices from the 32-token cache using the correct `replace_position`
- Cross-small-block transitions are safe because the 32-token cache covers all positions
- Trim logic updates `full_block_output` instead of `last_output`

**Tradeoff**: The cached hidden states are always from the most recent full-block forward.
As tokens change during denoising, the cache becomes stale. This staleness is the intended
tradeoff of layer reuse — the same tradeoff that produced ~73% (k=2) and ~67% (k=3)
accuracy in the old code.

---

## Phase 9: Observations (Mar 8, 2026)

### Throughput: k=2 provides no speedup

Layer reuse only skips computation during **small-block forwards** (8 tokens). The
full-block forward (32 tokens) — the most expensive step in each denoising cycle —
always runs all layers (required to build `block_past_key_values`). With k=2, only
every other small-block iteration skips 12/28 layers. The FLOP savings are too small
to offset the wrapper overhead.

- k=1 baseline: ~36 tok/s
- k=2: ~30-36 tok/s (no improvement)
- k=3: ~51-55 tok/s (~1.4-1.5x improvement)

k=3 skips 2/3 of small-block steps for 12 layers, crossing the threshold where
savings exceed overhead.

### Quality: k=3 causes repetition and output inflation

k=3 configs generate ~2x the tokens of baseline (6,800 vs 3,500) due to repetitive
output ("Janet Janet Janet..."). The model gets stuck in repetition loops because
stale cached hidden states don't carry enough signal to properly terminate generation.

This means the **wall-clock time is longer** for k=3 (123-133s vs 82-98s) despite
higher tokens/second. The throughput metric is misleading for practical use — the
speedup is per-token, but the model generates far more (mostly garbage) tokens.

### No subset differentiation at n=10

All k=2 configs produced 50% accuracy. All k=3 configs produced 30% accuracy. No
observable difference between first/middle/last subsets.

Possible explanations:
1. **n=10 noise**: With 10 samples, accuracy resolution is 10% steps. Subset
   differences may exist but be smaller than 10%.
2. **Two-tier cache equalizes subsets**: The Bug 6 fix (always slice from 32-token
   cache using correct `replace_position`) may have eliminated the position-dependent
   failure mode that previously caused middle/last to collapse. All subsets now
   receive correctly-positioned (but stale) hidden states, making the degradation
   uniform regardless of which layers are reused.
3. **Token count similarity**: k3_first and k3_middle both generated exactly 6,806
   tokens; k2_middle and k2_last both generated 3,412. Outputs may be nearly
   identical — needs per-sample comparison.

Awaiting limit_100 results to resolve.

---

## Known Issues

- Layer reuse does NOT apply to loglikelihood tasks (MMLU, GPQA) - see docs/experiments/00-layer-reuse-loglikelihood.md
- GPQA dataset requires HuggingFace authentication - see docs/experiments/01-gpqa-authentication.md
- Throughput metrics missing for loglikelihood tasks - see docs/experiments/02-missing-throughput-tracking.md
- `transformers>=5.0.0` breaks model loading (KeyError: 'default' in ROPE_INIT_FUNCTIONS) - pinned to `<5.0.0` in pyproject.toml
