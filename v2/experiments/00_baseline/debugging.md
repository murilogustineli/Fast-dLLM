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

## Phase 6: Investigation

### Context

**The problem**: Our refactored `generation_functions.py` produces correct results for
the `first` layer subset but collapses to 0-10% accuracy for `middle`/`last` (k>=2).
The old code (commit `8e18baac`) produced **uniform accuracy across all subsets**
(~73% for k=2, ~67% for k=3 on full GSM8K). So the collapse is a regression in our
code, not an architectural limitation.

**How layer reuse works**: We monkey-patch 12 transformer layer `.forward()` methods
with wrappers (`_patch_layers_helper` in `generation_functions.py`). On recompute steps
(`step % k == 0`), wrappers call `original_forward()` and cache the output. On reuse
steps (`step % k != 0`), wrappers return the cached output (sliced via `replace_position`
if the cached shape [B,32,D] doesn't match the input shape [B,8,D]).

**How block diffusion works**: Each block of 32 tokens is denoised iteratively. Within
each block, 4 small blocks of 8 tokens are processed. The `should_recompute` flag decides
whether to run a full-block forward (32 tokens, builds `block_past_key_values`) or a
small-block forward (8 tokens, reads from `block_past_key_values`). Layer reuse only
activates during small-block forwards.

**What we changed from old code (4 bug fixes)** — see [Bug Details](#bug-details) above:
- Bug 1 fix: Removed `original_forward()` call on the reuse path (was used for isinstance
  check). This means reuse steps now truly skip computation.
- Bug 2 fix: Removed `reuse_k <= 1` and `reuse_step % reuse_k == 0` from `should_recompute`.
  This was conflating block cache and layer reuse logic. **Side-effect lost**: the old code
  forced full-block forwards every k steps, keeping layer caches fresh with 32-token outputs.
- Bug 3 fix: `first` subset now patches 12 layers (was 11).
- Bug 4 fix: Layer caches registered in `reuse_state["caches"]` for batch trim logic.

**What we tried so far** (Feb 16, Bug 5 fix): Changed full-block forward handling from
`reuse_state["enabled"] = False` (disables wrappers entirely — no caching) to
`reuse_state["count"] = 0` (forces wrappers to recompute — `0 % k == 0` for all k).
This fixed the runaway gibberish (18-20K tokens -> ~3,500) but accuracy stayed at 0-10%.

**Key evidence: Old full GSM8K results disprove "layer sensitivity" hypothesis**

The full GSM8K runs (Phase 1, Jan 2026, commit `8e18baac`) show uniform accuracy
across all subsets for each k value:

| k | first | middle | last |
|---|-------|--------|------|
| 1 | 80.44% | 80.44% | 80.44% |
| 2 | 73.69% | 73.62% | 73.77% |
| 3 | 67.17% | 67.55% | 67.70% |

The accuracy drop from k1->k2->k3 proves layer reuse WAS affecting accuracy (it wasn't
no-op). However, the uniformity across subsets is now understood to be an artifact of
Bug 1 — `original_forward()` ran as a side-effect on every reuse step, keeping the
model's internal state (block cache, attention) coherent regardless of which layers
returned cached values.

### Key files

- Current code: `v2/experiments/00_baseline/generation_functions.py`
- Old working code: `git show 8e18baac:v2/generation_functions.py`
- Experiment proposal: `v2/experiments/00_baseline/proposal.md`
- Results: `v2/experiments/00_baseline/artifacts/gsm8k/` (full) and `gsm8k_limit_10/`

### Investigation checklist

Start with A (quick code read), then B and C (each is a small code change + limit_10
run). D and E provide supporting evidence. F is the fallback if nothing else works.

- [ ] **A. Check `replace_position` propagation** (code read, ~30 min)

  The wrapper slices cached 32-token output using `kwargs.get("replace_position")`.
  But `replace_position` is passed to `self.forward()` (the model), not directly to
  individual layers. If the model doesn't pass it through to layer kwargs, the wrapper
  always sees `replace_position=None` -> `replace_pos = 0` -> always slices positions
  0-7 regardless of which small block is being processed.

  **How to check**: Read the model's forward method. The model is
  `Efficient-Large-Model/Fast_dLLM_v2_7B` (Qwen-based). Trace how `replace_position`
  flows from `model.forward()` -> `model.model.forward()` -> individual `layer.forward()`
  calls. Check the HuggingFace model files or `modeling_qwen2.py`.

  **If NOT propagated**: The wrapper's slicing is broken for small blocks 1-3 (always
  returns positions 0-7). This would affect all subsets equally in the old code (uniform
  degradation), but could interact differently with our bug fixes to cause subset-dependent
  failure. This finding would change our entire approach.

  **If propagated**: Slicing works correctly and the issue is elsewhere.

- [ ] **B. Test: restore periodic full-block forwards** (code change + run, ~2 hours)

  Add back `reuse_step % reuse_k == 0` to `should_recompute` (from the old Bug 2).
  Keep the Bug 5 fix (`count=0`), do NOT revert to `enabled=False`. This tests whether
  more frequent cache refreshes rescue middle/last accuracy.

  In `generation_functions.py`, change the `should_recompute` block (~line 280) to:
  ```python
  should_recompute = (
      block_past_key_values is None
      or (reuse_step % reuse_k == 0)  # periodic cache refresh
      or (x_t[:, -block_size + small_block_start_idx] == mask_id).any()
  )
  ```
  Run all 9 configs with `--limit 10`.

  **If accuracy recovers** (~60-70% for middle/last): The issue is that our current code
  doesn't refresh caches frequently enough. We then need to find a way to refresh caches
  without forcing expensive full-block forwards every k steps.

  **If accuracy stays at 0-10%**: The issue is NOT cache refresh frequency. Move to C.

- [ ] **C. Test: reinstate Bug 1 side-effect** (code change + run, ~2 hours)

  Add back the `original_forward()` call on the reuse path (as a diagnostic, discarding
  the result). This tests whether the side-effect computation is necessary for correctness.

  In `generation_functions.py`, in the wrapper's reuse path (~line 77-110), add before
  the return:
  ```python
  # Diagnostic: run original_forward as side-effect (like old Bug 1)
  _ = original_forward(*args, **kwargs)
  ```
  Run all 9 configs with `--limit 10`.

  **If accuracy recovers**: Bug 1's side-effect was load-bearing. The attention computation
  in `original_forward` was updating block cache entries or other mutable state. We need
  to understand what state and find a targeted fix (not the full side-effect).

  **If accuracy stays at 0-10%**: Bug 1 side-effect is not the cause. The issue is purely
  in Bug 2 (should_recompute frequency) or something else entirely.

- [ ] **D. Add instrumentation** (code change, ~1 hour)

  Add counters/logging to the wrapper to understand actual behavior:
  - Per-layer: count of recompute vs reuse decisions per block
  - Cache tensor shape at each reuse decision ([B,32,D] vs [B,8,D])
  - `replace_position` value received (or None) at each wrapper call
  - Whether the shape-match branch or slice branch is taken

  Run one config (e.g., k2_middle) with `--limit 1` and analyze the log.

- [ ] **E. Run old code on limit_10** (checkout + run, ~2 hours)

  `git stash && git checkout 8e18baac -- v2/generation_functions.py` and run all 9
  configs with `--limit 10`. This gives a direct comparison on the small sample.
  Restore after: `git checkout HEAD -- v2/generation_functions.py && git stash pop`.

  **Purpose**: Confirm the old code gives ~60-70% for middle/last on limit_10 (proving
  the limit_10 sample is viable for detecting the regression).

- [ ] **F. Diff-driven bisection** (multiple runs, ~4-8 hours)

  If A-C don't identify the cause, systematically revert individual bug fixes (one at
  a time) on the current code and test each:
  1. Revert Bug 1 fix only (reinstate isinstance + original_forward)
  2. Revert Bug 2 fix only (add back `reuse_k <= 1` and `reuse_step % reuse_k == 0`)
  3. Revert Bug 3 fix only (change `range(1, subset_size + 1)` back to `range(1, min(n, subset_size))`)
  4. Revert Bug 4 fix only (remove `reuse_state["caches"]` registration)

  Each revert + run isolates one fix. The one that recovers accuracy is the culprit.

---

## Known Issues

- Layer reuse does NOT apply to loglikelihood tasks (MMLU, GPQA) - see docs/experiments/00-layer-reuse-loglikelihood.md
- GPQA dataset requires HuggingFace authentication - see docs/experiments/01-gpqa-authentication.md
- Throughput metrics missing for loglikelihood tasks - see docs/experiments/02-missing-throughput-tracking.md
- `transformers>=5.0.0` breaks model loading (KeyError: 'default' in ROPE_INIT_FUNCTIONS) - pinned to `<5.0.0` in pyproject.toml
