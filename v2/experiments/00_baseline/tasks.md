# Tasks: 00_baseline

## Phase 0: Planning [DONE]
- [x] Define 3x3 experiment grid (k x subset)
- [x] Create generation_functions.py with layer reuse logic
- [x] Set up smart runner (run.sh) and job template (_job.sh)

## Phase 1: GSM8K Full Runs [DONE — RESULTS SUSPECT]
- [x] Run all 9 configs on GSM8K (full, 1319 samples)
- [x] Verify all 9 summary.json files in artifacts/gsm8k/
- [x] Record accuracy and throughput in results.md

> Results suspect: ran on buggy code (commit `8e18baac`). Throughput invalid (Bug 1).
> Accuracy uniformity across subsets is an artifact of Bug 1's side-effect. Full re-runs
> needed after Phase 6 investigation. See [debugging.md](debugging.md#bug-details).

## Phase 2: MMLU Full Runs [DONE — LAYER REUSE N/A]
- [x] Run all 9 configs on MMLU (full)
- [x] Verify all 9 summary.json files in artifacts/mmlu/
- [x] Discover: layer reuse has no effect on loglikelihood tasks
- [x] Document finding in results.md

> All 9 configs produced identical accuracy (66.53%). Valid but uninformative for
> evaluating layer reuse.

## Phase 3: Remaining Tasks [PARTIALLY DONE — UNBLOCKED]
- [x] Run MMLU limit_10 (9 configs)
- [x] Run Minerva Math limit_10 (9 configs)
- [x] Run IFEval limit_5 (1 config)
- [ ] Run full Minerva Math (all 9 configs)
- [ ] Run full IFEval (all 9 configs)
- [ ] Run full GSM8K re-run (all 9 configs, corrected code)

## Phase 4: Bug Fixes [DONE]

Code review (Feb 2026) found 4 bugs in `generation_functions.py` that invalidate
throughput measurements. Accuracy measurements are still valid.

- [x] Bug 1 (CRITICAL): Reuse path calls `original_forward()` — zero actual speedup
- [x] Bug 2 (MODERATE): `should_recompute` conflates block cache + layer reuse
- [x] Bug 3 (MODERATE): "first" subset patches 11 layers instead of 12
- [x] Bug 4 (MINOR): Layer caches never trimmed
- [x] Validate fixes with k1/k2/k3 limit-10 runs

> Full bug descriptions and code snippets in [debugging.md](debugging.md#bug-details).

## Phase 5: Re-validation [DONE]
- [x] Run all 9 configs on GSM8K `--limit 10` — Feb 15 (Run 1)
- [x] Identify Bug 5 (cross-small-block stale cache)
- [x] Fix Bug 5: `reuse_state["count"] = 0` instead of `reuse_state["enabled"] = False`
- [x] Re-run all 9 configs on GSM8K `--limit 10` — Feb 16 (Run 2)
- [x] Email update to Dr. Lin re: bugs found, old results suspect
- [x] Submit full SLURM runs — Phase 6 resolved, see Phase 8

> Bug 5 fix solved runaway generation (18K -> 3.5K tokens) but accuracy for middle/last
> still 0-10%. See [debugging.md](debugging.md#bug-5-moderate-fixed---feb-16-stale-cache-from-disabled-wrappers-during-full-block-forwards).

### Validation Results: GSM8K limit_10

**Run 1 (Feb 15)** — mixed commits, Bug 5 unfixed:

| Config | Accuracy | Tokens/s | Notes |
|--------|----------|----------|-------|
| k1_first | 60% | 47.52 | No reuse (k=1) |
| k1_middle | 70% | 43.37 | No reuse (k=1) |
| k1_last | 70% | 42.94 | No reuse (k=1) |
| k2_first | 70% | 36.74 | OK |
| k3_first | 60% | 37.87 | OK |
| k2_middle | **10%** | 82.81 | COLLAPSED |
| k2_last | **10%** | 83.73 | COLLAPSED |
| k3_middle | **0%** | 96.57 | COLLAPSED |
| k3_last | **0%** | 88.05 | COLLAPSED |

**Run 2 (Feb 16)** — single commit, Bug 5 fixed:

| Config | Accuracy | Tokens/s | Notes |
|--------|----------|----------|-------|
| k1_first | 60% | 44.15 | No reuse (k=1) |
| k1_middle | 70% | 43.22 | No reuse (k=1) |
| k1_last | 70% | 43.70 | No reuse (k=1) |
| k2_first | 70% | 41.58 | OK |
| k2_middle | **10%** | 41.43 | Wrong answers |
| k2_last | **10%** | 42.04 | Wrong answers |
| k3_first | 60% | 38.65 | OK |
| k3_middle | **0%** | 38.64 | Wrong answers |
| k3_last | **0%** | 38.70 | Wrong answers |

## Phase 6: Investigation [DONE]

Root cause identified: cross-small-block cache contamination (Bug 6). The wrapper's
single-slot cache conflates 32-token full-block outputs with 8-token small-block outputs.
When reuse spans across different small blocks, wrong-position hidden states are returned.

- [x] A. Check `replace_position` propagation — confirmed it DOES propagate (not the bug)
- [x] Root cause found via code analysis — Bug 6 (cross-small-block cache contamination)
- [~] B-F superseded by Bug 6 discovery

> Full analysis, step-by-step trace, and fix design in
> [debugging.md](debugging.md#phase-6-investigation-resolved--mar-7-2026).

## Phase 7: Two-Tier Caching Fix [DONE — INVALIDATED BY BUG 8]

Fix Bug 6 by replacing the single-slot `layer_cache["last_output"]` with a dedicated
`layer_cache["full_block_output"]` that small-block recomputes cannot overwrite.

- [x] Implement two-tier caching in wrapper (recompute + reuse paths)
- [x] Update trim logic to use `full_block_output` instead of `last_output`
- [x] Run all 9 configs on GSM8K `--limit 10` — verified middle/last recovery
- [x] Fix Bug 7: `find_results_file()` in `log_utils.py` picked arbitrary results file

> Fix design in [debugging.md](debugging.md#fix-two-tier-caching).

> **INVALIDATED**: All limit_10 and limit_100 results from Phase 7 are invalid.
> Bug 8 (cache never populated) meant layer reuse had zero effect — all configs
> produced identical output. See Phase 9.

## Phase 8: Scaled Validation [DONE — INVALID]

Limit_100 sbatch runs completed but all 9 configs produced identical results
(80% accuracy, 31941 tokens, ~45 tok/s). Investigation revealed Bug 8: the cache
condition `tensor.shape[1] > current_input.shape[1]` was always False at the
layer level, so `full_block_output` was never stored and every "reuse" call
fell through to `original_forward()`.

- [x] Run all 9 configs on GSM8K `--limit 100` via sbatch (Mar 7)
- [x] Discover identical results → investigate → find Bug 8
- [x] Fix Bug 8: replace shape comparison with `step == 0`
- [x] Confirm fix: k1_first (538 tok, 34.3 tok/s, 100%) vs k3_middle (2267 tok, 58.4 tok/s, 50%)

> All limit_100 results invalid. See [debugging.md](debugging.md#bug-8-critical-fixed---mar-7-full_block_output-cache-never-populated).

## Phase 9: Re-validation with Bug 8 Fix [IN PROGRESS]

Bug 8 fix confirmed working on limit_2 (Mar 7). Debug prints removed, limit_10 complete,
limit_100 sbatch submitted (Mar 8).

- [x] Remove debug prints from `generation_functions.py`
- [x] Run all 9 configs on GSM8K `--limit 10` — verify differentiation across configs
- [x] Run all 9 configs on GSM8K `--limit 100` via sbatch (Mar 8)
- [ ] Verify implementation correctness (see Phase 10)
- [ ] If results look correct, run full GSM8K (all 9 configs, 1319 samples)
- [ ] Run full Minerva Math and IFEval
- [ ] Update results.md with final numbers

### Validation Results: GSM8K limit_10 (Mar 8)

| Config | Accuracy | Tokens | Time (s) | Tok/s |
|--------|----------|--------|----------|-------|
| k1_first | 70% | 3,540 | 81.9 | 43.2 |
| k1_middle | 70% | 3,540 | 97.9 | 36.2 |
| k1_last | 70% | 3,540 | 97.8 | 36.2 |
| k2_first | 50% | 3,380 | 94.9 | 35.6 |
| k2_middle | 50% | 3,412 | 95.7 | 35.7 |
| k2_last | 50% | 3,412 | 114.4 | 29.8 |
| k3_first | 30% | 6,806 | 133.3 | 51.0 |
| k3_middle | 30% | 6,806 | 123.4 | 55.2 |
| k3_last | 30% | 6,742 | 129.6 | 52.0 |

**Key findings:**
- k1 baselines identical (70% acc, 3540 tokens) — confirms k=1 disables reuse correctly
- Accuracy degrades with k: 70% → 50% (k=2) → 30% (k=3). Layer reuse is active.
- k3 shows ~1.4-1.5x throughput improvement (51-55 tok/s vs ~36 tok/s baseline)
- k2 shows NO throughput improvement (~30-36 tok/s, same or slower than baseline)
- No differentiation across subsets (all k2=50%, all k3=30%) — may be n=10 noise
- k3 generates ~2x tokens due to repetition (6,800 vs 3,500) — inflates wall-clock time
- Token counts suspiciously similar within k groups (k3_first = k3_middle = 6,806)

**Open question:** Does subset position matter? Need limit_100 to resolve (10% accuracy
resolution at n=10 is too coarse).

### Validation Results: GSM8K limit_100 (Mar 8)

| Config | Accuracy | Tokens | Time (s) | Tok/s | Notes |
|--------|----------|--------|----------|-------|-------|
| k1_first | 80% | 31,941 | 714.7 | 44.7 | |
| k1_middle | 80% | 31,941 | 726.4 | 44.0 | |
| k1_last | 80% | 31,941 | 715.5 | 44.6 | |
| k2_first | 53% | 37,193 | 1,967.1 | 18.9 | GPU contention (shared node) |
| k2_middle | 54% | 35,656 | 870.3 | 41.0 | |
| k2_last | 55% | 35,688 | 1,933.3 | 18.5 | GPU contention (shared node) |
| k3_first | 55% | 47,951 | 987.6 | 48.6 | |
| k3_middle | 53% | 45,934 | 964.2 | 47.6 | |
| k3_last | 53% | 45,646 | 958.1 | 47.6 | |

**Key findings:**
- k1 baselines identical (80% acc, 31,941 tokens) — correct
- k2 ≈ k3 accuracy (~53-55%) — degradation saturates, no further drop from k2 to k3
- k2_first/k2_last throughput invalid (GPU contention on shared node 004-25)
- k3 throughput ~48 tok/s vs ~44 tok/s baseline (~1.1x) — modest improvement
- k3 generates ~46% more tokens than baseline (repetition/inflation)
- **No subset differentiation at n=100** — first/middle/last within 2% for both k2 and k3

**Concern:** k2 ≈ k3 accuracy and zero subset differentiation are suspicious. Need to
verify implementation correctness before drawing conclusions. See Phase 10.

## Phase 10: Implementation Verification [TODO]

Before presenting results to Dr. Lin, verify the layer reuse implementation is correct.
Two suspicious findings need explanation: (1) k=2 and k=3 produce the same accuracy,
and (2) first/middle/last subsets show zero differentiation.

- [ ] **A. Per-sample output diff**: Compare actual generated text between k2_first vs
      k2_middle vs k2_last for the same input samples. If outputs are IDENTICAL across
      subsets, the subsets aren't actually targeting different layers (bug). If outputs
      differ but accuracy is similar, the effect is real but small.
- [ ] **B. Verify layer indices**: Confirm first=[1-12], middle=[8-19], last=[16-27]
      are actually being patched. Run a single sample and log which layers have modified
      `.forward()` methods.
- [ ] **C. Wrapper call verification**: Add lightweight instrumentation (single sample)
      to confirm: (a) wrapper is called, (b) reuse path is taken on the right steps,
      (c) cached tensor is actually different from freshly computed tensor.
- [ ] **D. Cache staleness check**: On a reuse step, compute the original forward AND
      return the cached value. Log the L2 distance between them. This measures how
      "stale" the cache actually is — if distance is ~0, layer reuse has no effect
      regardless of which layers are targeted.
- [ ] **E. k=2 vs k=3 output diff**: Compare generated text between k2_first and
      k3_first for the same inputs. If identical, something is masking the k difference.
