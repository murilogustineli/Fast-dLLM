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

## Phase 7: Two-Tier Caching Fix [DONE]

Fix Bug 6 by replacing the single-slot `layer_cache["last_output"]` with a dedicated
`layer_cache["full_block_output"]` that small-block recomputes cannot overwrite.

- [x] Implement two-tier caching in wrapper (recompute + reuse paths)
- [x] Update trim logic to use `full_block_output` instead of `last_output`
- [x] Run all 9 configs on GSM8K `--limit 10` — verified middle/last recovery
- [x] Fix Bug 7: `find_results_file()` in `log_utils.py` picked arbitrary results file

> Fix design in [debugging.md](debugging.md#fix-two-tier-caching).

### Validation Results: GSM8K limit_10 (Mar 7, Post-Fix)

All 9 configs produce 70% accuracy — middle/last collapse fully resolved.

| Config | Accuracy | Notes |
|--------|----------|-------|
| k1_first | 70% | No reuse (k=1) |
| k1_middle | 70% | No reuse (k=1) |
| k1_last | 70% | No reuse (k=1) |
| k2_first | 70% | OK |
| k2_middle | 70% | RECOVERED (was 10%) |
| k2_last | 70% | RECOVERED (was 10%) |
| k3_first | 70% | OK |
| k3_middle | 70% | RECOVERED (was 0%) |
| k3_last | 70% | RECOVERED (was 0%) |

> Note: All configs showing identical accuracy on limit_10 is expected — 10 samples
> is too few to differentiate. Full runs needed to see accuracy vs throughput tradeoff.

## Phase 8: Scaled Validation & Full Runs [IN PROGRESS]

Limit_10 confirmed the fix works but all configs show identical accuracy (10 samples
too few to differentiate). Running limit_100 via sbatch to get meaningful signal before
committing to full runs.

- [~] Run all 9 configs on GSM8K `--limit 100` via sbatch (submitted Mar 7)
- [ ] Analyze limit_100 results — confirm accuracy differentiation across k and subset
- [ ] Run full GSM8K (all 9 configs, 1319 samples)
- [ ] Run full Minerva Math (all 9 configs)
- [ ] Run full IFEval (all 9 configs)
- [ ] Update results.md with final numbers
