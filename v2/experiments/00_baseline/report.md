# Report: 00_baseline — Layer Reuse in Block Diffusion

**Experiment**: 00_baseline
**Date**: March 8, 2026
**Author**: Murilo

## Summary

Layer reuse in Fast-dLLM v2 has a fundamental design limitation: only the **first
patched layer** in each subset has meaningful cache staleness. The remaining 11 layers
produce cached outputs that are near-identical to fresh computation (relative L2
distance ~0.04%). As a result, **which layers are reused doesn't matter** — only the
reuse frequency (k) has a measurable effect on output quality.

## Experiment Design

- **Model**: Efficient-Large-Model/Fast_dLLM_v2_7B (Qwen2.5, 28 layers)
- **Grid**: reuse_k={1,2,3} x layer_subset={first,middle,last} (9 configs)
- **Benchmark**: GSM8K (0-shot, flexible-extract exact_match)
- **GPU**: Quadro RTX 6000 (SM 7.5, float16)

Layer subsets (12 layers each):
- first: layers 1-12
- middle: layers 8-19
- last: layers 16-27

## Results

### GSM8K Accuracy (limit_100, n=100)

| Subset | k=1 (no reuse) | k=2 | k=3 |
|--------|:-:|:-:|:-:|
| first  | 80% | 53% | 55% |
| middle | 80% | 54% | 53% |
| last   | 80% | 55% | 53% |

- k=1 baselines are identical across subsets (expected — no reuse active)
- k=2 and k=3 produce the same accuracy within noise (~53-55%)
- No differentiation across subsets for the same k value

### Throughput (limit_100)

| Subset | k=1 | k=2 | k=3 |
|--------|:-:|:-:|:-:|
| first  | 44.7 tok/s | 18.9* | 48.6 |
| middle | 44.0 | 41.0 | 47.6 |
| last   | 44.6 | 18.5* | 47.6 |

*GPU contention (two jobs on same node) — real k=2 throughput is ~41 tok/s.

- k=3 shows ~1.1x throughput improvement (~48 vs ~44 tok/s)
- k=2 shows no throughput improvement
- k=3 generates ~46% more tokens due to repetitive output, making wall-clock
  time longer despite higher tokens/second

### Per-Sample Output Comparison (limit_100)

| Comparison | Identical samples | Different samples |
|---|:-:|:-:|
| **k2_first vs k2_middle** | **95/100** | 5/100 |
| **k2_first vs k2_last** | **97/100** | 3/100 |
| k1_first vs k2_first | 40/100 | 60/100 |
| k2_first vs k3_first | 42/100 | 58/100 |

Different subsets produce nearly identical output (95-98%). Different k values
produce meaningfully different output (58-65%).

## Root Cause Analysis

### Why subset position doesn't matter

Instrumented wrapper runs (`verify_wrapper.py`) measured L2 distance between cached
and freshly-computed hidden states on every reuse call:

| Layer position in subset | Relative L2 distance |
|---|---|
| **First patched layer** | **~0.68** (68% different) |
| Layers 2-12 | ~0.0004 (0.04% different) |

**Only 1 of 12 patched layers has meaningful cache staleness.** The remaining 11
produce outputs that are near-identical to fresh computation.

This happens because:

1. The full-block cache stores 32-token outputs from the full-block forward
2. On reuse, we slice the correct 8-token window using `replace_position`
3. For layers 2-12, the INPUT is the previous layer's cached output — the same
   value in both the cached and fresh paths
4. Since `original_forward(same_input)` with the same block cache K/V produces
   the same output, cached ≈ fresh trivially
5. Only the first patched layer sees different input: its predecessor is unpatched
   and runs fresh (seeing current denoised tokens), while the cache was built
   before this round of denoising

### Why k=2 and k=3 produce the same accuracy

Two factors:

1. **Quality degradation comes from a single layer**: Whether that layer is stale
   every other step (k=2) or 2/3 of steps (k=3), the damage saturates quickly.
   The model can tolerate one stale layer regardless of how stale it is.

2. **Metric is forgiving**: `flexible-extract` finds the numerical answer anywhere
   in the output. k=3 produces correct answers early, then degenerates into
   repetition. The metric doesn't distinguish "correct answer + garbage" from
   "correct answer only".

### Subset overlap

The three subsets share layers due to the 12/28 ratio:

```
first ∩ middle: [8, 9, 10, 11, 12]  — 5 shared layers (42%)
middle ∩ last:  [16, 17, 18, 19]    — 4 shared layers (33%)
```

This is a contributing factor but not the primary cause (the single-layer staleness
effect is the main explanation).

## Implications

1. **The current layer reuse design is fundamentally limited.** Caching full-block
   outputs and slicing them means only the boundary layer (first patched) has
   meaningful staleness. Increasing the number of patched layers from 12 to 20
   would not change the result.

2. **To make layer position matter**, the caching mechanism would need to ensure
   all patched layers see stale values. Possible approaches:
   - Cache per-layer inputs rather than outputs
   - Invalidate block cache entries for patched layers (force stale K/V)
   - Cache at a different granularity (per-block rather than per-layer)

3. **The throughput-accuracy tradeoff is poor.** k=3 gives ~1.1x speedup at 26%
   accuracy loss (80% → ~54%). k=2 gives no speedup at all.

4. **Non-overlapping subsets** (e.g., first=[1-9], middle=[10-18], last=[19-27])
   would be a cleaner experimental design, but wouldn't change the fundamental
   single-layer staleness issue.

## Bug History

Eight bugs were found and fixed during this experiment (see [debugging.md](debugging.md)):

| Bug | Severity | Impact |
|-----|----------|--------|
| Bug 1 | CRITICAL | Reuse path called original_forward() — zero speedup |
| Bug 2 | MODERATE | should_recompute conflated block cache + layer reuse |
| Bug 3 | MODERATE | "first" subset patched 11 layers, not 12 |
| Bug 4 | MINOR | Layer caches never trimmed during batch trim |
| Bug 5 | MODERATE | Stale cache from disabled wrappers during full-block |
| Bug 6 | MODERATE | Cross-small-block cache contamination |
| Bug 7 | MINOR | find_results_file() picked arbitrary results file |
| Bug 8 | CRITICAL | Cache condition always False — reuse never activated |

All results prior to Bug 8 fix (Mar 7, 2026) are invalid.

## Files

- Experiment code: `generation_functions.py`
- Verification script: `verify_wrapper.py`
- Full debugging history: `debugging.md`
- Task tracker: `tasks.md`
- Experiment design: `proposal.md`
- Results: `results/gsm8k_limit_100/*/summary.json`
