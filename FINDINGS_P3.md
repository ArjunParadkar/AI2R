# FINDINGS P3 — spurious structure as symmetry breaking (breaking score b_z)

Block: SPEC3 S5 + P3.1–P3.4. Work ran 13:41–15:03 CDT on 2026-09-20. P3.1 is scored, P3.2 and P3.4 are done, and P3.3 is NOT SCORED: training reached step 500 of 12000.

## In plain English

b_z is the share of the input–output Jacobian's squared norm that falls in the hint slot. Like the run-2 floor, b_z must be told which input slot is the nuisance, and it discovers nothing. It cleanly separates trained shortcut models from trained clean or decoy models (≈0.97 against ≈0.0004–0.01). It fires on all 122 checkpoints where the run-2 floor fires, including all 7 H¹-evasion checkpoints. P3.1 still fails by the letter on two of its three clauses:

- The pooled rank correlation with the floor is 0.68, not above 0.9.
- b_z false-alarms on 39 of 102 decoy checkpoints and on 3 of 153 clean checkpoints.

The decoy firings cover the whole memorisation phase, and the 3 clean firings are all step 0. b_z measures sensitivity to the hint slot, not use of the hint as a shortcut. A randomly initialised network has b_z = 1/3 by symmetry of the three slots, and a network that memorises keys on every token, including a meaningless hint.

P3.3 could not be run inside the budget on the shared GPU. A penalised step cost 1.0 s for a single seed under contention, which is about 3.4 h per seed. Only the step-0 and step-500 checks exist.

## Contradicting the brief's framing (stated first)

1. **b_z at random init is 1/3 (0.3328–0.3392 on the fourteen step-0 models measured), and the registered threshold is 0.1705.**
   - The threshold is mean + 3σ over the 102 clean_s0/s1 checkpoints (mean 0.0292, population sd 0.0471). Step 0 lies inside that calibration set.
   - Every model therefore fires at step 0, including the three clean runs. "Zero firings on the 153 clean checkpoints" was unreachable by construction.
   - In P3.3 the trigger fired at step 0 on 5/5 seeds. The "interrupt" became an always-on λ=3 penalty from initialisation. It was run that way, by the letter.
2. **The decoy null fails, for a functional reason.**
   - b_z is 0.44–0.48 on the decoys at step 500 and stays above threshold while the decoy memorises (train accuracy 1.00, test accuracy 0.07–0.46).
   - It collapses at grokking: decoy_s0 goes 0.204 → 0.077 → 0.008 over steps 10500/11000/11500 while test accuracy goes 0.44 → 0.77 → 1.00. decoy_s1 goes 0.187 → 0.030 over steps 8000/8500.
   - Finals are 0.0004 and 0.0005.
   - Post-hoc reading: a memorising network depends on the hint token as part of each example's key. That is a real broken symmetry, but it is not the planted shortcut. "Symmetry breaking as the definition of spurious structure" therefore conflates memorisation with shortcut use.
3. **b_z is scale-free and infinitesimal, and both properties cause errors.**
   - Scale-free: r2_reencoded_s1 at step 6000 has floor 0.000 and reliance 0.000 (a collapsed model), yet b_z = 0.943.
   - Infinitesimal: P3.3 seed 3 at step 500 under the penalty has reliance 0.354 and decorrelated accuracy 0.703, yet b_z = 0.0002. A finite dependence on the hint token exists while the local derivative is flat. This is one checkpoint, and it points the same way as the run-2 evasion lesson.
4. **The P3.4 "earliest checkpoint with hint share of trace(g) > 10%" is step 0 for all nine runs.** Item 1 makes it uninformative.

## P3.1 — criteria by the letter (383 checkpoints, nine runs)

| Criterion | Measured | Verdict |
|---|---|---|
| Spearman(b_z, floor) > 0.9, pooled | 0.682 (ratio-of-means variant 0.678); planted runs only 0.425; null runs only 0.100 | **NOT MET** |
| b_z fires wherever the floor fires | 122/122, 0 misses | **MET** |
| All 7 evasion-window checkpoints fire | 7/7; re_s0 steps 3500/4000/4500/5500/6000 (b_z 0.969–0.981), re_s1 steps 500 (0.802) and 4000 (0.945) | **MET** |
| Zero firings on 153 clean + 102 decoy | clean 3/153; decoy 39/102 | **NOT MET** |

- False-positive detail, clean: calibration seeds 2/102 (step 0 of each run) and held-out cleanctrl_s2 1/51 (step 0).
- False-positive detail, decoy: decoy_s0 fires on 22 checkpoints (steps 0–10500) and decoy_s1 on 17 (steps 0–8000).
- The floor fires on 0 of those 255 clean and decoy checkpoints.
- **P3.1 PASS: NO.**
- b_z fires with the floor silent 48 times:
  - step 0 of all nine runs;
  - the 37 later decoy checkpoints;
  - re_s1 step 3500 (reliance 0.251) and re_s1 step 6000 (reliance 0.000).
- Post-hoc descriptive only, no new threshold: the minimum b_z where the floor fires is 0.747, and only 2 floor-silent checkpoints exceed that. The clean calibration maximum excluding step 0 is 0.086.
- Ranges per run:

| Run | b_z range | b_z final |
|---|---|---|
| shortcut s0/s1 | mean 0.935 / 0.964 | 0.967 / 0.989 |
| re-encoded s0/s1 | — | 0.975 / 0.943 |
| clean s0/s1/ctrl_s2 | — | 0.007 / 0.008 / 0.011 |

## P3.2 — per-layer b_z on finals (mean of per-input ratios; no threshold)

| Final | resid_mid | resid_post | logits |
|---|---|---|---|
| shortcut s0 | 0.964 | 0.965 | 0.967 |
| shortcut s1 | 0.989 | 0.989 | 0.989 |
| re-encoded s0 | 0.983 | 0.975 | 0.975 |
| re-encoded s1 | 0.932 | 0.878 | 0.943 |
| decoy s0 | 0.0004 | 0.0004 | 0.0004 |
| decoy s1 | 0.0005 | 0.0006 | 0.0005 |
| clean s0 | 0.0066 | 0.0079 | 0.0073 |
| clean s1 | 0.0100 | 0.0094 | 0.0080 |
| cleanctrl_s2 | 0.0097 | 0.0115 | 0.0106 |

- The breaking is already there after attention and is flat across depth, so there is no informative depth profile here.
- The grokked decoy is more invariant than the clean model, because the clean model never saw non-NULL hints.
- The decoy null in the positive sense: the hint is present in the input and unused, and b_z ≈ 0 at convergence.

## P3.4 (exploratory) — eigenvectors of E_x[g] on the 384-d slot space

- shortcut_s0 final:
  - Hint share of trace 0.953.
  - Top-10 hint mass: 1.00 1.00 1.00 1.00 1.00 0.97 0.95 0.85 0.01 0.88.
  - First eigenvector with > 50% hint mass: rank 1.
- shortcut_s1 final:
  - Hint share of trace 0.960.
  - Top-10 hint mass: eight at 1.00, then 0.00 and 0.00.
  - First eigenvector with > 50% hint mass: rank 1.
- Step 500:
  - Hint share 0.792 (s0) and 0.787 (s1).
  - First > 50% rank is 1 for both.
  - Hint masses are 0.49–0.95, with one or two (a,b)-dominated vectors among the top 10 (ranks 4; 3 and 9).
- Earliest checkpoint with hint share > 10%: step 0 on all nine runs (uninformative). Checkpoints above 10%:
  - clean 2/51, 1/51, 1/51;
  - decoy 22/51 and 17/51;
  - shortcut 51/51;
  - re-encoded 13/13.

## P3.3 — NOT SCORED

- The trigger fired at step 0 on 5/5 seeds (b_z 0.333–0.339 against 0.1705). λ=3 has been on since step 0 and was never escalated.
- The step-500 check is the only later check:

| Seed | Decorrelated accuracy | In-distribution test accuracy | Reliance | b_z |
|---|---|---|---|---|
| s0 | 0.943 | 0.976 | 0.031 | 0.0000 |
| s1 | 0.991 | 0.994 | 0.000 | 0.0002 |
| s2 | 0.951 | 0.984 | 0.014 | 0.0000 |
| s3 | 0.703 | 0.883 | 0.354 | 0.0002 |
| s4 | 1.000 | 1.000 | 0.000 | 0.0001 |

- No b_z firing after step 0.
- The three pass criteria cannot be evaluated: none of decorrelated accuracy at step 6000, b_z over (6000, 12000], or test accuracy at step 12000 exists yet.
- Rate: the five-seed stacked run takes 4.5–5.3 s/step. A single penalised seed alone measured 1.0 s/step under the GPU contention, against 0.40 s/step plain. Twelve thousand steps would take about 15 h.
- The unit `wb-r3-bz` was STOPPED by the main session at 15:10 (AMENDMENTS A-12): it could not reach step 6000 before the deadline and was starving the other blocks of GPU.
  - It cannot reach step 6000 before the 16:38 deadline (roughly step 1700 by then).
  - It saves state at every 500-step check and resumes from the last check if restarted.
  - To stop it: `systemctl --user stop wb-r3-bz`. The main session should decide.
  - To re-score: `python analysis/run3_blockP3.py --only score33`.

## What is claimable

- A Jacobian block-norm share, told the nuisance slot, fires wherever the functional floor fires in these nine runs (122/122). That includes the 7 checkpoints where the H¹ shape monitor was evaded.
- b_z is not a shortcut detector:
  - it fires on memorisation (decoy) and on random init;
  - it ignores output scale (the collapsed model);
  - being infinitesimal, it can be near zero while finite hint-dependence remains (seed 3, one checkpoint).
- The run-2 floor remains the better-behaved statistic, with 0 false firings on the same 255 clean and decoy checkpoints.
- Neighbouring work: LieGG (Moskalev, Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022) and the Lie-symmetry-discovery line extract symmetry generators from trained networks. b_z is much weaker than that: it is a one-number test of invariance to a *pre-specified* input block.

## Timings

| Item | Time |
|---|---|
| Planted tests (6 passed) | 2.8 s |
| b_z per checkpoint | ~7 s (2000 probes; contended GPU or 2-thread CPU) |
| P3.1 clean_s0 + clean_s1 on GPU | 1355 s, including about 10 min of host thrash |
| P3.1 other runs on CPU | 355–975 s per worker |
| P3.2 | 338 s |
| P3.4 | 27 s |
| P3.3 | 2369 s for 500 steps |

## Deviations

1. P3.3 trains the 5 seeds in ONE process, stacked with `torch.func.vmap`, instead of 5 processes.
   - Reason: 5 CUDA processes would use about 6.75 GB of RSS, over the 6 GB cap, and the host hit load 226 with swap full.
   - The seeds stay mathematically independent: the loss is a sum of per-seed terms, AdamW is elementwise, and each seed has its own data, init, λ and penalty draws.
   - The penalty is the literal run-1 `invariance_penalty`, called through a shim.
   - The backward pass is split into separate passes (same gradient) to bound GPU memory.
   - The step is replayed as a CUDA graph with AdamW `capturable=True`. Steps 0–2 ran eagerly.
   - Penalty RNG draws differ from a per-process run.
2. During the host thrash I stopped six of my own processes by explicit PID (five CPU workers and a smoke test), and later my own GPU P3.1 process. I touched nothing else.
3. Two relaunches of `wb-r3-bz`, which is my unit.
   - The first died of CUDA OOM at step 0.
   - I stopped the second, an eager run, before it reached the step-500 check (it ran at under 0.75 steps/s), to switch to the CUDA-graph step.
   - Failed logs are in `logs/_r3_bz_failed/`.
4. P3.1 devices: clean_s0/s1, and hence the threshold, were computed on GPU (batch 250). The other seven runs and the P3.3 checks were computed on CPU (batch 50). The differences are float32 rounding.
5. Computation and scoring choices:
   - σ is the population sd (ddof=0), as in run 2.
   - Inputs with J=0 count as ratio 0 (none occurred).
   - The P3.4 "shortcut finals" are z113_shortcut_s0/s1 only.
   - The floor values are read from `blockD12.json`, not recomputed.
6. Figure details:
   - The small panel also shows clean_s0.
   - The decoy is labelled "hint independent of answer".

## Files (all under `/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight/`)

- `whiteboxtopo/symmetry_breaking.py`
- `tests/test_run3_breaking.py`
- `analysis/run3_blockP3.py` (stages: p31 [--runs], merge31, score31, p32, p34, fig, score33)
- `analysis/run3_blockP3_train.py`
- `scripts/run3_bz_interrupt.sh`
- `results/run3/blockP3.json`
- `results/run3/p31_parts/`
- `figures/run3_bz_vs_floor.png`
- `runs/r3_bz_s{0..4}/history.json` (plus `runs/r3_bz_s0/stacked_state.pt`)
- `logs/r3_bz_s{0..4}.log`, `logs/r3_bz_stacked.log`
- `logs/r3_blockP3_p31*.log`, `logs/r3_blockP3_score31.log`

*(Written by the Block-P3 agent and saved verbatim by the main session, which added only the note that `wb-r3-bz` was stopped.)*
