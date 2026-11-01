# AMENDMENTS to SPEC2.md

**A-1 · 2026-09-20 09:27 · before any computation.** The brief asks D1/D2 to be run on "the re-encoded model from run 1's
structure-penalty experiment". That model does not exist on disk: `analysis/z113_interrupt.py` never saved checkpoints, and
the two structure runs were stopped at steps 5000/5500 with only their logs as the record (`logs/p3_structure_s*.log`).
It is therefore REGENERATED: same script, same seeds (0 and 1), same schedule (λ 3→30→300, checks every 500 steps), 6000
steps, with a checkpoint saved at every check. Before D1 is applied, the regenerated run must itself reproduce the evasion
(≥1 checkpoint with 0 H¹-detected classes at ablation reliance > 0.9); if it does not, D2 is reported as "evasion did not
reproduce" and scored not-met. GPU non-determinism means the trajectory need not match run 1 step for step.

**A-2 · 2026-09-20 09:36 · before any D computation.** Two operational readings fixed, one secondary statistic added.
(i) "12 equal arcs with 50% overlap": arc centres at i·113/12, arc width 2·113/12, so adjacent arcs share half their length.
(ii) single-linkage at threshold t = connected components of the graph joining points closer than t, inside each cover element.
(iii) NMI between two fine partitions is biased towards 1 (if most Mapper nodes are singletons under both lenses, NMI → 1 for
any model). The pre-registered statistic and threshold stay as written; **adjusted mutual information (AMI)** of the same two
node assignments is reported beside every A as a secondary, together with the number of Mapper nodes. It does not change a verdict.
(note: only the clock stamps in SPEC2/AMENDMENTS/PAUSE_1 were corrected at 09:42 — my estimates ran ~1 h fast; no content changed. File mtimes are authoritative.)

**A-3 · 2026-09-20 10:05 · AFTER D1/D2 results — POST-HOC, cannot upgrade any verdict.** The pre-registered D1 threshold
(mean + 3σ of raw Mapper NMI on clean models) evaluates to 1.139, above the maximum of NMI: D1 can never fire, on any model
(shortcut models: 0/51, 0/51). Consequences, by the letter: D2 NOT MET; the D3 trigger never fires, so D3 by the letter is
the no-intervention run whose outcome is already on disk (decorrelated accuracy 0.012–0.016) → D3 NOT MET; D4 vacuous.
To still learn something, D3/D4 are run with the trigger replaced by the statistic SPEC2 named as "the floor": direct NMI(ŷ, z)
on the decorrelated probe, threshold = mean + 3σ over clean_s0/clean_s1 checkpoints = 0.537 (0 firings on cleanctrl_s2 and both
decoys; fires on 50/51 shortcut checkpoints). Everything from these runs is labelled POST-HOC TRIGGER.

**A-4 · 2026-09-20 10:41 · POST-HOC arm, cannot upgrade any verdict.** At step 4000 the pre-registered penalty (uniform z ≠ z')
has cleared the shortcut at λ=3 on 1 of 5 seeds; on the other four it escalated to λ=300 and flattened the model (reliance 0,
decorrelated accuracy 0.01). Hypothesis: uniform z' pairs a present hint with NULL only ~1 time in 114, and run 1 found
presence-invariance to be the part that matters. Added arm: run 1's penalty (z vs z+1 AND z vs NULL every sample) with the
run-2 direct trigger (which fires at step 500 on all seeds — earlier than H¹ on seeds 1, 2, 4), schedule R, 6000 steps, 5 seeds
→ `runs/r2_align_Rrun1pen_s*`. It also tests whether earlier detection rescues run 1's unexplained seed-1 failure.

**A-5 · 2026-09-20 10:52 · time-budget deviation.** With the A-4 arm sharing the GPU, a 500-step check takes ~10 min with ten
processes. Schedule G (reported-only in SPEC2) is cut to 6000 steps (D3 score only; no D4 extension) and is started by hand when
schedule R ends. Schedule R keeps its full 12000 steps.

**A-6 · 2026-09-20 11:21 · operational.** Schedule-R seeds 0, 1, 2, 4 were stopped at step 6500. All four were flattened at λ=300
(decorrelated accuracy 0.009–0.012 at step 6000 — D3 not met), and D4 is scored only on seeds that passed D3, so their extension
could not change any cell. Their histories to step 6500 are on disk. Seed 3 (passed D3) continues to 12000 for D4.

---
# Run 3 amendments (to SPEC3.md)

**A-7 · 2026-09-20 13:47 · before any S3/P1/P6.3 computation.** Degenerate guard for S3. If the off-diagonal distances are nearly
constant — std(D)/median(D) < tol (= 0.10) — the metric space is simplex-like and *every* permutation is a tolerance-isometry (an
untrained model's D is of this kind). Such a D is reported as `degenerate` with no discovered group; for every criterion it counts as
"order ≤ 2 / no 113-cycle". Also: the candidate search is capped at 20,000 anchor triples (deterministic order); if the cap is hit
the result is flagged `truncated`. Element order of a permutation = lcm of its cycle lengths; "cyclic subgroup of order ≥ m" =
some verified σ with element order ≥ m.

**A-8 · 2026-09-20 13:56 · instrument fix found on planted truth, before any network was touched.** (i) Resolution guard: a tolerance
larger than the point spacing cannot resolve group elements (a regular 113-gon gave 2,712 "isometries" at tol·median = 2.5 × its point
spacing). The verification tolerance is T = min(tol · median(D), 0.5 · median nearest-neighbour distance). (ii) The three-anchor
assignment is refined twice using all points as anchors (cost[k,m] = mean_i |D[σi, m] − D[i, k]|) before verification; without it noise
makes the first assignment swap near-equivalent points and true isometries fail verification (planted noisy Z113: 5 of 226 found).

**A-9 · 2026-09-20 14:14 · AFTER P1.1 results — POST-HOC, cannot upgrade a verdict.** By the letter the group is not discovered
(orders 1, 1, 2, 13, 5). Diagnostic: a true translation changes D^g by 3% of the median on average on grokked models (memoriser 22%,
random permutation 21–33%) but by 14–42% in the worst entry, above the registered max-norm tolerance (10%). Added to S3 and reported
beside every by-the-letter order: `order_mean` = number of candidates with MEAN |D_σ − D| < tol·median (same candidates, same tol),
whether a full cycle is among them, and δ* = the smallest mean defect of any non-identity candidate, in units of the median.

**A-10 · 2026-09-20 14:17 · POST-HOC refinement of A-9 (first checkpoint of the first sweep exposed it).** An untrained model's D is not
flagged degenerate by A-7 (spread ≥ 0.10) yet has so little structure that ANY permutation changes it by < 1% on average: 804 candidates
passed the A-9 mean-norm test at step 0. The mean defect is therefore measured against chance: δ_rand = mean defect of 10 random
permutations; `order_rel` = number of candidates with mean defect < 0.25·δ_rand; R* = δ*/δ_rand (1 = no better than a random relabelling).
A-9's `order_mean` is kept in the output for the record and is not used.

**A-11 · 2026-09-20 14:48 · time-budget deviation.** A full 51-checkpoint sweep costs 40–70 min per model on CPU under the current load.
The UNSEEN fresh-seed sweeps for P6.3 (r3_z113_*_s3, s4) are run on every checkpoint within ±4 of the grokking checkpoint and every 5th
checkpoint elsewhere (grokking located from test accuracy in the training history, as always). Fresh ResNet seeds: training under GPU
contention is ~20× slower than in run 1; P6.4 is scored on whichever fresh ResNet models have finished by 15:50 and says which.

**A-12 · 2026-09-20 15:10 · P3.3 abandoned, NOT SCORED.** Under GPU contention the penalised interrupt ran at 4.5–5.3 s/step (five seeds
stacked); step 6000 was unreachable before 16:38. `wb-r3-bz` was stopped at step ~700. Recorded as NOT SCORED, not as a fail or a pass. What
exists: the trigger fired at step 0 on 5/5 seeds (b_z = 1/3 at random init > threshold 0.17), so the "interrupt" was an always-on λ=3 penalty;
at step 500 decorrelated accuracy was 0.943 / 0.991 / 0.951 / 0.703 / 1.000.

---
# Run 4 amendments (to SPEC4.md)

**A-13 · 2026-09-20 16:14 · operational, before any analysis.** The GPU is software power-capped by the system (510 of 3090 MHz, 30 W,
throttle reason 0x4): total Z113 throughput is ~50–70 steps/s however many processes share it, and a concurrent ResNet job slowed six Z113
runs threefold. Changes: (i) the ResNet specimens are restarted AFTER Z113 wave 1 and run alone with wave 2; (ii) wave-2 lengths: decoy 5, 6
keep 20,000 steps (run-2 decoys grokked at 8.5–11.5k); shortcut 5, 6 and aonly_5 are cut to 8,000 steps (the shortcut is learned by step 500;
only final checkpoints of these three are used, for P3.1 and the tower). Clean 5–8 and memorisers 5, 6 keep 25,000 steps. No threshold changes.

# RUN 5

**A-14 · 2026-09-21 13:16 · clerical, no threshold touched.** SPEC5.md was saved at 13:12:51 and the specimens launched at 13:14. The header's
"written 13:11–13:20" was an estimate typed before the file was saved; it was corrected to the file's mtime at 13:16, after the launch. No other
line of SPEC5.md was changed. Compute order: the Z113 sweeps and final-model instruments run as soon as the Z113 trainers exit (CPU), while the
ResNets are still training; none of their output is read or scored until the sanity gate and the augmented ResNets' shift consistency are
written into PAUSE5_1.md, as the brief's PAUSE 1 requires.

# RUN 6

**A-15 · 2026-09-21 23:02 · F.2 clause shape, fixed before any Run-6 model is analysed.** A smoke test of the F pipeline on a *Run-5* model
(r5_z113_clean_s9, already seen) showed what Runs 1 and 4 had already recorded: the softmax margin's Spearman with the failure radius is
POSITIVE (+0.52 there; +0.21–0.30 in Run 1, +0.93 on ResNet), because a larger margin means a larger radius. The brief's "Spearman(ε, radius)
more negative than softmax margin's" is therefore satisfied by any ε with ρ < +0.5, which is not what it means. Registered reading, applied
to every Run-6 model: **F.2 passes iff |ρ(ε, radius)| > |ρ(margin, radius)| with ρ(ε, radius) < 0**, on 5/5 clean models. The brief's literal
reading is also reported. No Run-6 model had been trained to its final checkpoint or analysed when this was written.
