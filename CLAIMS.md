# CLAIMS — what the ten-hour run found, number by number

Spec: `docs/WHITEBOX_Final_10hr_Specification.pdf`. Every row gives the pre-registered criterion (§6.3), the number,
its null, the file it lives in, and a verdict **by the letter**. Anything decided after seeing data is marked
**POST-HOC** and never replaces the pre-registered number. The word "complete" is retired: no system passes every cell.

Cells marked ⏳ were still computing when this was last edited (see RUNLOG.md for the time).

## Scorecard (by the letter)

| cell | criterion | result | verdict |
|---|---|---|---|
| P1 Z113 | ≥3 of top-5 frequencies at r>0.9; answer-fibers β₁=1 | frequencies: 5/5 at r = 0.995, 0.996, 0.995, 0.971, 0.906 — by the **symmetry-informed** isotypic route; unsupervised Rips: 70–207 unnamed classes, best r ≈ 0.2. Fibers: 0 certified circles in 20 fibers, raw β₁ = 3–9 — **and that is the right answer** (finding 1b) | frequencies PASS with the group action supplied, FAIL unsupervised; fibers FAIL (the pre-registered ground truth was wrong) |
| P1 ring | one H¹ class; r > 0.99 | Euclidean Rips: r = 0.80 / 0.75 / 0.75. **POST-HOC** pullback-geodesic Rips: one class (lifetime ratio 17–20), r = 0.979 / 0.993 / 0.993; network's own output r = 0.983 / 0.993 / 0.989; shuffle null 0.01 | FAIL as pre-registered; 2/3 seeds pass post-hoc, at the model's own ceiling |
| P1 rot-MNIST | circle in ≥7/10 digit fibers, r>0.9 | penultimate/fc1: 0/10. conv1, conv2: 10/10 — but raw pixels 10/10 and an untrained net 10/10 | FAIL (early-layer circle is inherited from the input, not learned) |
| P6 Z113 | G jumps >0.4 within ±3 ckpts of the test jump; scrambled G<0.2 | jump 0.06 / 0.16 (seeds 0,1); 0.81 on the 50%-data control, which groks at ckpt 1 (jump is from the untrained init). Scrambled G = 0.59. **POST-HOC** interior-only G: jump 0.17 / 0.35 / 0.59, scrambled 0.076 | FAIL |
| P6 ResNet | clean G>0.9; scrambled <0.3; shuffle null → chance | clean 1.00; scrambled 0.89; null 0.09 | FAIL (scrambled) |
| P7 Z113 | births within ±3 ckpts of grokking; cross-seed order τ>0.6 | births within ±3: 4/5, 5/5, 3/4. Order: **not computable** — seeds learn different frequencies (s0 {1,39,42,51,56}, s1 {3,6,26,50,51}, ctrl {5,17,31,40}) | births PASS-ish (12/14); order criterion ill-posed |
| P2 | path-product/IG r>0.95; margin CV<0.2 | r = 1.0000 on 100% of inputs; CV = 0.072 / 0.101 / 0.084 (Euclidean margin CV 0.13) | PASS on both numeric criteria (caveat: the constant-pattern segment is 1.5% of the first edge). The spec's *ground truth* — the boundary geodesic changes a+b by one residue — is false: 0%, 0%, 1.3% of 300 geodesics per seed end at c±1 (finding 1c) |
| P3 | onset within 5% of training; 0 FP on clean; interrupted within 2 pts of clean; beats no-intervention by >10 pts worst-group | **Detection** (5 seeds): onset at step 500 on every seed, first flag at 500 / 1000 / 1500 / 500 / 1000 → 0–4% late; 0 flags in 153 clean checkpoints and 0 in 52 checkpoints of the harder **decoy** null (hint present as often, uninformative; ablation changes 0.0% of predictions) vs 25/26 on each shortcut model. **Intervention** (pre-registered schedule): 4 of 5 seeds end at decorrelated / worst-group accuracy 0.998–1.000 after ONE round at λ=3 (clean model 1.00; no-intervention 0.016 / 0.012; SPARE-style reweighting 0.014 / 0.013). Seed 1 never responds and escalation to λ=300 collapses it (test 0.26); a POST-HOC gentler schedule (×3, checks every 250 steps) collapses it too. Cause unknown | detection PASS; intervention PASS on 4/5 seeds, with the caveats of finding 6 |
| P4 | AUC > softmax and > Euclid kNN on both OOD sets | Z113 (3 seeds × 4 sets): d_g beats softmax on 0/12. Ring (3 seeds × 3 off-ring sets): 0.93–0.999 vs baselines 0.99–1.00. ResNet: SVHN 0.63 vs softmax 0.83; noise 0.72 vs Euclid 0.89. GPT-2: 0.40 / 0.38 (below chance) vs softmax 0.92 / 0.81 | FAIL everywhere |
| P5 | Spearman < −0.5, p<0.01, beats softmax margin | Z113 (curvature bug fixed, representative probe): −0.12 / −0.14 / −0.23, p<1e-4, null ≈ 0; softmax margin +0.21 / +0.30 / +0.30; on the fiber probe +0.02 (p=0.57). ResNet +0.32 (**wrong sign**; pre-fix +0.28), margin +0.93. GPT-2 +0.04 (null +0.05), margin +0.72 | FAIL |
| Speed | open() < 60 s per layer | `open_model`, quiet machine: **4.6 s** (Z113 resid_mid, 1,000-point probe: fit 0.6, null band 1.6, P1 1.1, symmetry route 0.06, P6 0.2, P4 0.1, P5 1.0) and **8.3 s** (ResNet-18 penultimate, 1,500 points); 7.1 s / 12.9 s with the GPU shared by six training jobs. Probe extraction 0.9 s / 2.0 s extra. `logs/demo_transcript.txt`, `results/demo_timings.json` | PASS |

## What the run actually established

1. **The pullback metric is the right metric for cohomology.** The ring RNN's hidden state is only ~50% heading
   (R² of the state on heading harmonics: 0.50–0.55; velocity history and its interaction with heading bring it to 0.78).
   In the Euclidean metric that is a thick blob and the H¹ coordinate tracks heading at 0.75–0.80. Measured by the
   geodesic distance of the g-weighted graph the same cloud is one clean circle, with one class 17–20× longer-lived than
   the next, whose coordinate tracks heading exactly as well as the network's own read-out does (0.979/0.993/0.993 vs
   0.983/0.993/0.989). The metric quotients out what the output does not read. `results/ring_pullback.json`;
   `p1_inventory(..., metric='g')`.
1b. **Z113 answer-fibers are intervals, and the network told us why.** The spec expected each answer-fiber {(a, c−a)} to be
   a circle. It is not: the model is commutative at resid_mid — ‖h(a,c−a) − h(c−a,a)‖ is 0.3–0.8% of the fiber's radius on
   three seeds — so inside a fiber every frequency carries e^{ika} + e^{ik(c−a)} = 2e^{ikc/2}·cos(k(a − c/2)), a standing
   wave. The fiber's image in each frequency plane has axis ratio 0.016–0.025 (a segment); the same planes on the b=0
   slice, where nothing folds the orbit, give 0.996–0.999 (a circle). The fiber is S¹/(a ~ c−a): an interval.
   `results/z113_fiber_symmetry.json`, `results/z113_fibers.json`.
1c. **Where the decision boundary is, is set by the frequencies P1 found.** The geodesic from an input to the nearest
   differently-answered point never goes to the neighbouring residue. It goes to the residue shifts Δ that almost preserve
   every key-frequency phase at once, argmin_Δ Σ_k var_k·(1 − cos 2πkΔ/113) — computed from the P1 inventory alone. The top-3
   predicted shifts (e.g. Δ = 38, 4, 40 for frequencies {3, 26, 50, 51, 6}) contain 52% / 43% / 25% of the geodesic
   destinations and 42% / 67% / 47% of the logit runners-up on the three seeds; chance is 5%. The geodesic's destination
   and the logit runner-up agree only 8–23% of the time: "nearest along the manifold" and "second-largest logit" are
   different notions of the competing answer. `results/z113_p2_boundary.json`.
2. **Geodesic closure is a real structural signal at grokking, but it is not a step.** Computed from training points
   only, G_interior sits at 0.06–0.09 through 4,000–8,000 steps of perfect train accuracy, starts rising when the
   frequency circles are born, and keeps rising for ~10 checkpoints after test accuracy has saturated. The scrambled
   model never leaves 0.07. After grokking it oscillates with the optimiser's loss spikes (Spearman with train loss
   −0.51 / −0.54 / −0.60) while accuracy reads 1.00 — it sees a disturbance of the representation that accuracy does
   not. `figures/grokking.png`, `results/p6_p7_score.json`.
3. **The spec's argument that G "cannot be faked" is wrong at a late layer.** A ResNet that memorised random labels
   scores G = 0.89 at the penultimate layer: memorisation also makes training points class-contiguous there. With
   unlabeled held-out inputs added as path nodes (POST-HOC, no held-out label used) the clean net reads 0.99–1.00 at
   every depth — and the scrambled net still reads 0.54 / 0.55 / 0.60 / 0.86 / 0.88 from layer1 to the penultimate (label-permute
   null 0.10). The amendment does not rescue the criterion, and there is a reason: a metric derived from the output makes
   "near" mean "same output", so a geodesic between two same-output points running through same-output points is close to a
   tautology. The two networks do separate — most at early layers, least at the last — and the fiber's β₀ separates them
   more cleanly than G does (1.0–1.2 vs 7.9–9.1). `results/resnet_depth.json`.
4. **P4's definition fails for a structural reason.** d_g pushes the off-manifold displacement through the Jacobian,
   which deletes exactly the directions the head ignores — and an OOD input is displaced mostly along those. On GPT-2
   the score is *below chance* (0.40, 0.38): OOD points look closer than in-distribution ones. Measuring the same
   normal displacement in activation space (POST-HOC `tubular_distance`) helps (GPT-2 0.79 / 0.61) but still loses to
   softmax confidence and entropy everywhere it was tried.
5. **P5 does not hold.** After fixing a bug the test suite found (the curvature fit used tangent dimension 4 regardless
   of the manifold), ρ = ‖II‖·Lip correlates with measured failure radius with the predicted sign on all three Z113 seeds
   on the representative probe (−0.12 / −0.14 / −0.23, p < 1e-4, shuffled null 0) — a fifth of the pre-registered
   strength, below the softmax margin, and gone on the fiber probe of the same model (+0.02, p = 0.57). On ResNet-18 the
   sign is wrong (+0.32; ρ is anti-predictive of errors, AUC 0.35). On GPT-2 it is null (+0.04). The manifold-margin
   variant (POST-HOC, geodesic distance to the nearest differently-labelled probe point) does better, +0.84 on ResNet,
   and still loses to the softmax margin (+0.93).
6. **The interrupt works when it is given z, and fails when it is derived from the structure.** The penalty that
   removed the planted shortcut on 4 of 5 seeds (decorrelated accuracy 0.01–0.02 → 0.999–1.000 after one round at λ=3;
   SPARE-style reweighting leaves it at 0.01) is a symmetric-KL consistency loss across hint values on random inputs. It needs to know which input token the nuisance is; any practitioner who knows that
   can apply it without topology. The spec's L_c taken literally — invariance along the *discovered* circle at the
   layer, no knowledge of the input token (`--mode structure`, 2 seeds, λ escalated 3 → 30 → 300, stopped at step
   5,000) — destroyed the circle and left the shortcut fully intact: on seed 0 the detector reported 0 spurious classes
   for three consecutive checkpoints on a model with shortcut reliance 1.000 and decorrelated accuracy 0.014; seed 1
   degraded (test 0.32) with reliance still 0.91. **Detection by H¹ can be evaded by re-encoding.** That is the most important negative in the run.
7. **rot-MNIST: what the network learned is the destruction of the circle.** The rotation orbit is a circle in pixel
   space and stays one through two conv layers of *any* network, trained or not. After pooling the trained net has
   erased it (median r 0.57, fc1 0.29) further than chance wiring does (0.79, 0.57). The fiber carries no rotation
   coordinate because the classifier quotiented it out — a learned invariance, read as an absence.
8. **GPT-2 layer 6 (exploratory).** Manifold score m = 0.06 → every answer below is flagged low-confidence, as designed.
   Intrinsic dimension 12–16 of 768. The weekday circle is certified out-of-sample by the symmetry-informed route
   (r = 0.992, lifetime ratio 25, shuffle null 0.005); months give a plane (angle r = 0.97) without a clean H¹
   coordinate (0.47); unsupervised Rips finds neither (0.11, 0.01).

## Caveats that apply to everything above

* **The isotypic route is symmetry-informed.** It is handed the group parameter (a for Z113, rotation index, weekday
  index). It discovers *which characters* the network uses and certifies each circle by H¹, out-of-sample where D ≫ n;
  it does not discover the group.
* **The manifold score m is fragile.** The same grokked model scores 0.99 or 0.45 depending on the neighbourhood size
  used for the spectral-gap test; ResNet layers score 0.01–0.04 and GPT-2 0.06 although every other statistic on them
  behaves. An attempted continuous replacement was no better and was removed. m is reported everywhere and trusted
  nowhere yet.
* **G under-reads a perfectly generalising model** when classes are thin curves sampled sparsely (Z113: 15–40 points
  per answer fiber winding through a torus): post-grok G_interior is 0.6–0.85 at test accuracy 1.00. As a label-free
  tracker of test accuracy over 68 checkpoints (unlabeled held-out path nodes) it ties mean softmax confidence in rank
  (Spearman 0.837 vs 0.836), is more linear (Pearson 0.85 vs 0.65) and is a worse estimator (MAE 0.38 vs 0.27).
* Library bugs found by the test suite during the run and fixed: β₀ of fibers (a connected torus read 624),
  tangent dimension of the curvature fit, single-shuffle noise band (50% false positives on pure noise). Pre-fix outputs
  are kept beside the new ones (`*_v1_dimcap4.json`).
* All P3 arms and both decoys were trained with 6–9 processes sharing one 8 GB laptop GPU; wall-clock numbers from
  those runs mean nothing. The `open()` timings were taken with that same contention and are therefore upper bounds.
* Three jobs were OOM-killed during the night (30 GB RAM, too many concurrent CUDA processes); each was rerun to
  completion, sequentially. Wide ResNet conv layers are PCA-reduced to 256 dims (54–87% of variance) before the graph.

## Deliverables (spec §7)

1. One-minute demo — `logs/demo_transcript.txt` (`python analysis/demo.py`)
2. Grokking figure — `figures/grokking.png`
3. Inventory figures — `figures/inventory.png`
4. Interrupt figure — `figures/interrupt.png`
5. P4/P5 against baselines — `figures/p4_p5.png` (AUCs and correlations; per-point scores were not stored, so no ROC curves)
6. The grid — `figures/grid.png`, `results/grid.json`
7. Library, tests, timings, this file — `whiteboxtopo/`, `tests/` (143 pass, 1 documented xfail), `results/test_timings.json`

## Where each number lives

| quantity | file | produced by |
|---|---|---|
| Z113 P2/P4/P5/P6-static, 3 seeds | `runs/z113_{clean_s0,clean_s1,cleanctrl_s2}/static.json` | `analysis/z113_static.py` |
| Z113 bundle (G, G_int, m, β₀, circles per checkpoint) | `runs/z113_*/sweep.json` | `analysis/z113_sweep.py` |
| P6/P7 scores | `results/p6_p7_score.json` | `analysis/score_p6_p7.py` |
| P3 detection vs onset, clean false positives | `results/p3_detect.json` | `analysis/z113_p3_detect.py` |
| P3 decoy null; reference accuracies | `results/p3_decoy.json`, `results/p3_reference_dec_acc.json` | `analysis/z113_p3_decoy.py` |
| P3 interventions | `logs/p3_*.log`, `runs/p3_*/history*.json`, `results/p3_interrupt_summary.json` | `analysis/z113_interrupt.py`, `analysis/fig_interrupt.py` |
| ring: Euclidean vs g-geodesic, what the state carries | `results/ring_pullback.json` | `analysis/ring_pullback.py` |
| ring: circle birth vs error | `results/ring_birth.json` | `analysis/ring_birth.py` |
| ring: P4 off-ring | `results/ring.json` | `analysis/ring_analysis.py` |
| rot-MNIST layers + controls | `results/rotmnist_layers.json`, `results/rotmnist_controls.json` | `analysis/rotmnist_*.py` |
| ResNet P6/P4/P5 | `results/resnet.json` (+ `_v1_dimcap4`) , `results/resnet_depth*.json` | `analysis/resnet_analysis.py`, `analysis/resnet_depth.py` |
| Z113 fibers, commutativity, boundary shifts, unsupervised-g null | `results/z113_fibers.json`, `z113_fiber_symmetry.json`, `z113_p2_boundary.json`, `z113_unsupervised_g.json` | `analysis/z113_extras.py` |
| GPT-2 exploratory | `results/gpt2_explore.json` | `analysis/gpt2_explore.py` |
| label-free G vs confidence | `results/z113_unlabeled_G.json` | `analysis/z113_unlabeled_G.py` |
| unit tests + timings | `tests/`, `results/test_timings.json` | `python -m pytest tests -q` (143 pass, 1 xfail) |

---

# RUN 2 — Morse-theoretic upgrade (2026-09-20, 09:24 → 13:15)

Pre-registration `SPEC2.md` (+ `AMENDMENTS.md` A-1…A-3), brief `docs/RUN2_BRIEF.md`, log `RUNLOG2.md`, pause reports
`PAUSE_1..4.md`, per-block write-ups `FINDINGS_A..G.md`, one-page view `figures/grid_both_runs.png`.

## Scorecard (by the letter of SPEC2)

| block | criterion | result | verdict |
|---|---|---|---|
| G manifold score | bootstrap CV < 0.15 everywhere; ResNet penultimate > 0; ordering Z113 > ring > ResNet > GPT-2 | CV 0.00 / 0.29 / undefined / 0.24; ResNet penultimate 0.000; ordering 0.60, 0.40, 0.00, 0.60 | FAIL |
| E1 unsupervised naming | ≥4 of top-5 classes by output sensitivity named at r > 0.9, no group parameter | 0/5 on three seeds; feature-shuffle null 0/5 | FAIL |
| E2 (no criterion) | prediction: fewer classes above band under pullback | ring 2→1 ✓, GPT-2 22→0 ✓, Z113 12→26 ✗ | 2 of 3 |
| E3 invariance as quotient | Z₂ witness on every Z113 model; rot-MNIST invariance depth exists | Z113 MET on 5 models incl. 2 UNSEEN (axis ratio 0.017–0.044, slice 0.996–0.999, swap 0.3–0.8%); rot-MNIST NOT MET | FAIL (Z113 half met) |
| D2 alignment vs evasion | raw Mapper NMI above clean mean+3σ on the evasion window | threshold 1.139 — unreachable; cannot fire on any model | FAIL |
| D3 / D4 interrupt | decorrelated acc > 0.98 on 5/5 seeds; stays clean for 2× steps | by the letter the trigger never fires → no intervention (0.012–0.016). POST-HOC direct trigger: registered penalty 1/5 (R), 1/5 (G), four seeds flattened by escalation; run-1 penalty + direct trigger **5/5** at λ=3 (A-4), incl. seed 1; D4 on the one passing R seed: 0 firings in 12 checks | FAIL |
| D nulls | 0 firings on clean control and decoys | raw NMI 0 (vacuous); direct NMI 0/153; Mapper-AMI 0/51 clean, **44/102 decoy** | — |
| A depth bipersistence (P6) | U_clean − U_scr > 0.4 on ResNet and Z113; GW_scr > 2×GW_clean | 0.000 and 0.000 (kill test fires: ambient fibers never connect); GW ratio 0.97, reversed on Z113 | FAIL |
| B kernel displacement (P4) | one score beats softmax conf and entropy on SVHN and CIFAR-100; GPT-2 ι > 0.5 | ι 0.57/0.59, 1−κ 0.43/0.64, ι(1−κ) 0.50/0.64 vs 0.83/0.82; GPT-2 ι 0.82/0.54 (met, ≡ Euclidean 1-NN) | FAIL |
| C Morse–Smale (P5) | ρ′ > 0.90 ResNet, > 0.5 Z113, beats softmax margin somewhere | +0.78; +0.19/+0.18/+0.16; 0 of 5 systems | FAIL |
| F vineyard (P7) | onset mode ±2 of grokking on every seed; W2 < null; spreads < 30% | 4/5 seeds (UNSEEN decoy_s1 +27); 6/10 pairs; count spread 0.71, lifetime spread 0.05 | FAIL |

No Run-2 block passes by the letter.

## What Run 2 established

1. **The evasion result is reproducible and deterministic.** Regenerating the structure-penalty run (A-1: it had never been
   checkpointed) reproduces run 1's trajectory step for step. On seed 0 the H¹ detector reports 0 spurious classes at steps 3500,
   4000, 4500, 5500 and 6000 while ablation reliance is 1.000 and decorrelated accuracy 0.014.
   **Finding (theorem-shaped).** *A monitor that certifies a property of the representation's shape, used as a training
   signal, can be satisfied by a re-encoding that leaves the function unchanged.* Run-1 numbers: three consecutive checkpoints of
   zero detected classes at reliance 1.000; run 2: five of the last six. `FINDINGS_D.md`, `figures/run2_alignment_vs_h1.png`.
2. **Shape can be evaded; function cannot — and function needs z.** The statistic that catches the re-encoded model is the plain
   dependence of the prediction on the nuisance on a decorrelated probe (7/7 evasion checkpoints, 0/153 false positives on clean
   and decoy models). It is the ablation ground truth in another form. The Mapper construction built on top of it is worse:
   unreachable threshold as registered; chance-adjusted, it false-alarms on decoy models (44/102) because it sees that z is
   *represented*, not that z is *used*.
3. **The Z₂ quotient replicates on unseen models.** Answer-fibers are S¹/Z₂ on five grokked models, two of which had never been
   examined when the criterion was written. This is the one confirmatory positive of Run 2.
4. **Three principled replacements for the failed cells are negative, each verified first on planted ground truth.**
   A: ambient class fibers do not connect at any depth in either network at the registered scale (distance concentration); GW to
   the input fiber does not track memorisation. B: near-OOD displacement is proportionally *more* visible to the head, not less
   (invisible fraction AUROC 0.33); ι ≡ Euclidean 1-NN where the kernel is most of the space. C: separatrix distance restates
   run 1's manifold margin on ResNet (ρ 0.974) and degenerates to an edge length where classes are many and sparsely sampled;
   softmax margin is not a separatrix distance in disguise.
4b. **Post-hoc, the interrupt can be made to work on all five seeds — without any topology.** A functional trigger (fires at
   step 500 on every seed, where H¹ was 1–2 checkpoints late on three) plus run 1's presence-and-value KL penalty clears the
   shortcut on 5/5 seeds at λ = 3 in one round, including the seed that failed twice in run 1. The brief's "generalised" penalty
   clears 1/5; every escalation past λ = 3, in either run, destroyed the model. One run per seed; needs z. `FINDINGS_D.md`.
5. **Held-out models did their job.** F fails only because an unseen model (decoy_s1) was included: its circles form at grokking
   and are then repeatedly disrupted while test accuracy reads 1.00.
6. **Neither scalar manifold score is fit for purpose**; the ID(k) profile separates plateau (grokked Z113) / decay (ring,
   ResNet — fastest at the penultimate layer) / growth (GPT-2, Z113 memoriser).

Softmax confidence, entropy and margin remain unbeaten on every system where they apply.

## Where the Run-2 numbers live

| quantity | file | produced by |
|---|---|---|
| G | `results/run2/blockG.json` | `analysis/run2_blockG.py` |
| E1/E2, E3 | `results/run2/blockE12.json`, `blockE3.json` | `analysis/run2_blockE12.py`, `run2_blockE3.py` |
| D1/D2 | `results/run2/blockD12.json`; regenerated model `runs/r2_reencoded_s{0,1}/`, `logs/run2_reencoded_s*.log` | `analysis/run2_blockD12.py` |
| D3/D4 | `runs/r2_align_{R,G}_s*/history.json`, `results/run2/d34_summary.json` | `analysis/run2_blockD34.py` |
| D5 | `results/run2/blockD5.json` | `analysis/run2_blockD5.py` |
| A | `results/run2/blockA.json`, `blockA_bipersistence.npz`, `figures/run2_bipersistence.png` | `analysis/run2_blockA.py`, `whiteboxtopo/depthprofile` |
| B | `results/run2/blockB.json`, `blockB_scores.npz`, `figures/run2_kernel_roc.png` | `analysis/run2_blockB.py`, `whiteboxtopo/kernel` |
| C | `results/run2/blockC.json`, `blockC_points.npz`, `figures/run2_morse_smale.png` | `analysis/run2_blockC.py`, `whiteboxtopo/morse` |
| F | `results/run2/blockF.json`, `figures/run2_vineyards.png` | `analysis/run2_blockF.py` |
| demo | `logs/demo_run2_transcript.txt` | `analysis/demo_run2.py` |

---

# RUN 3 — The Symmetry Group (2026-09-20, 13:38 → 15:40)

Pre-registration `SPEC3.md` (+ `AMENDMENTS.md` A-7…A-12), brief `docs/RUN3_BRIEF.md`, log `RUNLOG3.md`, pauses `PAUSE3_0..3.md`, findings
`FINDINGS_S, P1, P6, P3, P7, P2, IMG.md`, one-page view `figures/grid_three_runs.png`. Neighbouring work cited in every findings file:
LieGG (Moskalev, Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022).

**Said first: the group is not discoverable on Z113 by the letter.** Max-norm isometries of the output-pullback metric: orders 1, 1, 2, 13, 5.

## Scorecard (by the letter of SPEC3)

| block | result | verdict |
|---|---|---|
| S instruments | 7 instruments verified on planted truth; 2 faults caught there; 202 tests pass, 1 xfail | PASS |
| P1.1 group / frequencies | orders 1, 1, 2, 13, 5 (need ≥ 113 on ≥ 4/5); letter generator on 2 models only | FAIL |
| P1.2 memoriser | order 1 | met |
| P1.3 ring | order 1 on 3/3 | FAIL |
| P1.4 rot-MNIST Killing | rotation residual 0.26–0.34 at conv1/conv2 (need < 0.1); null-fraction criterion ill-posed | FAIL |
| P1.5 swap quotient (RE-SCORE) | q = 0.002–0.005 at resid_mid on 5/5; depth = resid_mid | met |
| P6.1 Killing dimension | 0 vs 0 (untrained 0) | FAIL |
| P6.2 invariance fraction | 0.997070 vs 0.997062 (unreachable, stated in advance) | FAIL |
| P6.3 order across training | never ≥ 113 on a clean checkpoint, seen or fresh; memorisers ≤ 1 | FAIL |
| P6.4 fresh seeds | connection-scale slope: generalisers 4/4, memorisers 2/4; ID profile: 4/6 (memorisers classify plateau, not growth) | FAIL (generaliser half met on every fresh model) |
| P3.1 breaking score | fires 122/122 where the floor fires incl. 7/7 evasion; Spearman 0.68; false alarms 39/102 decoy + every step 0 | FAIL |
| P3.3 interrupt | NOT SCORED (A-12) | — |
| P7.1 ring, P7.3 decoy_s1 | letter order never forms; ≥ 113 on 16% of decoy_s1's post-grok checkpoints | FAIL |
| P2.1 orbit explanation | 1.000 on 5/5; memoriser 0.000 | met |
| P2.2 boundary from discovered characters (RE-SCORE) | 49% / 22% / 14% (need 53.6%) | FAIL |

## What Run 3 established
1. **A trained network's symmetry is statistical, not entrywise.** A true translation changes the output-pullback distance matrix by 3% of its
   median on average on a grokked model (memoriser 22%) and by 14–42% in the worst entry. Any max-norm notion of isometry returns the trivial
   group for everyone; so does a Killing algebra over standard generators (dim 0 for clean, scrambled and untrained ResNet-18).
2. **The orbit explanation is exact** on 5/5 generalisers and holds for 0% of the memoriser's inputs.
3. **The swap quotient is kept, not learned.** q(swap) at resid_mid: 0.012 at init → 0.2–0.46 during memorisation → 0.002–0.005 at
   generalisation (0.58 in the memoriser). It returns at the grokking checkpoint or 1–2 after it, not before.
4. **Symmetry breaking is not spurious structure.** b_z never misses where the functional floor fires, including the H¹-evasion window, but it
   also fires through the whole memorisation phase of a decoy model (hint uninformative) and on every untrained model (b_z = 1/3). It is
   infinitesimal and scale-free, and both properties produced errors. The run-2 functional floor remains the better statistic.
5. **Replicated on fresh, unseen seeds (generaliser half):** class-fiber connection scale falls with depth (slopes −0.49…−0.57 on 4/4) and
   the ID(k) profile class holds (plateau on grokked Z113, decay on clean ResNet penultimate; 20/20 bootstraps each).
6. **POST-HOC, to be pre-registered before it is claimed:** under a mean-norm full-cycle rule a 113-cycle generator is found on 5/5 generalisers
   and never on a memoriser (0 of 73 checkpoints); it appears within one checkpoint of grokking (two of the five are fresh seeds whose sweeps
   ran after the rule was fixed); characters taken along it recover 4–5 of 5 inventory frequencies under a unique unit; it persists through the
   circle flicker of decoy_s1. It is a label-free statistic of the function on held-out inputs. The group *order* is not recoverable this way.
7. **Exploratory:** with z hidden, the nuisance is the top eigen-direction of the averaged input-pullback metric on both planted models (hint
   share of trace 0.95–0.96; clean < 1%).

The three definitions the run set out to test are not supported as registered: generalisation is not visible as group size in any registered
measure; spurious structure is not symmetry breaking (memorisation breaks symmetry too); "what it learned = the group" is recoverable only post-hoc.

## Where the Run-3 numbers live
`results/run3/`: `blockZ_finals.json`, `sweep_*.json`, `sweep_scores.json`, `blockRing.json`, `blockP22.json`, `blockP64.json`, `blockP3.json`,
`blockIMG*.json`, `grid_three_runs.json`. Scripts `analysis/run3_*.py`. Fresh models `runs/r3_*`. Demo `logs/demo_run3_transcript.txt`.

---

# RUN 4 — The Tower of Quotients (2026-09-20, 15:59 →)

Pre-registration `SPEC4.md` (+ A-13), brief `docs/RUN4_BRIEF.md`, log `RUNLOG4.md`, pauses `PAUSE4_1..3.md`, findings `FINDINGS4_P6, P1, P7, P3.md`,
one-page view `figures/grid_four_runs.png`. Neighbouring work cited throughout: LieGG (Moskalev, Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022).
Specimen fix delivered: TRUE memorisers (ResNet 99.6 / 99.4% train, 10% test on a 5,000-image subset; Z113 100% train, 1% test), matched clean models,
all fresh and unseen.

## Scorecard (by the letter of SPEC4)
| cell | result | verdict |
|---|---|---|
| sanity gate | every memoriser ≥ 99% train and at chance on test; every clean model ≥ 99% train | all admitted |
| P6.1 concentration | C_clean − C_mem = +0.02, −0.25, +0.05, −0.03 (needed > 0.25) | FAIL |
| P6.2 slot collapse | clean (0.49, 0.50, 0.01); memorisers max slot 0.38, 0.39 | PASS |
| P6.3 architectural symmetry | Z113 0.003 vs 0.56 ✓; ResNet shift consistency 0.68 vs 0.19 (clean needed ≥ 0.9) | FAIL |
| P6.4 depth slope | 7 of 8 on the right side; one ResNet memoriser −0.21 | FAIL |
| P6.5 group presence | 4/4 vs 0/2 | PASS |
| **P6 block** | 2 of 5 instruments (needed 4) | **FAIL** |
| **P1.1 group discovered, registered in advance** | generator 4/4 fresh generalisers, 0/2 memorisers; ≥ 4 of 5 frequencies at r > 0.9 on 4/4, unique unit | **PASS** |
| P1.2 planted truth for the statistical-symmetry rule | 3 of 3 | PASS |
| P1.3 tower (structural) | two linearisation artefacts | FAIL |
| P1.4 orbit explanation | 1.000, 1.000, 1.000, 0.990 vs 0.000, 0.000 | PASS |
| P7.1 / P7.2 / P7.3 timing | generator leads grokking (−4, −1, −3, 0); one seed without a memorisation phase; collapse at grok+2 / +3 | FAIL ×3 |
| **P3.1 dominant-direction rule, with the null that can fail** | fires on shortcut (0.957, 0.951) and on the legitimately dominant a-only task (0.997); silent on clean and decoys | **PASS** |
| P3.2 interrupt | decorrelated accuracy at step 6000: 0.999 / 1.000 / 1.000 / 1.000 / 1.000; floor re-firings in (6000, 12000]: 0; in-distribution test at 12000: 0.999 / 1.000 / 1.000 / 1.000 / 1.000 | PASS |

## What Run 4 established
1. **The group is discoverable from the input–output map with Jacobian access (level B) when the rule is fixed in advance** — generator and representation, on four unseen generalisers,
   absent on true memorisers. Pre-registered pass of a discovery cell (Run 4, four unseen generalisers). Caveat: planted-shortcut models carry it too.
2. **An unsupervised dominance rule on the input-pullback metric finds a planted shortcut with z hidden**, and correctly also fires on a task that
   legitimately has one dominant input: it detects dominance, not spuriousness. The slot profile of ḡ has four regimes — thirds (untrained),
   spread (memoriser), two task slots (generaliser), one slot (shortcut or single-input task).
3. **Memorisation is not un-concentrated geometry.** True memorisers concentrate ḡ as much as generalisers. The other four instruments separate
   every matched pair with no overlap (swap quotient 0.004 vs 0.57; shift consistency 0.68 vs 0.18; depth slope −0.5 vs −0.1; hint share 0.02 vs 0.30;
   generator 4/4 vs 0/2) while two of them miss a registered threshold on one side.
4. **The generator leads grokking** on fresh seeds (found at 17–93% test accuracy, up to four checkpoints before the jump); swap-invariance is
   architectural at init, broken while memorising, restored with generalisation; the nuisance slot's share of the geometry declines steadily and
   then collapses. All three on 6/6 generalisers, none on 2/2 memorisers; none of the registered timings met.
5. Two more instrument limits, both "infinitesimal misses finite": the linearised pullback reads a NULL→hint change as null at the embedding of
   every model, and never reads translation as null although predictions are exactly invariant.

**P3.2 note (2026-09-20 23:49).** PASS by the letter on 5/5 seeds. Not covered by the criterion and recorded for the record: seed 0's decorrelated accuracy dipped to 0.943 at step 7500 (floor silent, direct NMI 0.406; reliance −0.001) and recovered to 0.999 by step 8000; all other checks in (6000, 12000] read 0.997–1.000. One run per seed. The interrupt needs z (the nuisance must be known) and is not topological. LieGG (Moskalev et al., NeurIPS 2022) remains the cited neighbour for the symmetry instruments.

---

# RUN 5 — Registration (2026-09-21, 13:11 → 22:45; laptop suspended 14:00–19:05 with the analysis mid-flight, resumed intact)

Spec: `SPEC5.md` (saved 13:12:51, before any Run-5 model existed). A confirmation run: thresholds fixed from Run 4's matched pairs, scored on
eleven models trained afterwards. Authors: Arjun Paradkar & Reva Paradkar (independent). LieGG (Moskalev et al., NeurIPS 2022) cited throughout.

## Scorecard (by the letter of SPEC5)

| cell | criterion | result | verdict |
|---|---|---|---|
| gate | 11 specimens | all pass (Z113 clean/decoy 1.00/1.00; Z113 mem 1.00/0.009; ResNet+aug 0.99/0.78; ResNet mem 0.99/0.10) | ok |
| P6.a q(swap) | clean < 0.10, mem > 0.30 | 0.004 / 0.006 / 0.004 vs 0.465 / 0.676 | **PASS** |
| P6.b shift consistency | clean ≥ 0.75, mem ≤ 0.35 | 0.988 / 0.984 vs 0.202 / 0.180 (untrained 0.99 / 0.85 — clean side, as predicted) | **PASS** |
| P6.c depth slope | clean < −0.35, mem > −0.30 | clean −0.53 / −0.56 / −0.57 / −0.54 / −0.54; mem +0.02 / −0.01 / **−0.47** / −0.14; label-perm null ResNet clean −0.43, ResNet mem s3 −0.30 | **FAIL** (8/9) |
| P6.d slot collapse | clean a,b ≥ 0.40, hint ≤ 0.05; mem max ≤ 0.42 | clean (0.48,0.48,0.04) (0.49,0.49,0.02) (0.49,0.49,0.02); mem max 0.352, **0.445** | **FAIL** (4/5) |
| P6.e group presence | clean found, mem none | 3/3, 0/2 (untrained 0/5) | **PASS** |
| P6 block | all five | 3 of 5 | **FAIL** |
| P7.a lead | ≥ 0 on 3/3; never on mem | 0, 0, +1 (test acc at acceptance 0.68, 0.60, 0.51); 0/102 | **PASS** |
| P7.b restoration | q < 0.05 within ±3 on 3/3; mem > 0.30 | 0.010 / 0.022 / 0.003; mem min 0.34 / 0.50 | **PASS** |
| P7.c decoy hint collapse | < 0.01 in [g, g+4]; decline ρ ≤ −0.8 from > 0.30 | 0.0002 / 0.0002 at g+1; ρ −0.99 / −1.00 from 0.54 / 0.49 | **PASS** |
| P7.d persistence | ≥ 0.95 on 3/3 | **0.946** / 1.000 / 0.976 | **FAIL** |
| T.1 | decoys: hint null from resid_mid | ≤ 0.010 | **PASS** |
| T.2 | decoys: translation > 0.20 everywhere | 0.28–0.83 | **PASS** |
| T.3 | memorisers: hint > 0.15 everywhere | 0.42–0.71 | **PASS** |
| T block | | | **PASS** |

## What Run 5 established
1. **The definition survives on the clean side and is not registered on the memoriser side.** 17 of 19 clauses; every generaliser clause held.
   Two memorisers crossed ceilings set from two: one leans 45% on its hint slot, one ResNet memoriser's fibres tighten with depth — and its
   label-permutation null says that tightening is not about labels.
2. **The depth-slope instrument is a transformer instrument.** Z113: null −0.11 to −0.14 against −0.53 to −0.57 measured. ResNet: null −0.43
   against −0.54 on clean nets. On the CNN it mostly reads whole-cloud contraction; it should not be used there without the null subtracted.
3. **The generator arrives with the jump, not ahead of it.** Dense sweeps: accepted at the largest jump on 2/3 seeds and one checkpoint before
   on 1/3, at test accuracy 0.51–0.68; Run 4's leads of 1–4 were sparse-sweep artefacts (SPEC5 §4 said so in advance). "Never after the
   jump" holds on these dense Run 5 seeds only (5/5 incl. decoys); Run 6 had one seed one checkpoint after (lead −1) and "never on a memoriser" holds on 102 checkpoints.
4. **The finite tower replaces the linearised one.** Both Run-4 artefacts are gone; all three registered clauses pass; the tower orders
   transformations by how much of them the network keeps (symmetries → 0 with depth, answer-changing translations → grow).
5. Concentration is removed for good: on seven new models memorisers concentrate ḡ at least as much as generalisers (0.63–0.80 vs 0.55–0.64).

Post-hoc, labelled, upgrading nothing: a P6.d clause of the form "hint > 0.10 or max(a, b) < 0.40" reads 5/5; P7.d at 0.94 reads 3/3.
Files: `results/run5/`, `FINDINGS5_{P6,P7,T}.md`, `PAUSE5_{1,2,3}.md`, `figures/run5_*.png`, `figures/grid_five_runs.png`, `logs/demo_run5_transcript.txt`.

---

# RUN 6 — Orbit-Closure (2026-09-21 22:52 → 2026-09-22 02:20; GPU power-capped at 30 W throughout — specimens took 2.1 h instead of 36 min)

Spec: `SPEC6.md` (22:52:58, before any Run-6 model). A-15 fixed the F.2 clause shape before analysis. Thirteen fresh specimens, all through the
gate. Cited: Golan & El-Yaniv 2018; Hendrycks & Gimpel 2017; Liang et al. 2018; Azulay & Weiss 2019; Zhang 2019; LieGG.

## Scorecard (by the letter of SPEC6 + A-15)

| cell | criterion | result | verdict |
|---|---|---|---|
| P6.a / P6.b / P6.e | Run-5 thresholds | 0.005 ×3 vs 0.52–0.63; 0.989 ×2 vs 0.17–0.19; 3/3 vs 0/3 | **PASS / PASS / PASS** |
| P6.c null-subtracted slope | clean Δ < −0.25, mem > −0.10 | Z113 −0.41/−0.39/−0.45 vs −0.06/−0.05/−0.05 ✓; ResNet −0.12/−0.12 vs −0.12/−0.11/−0.13 ✗ | **FAIL**; transformer-only **PASS** |
| P6.d re-registered | clean a,b ≥ 0.40, hint ≤ 0.05; mem hint > 0.10 or max(a,b) < 0.40 | 3/3; 3/3 (hint 0.31, 0.42, 0.43); untrained memoriser-side 6/6 | **PASS** |
| P6 block | all five | 4 of 5 | **FAIL** |
| P7.d | persistence ≥ 0.94 on 3/3 | 0.871 / 1.000 / 0.970 | **FAIL** |
| K.2 Z113 unknown | AUROC(κ) ≥ 0.95 on OOV and p′=97 | OOV 0.994/0.996/0.996 (msp 0.79–0.89); p′=97 0.500 ×3 (valid inputs; predicted) | **FAIL** by the letter; OOV passes |
| K.3 ResNet unknown | κ_s > msp, entropy, ODIN; ≥ Golan-72 on SVHN & CIFAR-100 | SVHN κ_s 0.35/0.42 vs msp 0.81/0.74; CIFAR-100 0.69/0.68 vs 0.71; ODIN 0.74–0.81; Golan 0.11–0.45 | **FAIL** (beats Golan 6/6; loses to every softmax score) |
| K.4 | memoriser train κ < 0.3; clean > 0.95 | 0.00 ×3, 0.23 ×3; 1.00 ×5 | **PASS** |
| F.2 (A-15) | \|ρ(ε, r)\| > \|ρ(margin, r)\| | Z113 −0.03/−0.02/−0.13 vs +0.53/+0.46/+0.36; ResNet −0.41/−0.43 vs +0.93/+0.94 | **FAIL** 0/5 (brief's literal clause: vacuous pass) |
| F.2 direction | cos > 0.5 on ≥ 70% | 15%, 16% | **FAIL** |
| F.3 non-group null | group ε more negative than random-transformation ε | Z113 random −0.04/−0.14/−0.18 ≥ group; ResNet −0.37/−0.38 vs −0.41/−0.43 | **FAIL** (Z113) |
| U | Spearman(κ_s, ε̄) < −0.6, 5/5 clean | −0.84, −0.68, −0.66, −0.97, −0.97; memorisers vacuous (IQR 0.00 / 0.15–0.23) | **PASS** (by construction, SPEC6 §1) |

## What Run 6 established
1. **The definition of understanding is registered on the transformer** (four instruments + group, 6/6 fresh models, Runs 5–6) and has
   three instruments on the CNN: null-subtracted, a CNN's class fibres do not tighten with depth any differently in a generaliser than in a
   memoriser (−0.12 vs −0.11…−0.13).
2. **Orbit coherence is a knowledge detector only where the group is learned.** On Z113 an OOV token breaks the learned orbit (AUROC 0.99,
   softmax 0.79–0.89). On a CNN, whose group is the architecture's own, coherence holds on SVHN, CIFAR-100, noise and the untrained
   network alike (κ_s ≈ 0.98–0.996) — it is below chance against SVHN and below every softmax baseline everywhere. The same fact was in
   SPEC5 §3 / SPEC6 for the P6 nulls and was not carried through to K.3.
3. **Equivariance defect does not predict failure.** |ρ| 0.02–0.13 (Z113) and 0.41–0.43 (ResNet) against the margin's 0.36–0.53 / 0.93;
   random non-group transformations do as well on Z113 and nearly as well on ResNet. Sixth run, third instrument, same result, same reason.
4. **Memorisers know nothing by the orbit definition** (κ = 0.00 on Z113, 0.23 = 3/13 on ResNet), and the generator flickers on one fresh
   seed (0.87), accepted four checkpoints before its jump.
5. Golan-style designer transformations are worse than the model's own on every cell (6/6), so "model-discovered beats designer-chosen" holds — between two scores that both lose to max softmax.

Post-hoc, labelled, upgrading nothing: K.2 on the OOV set alone reads 3/3; P6 transformer-only reads 5/5 instruments; persistence at 0.85 reads 3/3.
Files: `results/run6/`, `FINDINGS6_{P6,K,F,U}.md`, `figures/run6_*.png`, `figures/grid_six_runs.png`, `logs/demo_run6_transcript.txt`.

---

# RUN 7 — The Extractor, and the End (2026-09-22 08:07 → 09:30; five-hour budget; GPU capped at 30 W)

Spec: `SPEC7.md` (08:07:01, clock start). No new training except three early-stopped Z113 seeds. Cited: LieGG; Bouchacourt et al. 2021;
Immer et al. 2022; Golan & El-Yaniv 2018; Hendrycks & Gimpel 2017; Liang et al. 2018.

## Scorecard (by the letter of SPEC7)
| cell | criterion | result | verdict |
|---|---|---|---|
| X.4 (i) | ≥ 3 learned families at penultimate, both clean ResNets | exactly 3 each: brightness, contrast, gamma (λ 0.69–0.72) | **PASS** |
| X.4 (ii) | shift, flip built-in (q_untr < 0.05) | q_untr 0.17–0.30 — partially learned by augmentation (λ 0.35, 0.27) | **FAIL** |
| X.4 (iii) | contrastive search recovers a learned family, 2/2 | brightness, |cos| 0.95–0.99, λ 0.76–0.79 | **PASS** |
| X.4 (iv) | memorisers < 2 learned | 0, 0, 0 | **PASS** |
| X.4 (v) | Z113 orbit translation λ < 0.2; swap λ ≥ 0.5 clean, < 0.2 mem | orbit translation ✓ 6/6; swap 0.40–0.73 (2/6 < 0.5) and built-in by the rule (q_untr 0.01–0.03); mem −50 to −76 ✓ | **FAIL** |
| X block | | 3 of 5 | **FAIL** |
| K′.2 | κ_s^learned ≥ built-in + 0.20 on SVHN and CIFAR-100, both seeds; untrained null < 0.9 | SVHN +0.28 / +0.31 ✓; CIFAR-100 −0.00 / +0.01 ✗; untrained 0.997 / 0.996 ✗ | **FAIL** |
| P5′ Z113 | AUROC(κ) > msp on 3/3 early-stopped seeds; null | 0.983 / 1.000 / 1.000 vs 0.563 / 0.852 / 0.900; random 0.44–0.49 | **PASS** |
| P5′ ResNet | κ_s^learned > msp on (a) or (b), both seeds; null | (a) 0.69 / 0.67 vs 0.84; (b) 0.67 / 0.66 vs 0.76; non-learned null below on 4/4 | **FAIL** |
| R accuracy | rebuilt ≥ 0.98, 6/6 | 1.000 ×6 | **PASS** |
| R ablation symmetry | ≥ 4/5 same direction, 6/6 | 1, 1, 1, 1, 1, 0 | **FAIL** |
| R γ | < 0.2 | 0.000 ×6 (logit variance explained 0.39–0.58) | **PASS** |
| P7 | persistence ≥ 0.85, 3/3 | 0.871 / 1.000 / 0.970 | **PASS** |

## What Run 7 established
1. **The extractor is real.** Subtract the untrained network and an augmented ResNet-18 has exactly three learned symmetries — brightness,
   contrast, gamma — none of them augmented, on both seeds, at every layer from layer2 on; a memoriser has none and is 2–3× more fragile
   than an untrained net to rotation, scale and shear; a gradient search over a 10-dimensional Lie algebra converges to brightness unaided.
   Training makes the network 1.5–3× *more* sensitive to noise, blur and JPEG. That is P1 on a CNN, and it is a different answer from "shifts and flips".
2. **"Built-in" is not one threshold.** The untrained transformer is blind to swap (q 0.01–0.03) and the untrained CNN is not blind to a 1–5 px
   shift (0.17–0.30), so the same rule excludes the right thing on one architecture and the wrong thing on the other. Memorisers *break* the
   transformer's built-in swap (λ −50 to −76); generalisers keep it — Runs 4–6 in the extractor's own units.
3. **Learned symmetries lift the CNN's knowledge boundary where the unknown set violates them** (SVHN, noise: +0.3) and not where it does not
   (CIFAR-100). Orbit coherence marks inputs on which the learned rule breaks; it does not mark semantically new images on which the rule holds.
   The cosine-based κ_s cannot separate a trained from an untrained network (0.997 either way): the extractor's normalised q can.
4. **Natural failure is orbit incoherence on the transformer** (AUROC 1.000, 1.000 on grokking-in-progress seeds; softmax 0.85, 0.90) — P5
   answered there. On the CNN three photometric invariances do not predict a semantic error (0.67–0.69 vs softmax 0.84).
5. **Reconstruction:** the generator and five characters specify the function — eleven fitted weights rebuild it at 100% — but the network's
   circuit is a different implementation (removing one frequency costs the rebuild nothing and the network up to 78 points).

Post-hoc, labelled, upgrading nothing: X.4(ii) with "built-in" read as q_untr < 0.3 → 2/2; K′.2 on SVHN alone → 2/2; P5′ Z113 on the two seeds inside the window → 2/2.
Files: `results/run7/`, `FINDINGS7_{X,K,P5,R,P7}.md`, `figures/run7_*.png`, `figures/grid_seven_runs.png`, `logs/demo_run7_transcript.txt`.

## Deferred (stated, not attempted)
- Čech gluing on GPT-2 IOI (compute).
- Learned-symmetry extraction on language models (needs a transformation library for text).
- Continuous / non-abelian group discovery beyond the 10-dimensional affine-plus-photometric Lie search.
- A κ built from the extractor's normalised displacement rather than a cosine (the untrained-null artefact of Runs 6–7).

---

# CLAIMS — final map (Runs 1–7). Every claim → run, evidence file, null, baseline, status.

| claim | run | evidence | null | baseline | status |
|---|---|---|---|---|---|
| The pullback metric g = JᵀJ is the right metric for cohomology of a representation (ring RNN: one clean circle at the network's own read-out accuracy) | 1 | results/ring_pullback.json | shuffle 0.01 | Euclidean Rips r = 0.75–0.80 | post-hoc, replicated 3 seeds |
| Z113 answer-fibres are intervals S¹/Z₂, not circles (the layer is commutative) | 1 | results/z113_fiber_symmetry.json | b = 0 slice: circle | — | finding |
| Path-product attribution equals integrated gradients (r = 1.000) | 1 | results/z113_p2.json | — | IG | registered PASS |
| A shape monitor used as a training signal is evaded by re-encoding; the function is unchanged (the evasion) | 1–2 | runs/r2_reencoded_s*, FINDINGS_D.md | — | — | reproduced from seed |
| A decorrelated-probe functional floor is not evaded and has 0/153 false alarms | 2 | results/run2/blockD12.json | clean + decoy checkpoints | Mapper NMI/AMI (fails its null) | registered (secondary) |
| Functional trigger + presence-and-value KL penalty removes a planted shortcut at λ = 3 in one round, 5/5, no re-firing over 2× training | 4 | results/run4/score.json P3_2 | no-intervention 0.012 | SPARE-style reweighting 0.014 | registered PASS |
| A trained network's symmetry is statistical: a full 113-cycle among mean-accepted permutations, found from the input–output map with Jacobian access (level B) on every fresh generaliser and no memoriser | 3–6 | results/run{4,5,6}/z113_finals.json | untrained: none; memorisers: none; planted truth | LieGG (continuous) | registered PASS (Run 4 P1.1; Runs 5–6 P6.e) |
| The generator's characters recover the learned frequencies under one unit of Z₁₁₃ˣ | 4 | results/run4/z113_finals.json | — | symmetry-informed inventory | registered PASS |
| Orbit explanation: predictions transform with the discovered group on 1.000 of inputs (memorisers 0.000) | 4, 6 | results/run4, run6 K.4 | memorisers | — | registered PASS |
| The dominant-direction rule finds a planted shortcut unsupervised and also fires on a legitimately single-input task (detects dominance, not spuriousness) | 4 | results/run4/z113_finals.json | decoys 0.0002; a-only fires | — | registered PASS with its limitation |
| Understanding = slot collapse + preserved swap symmetry + fibres tightening with depth + recoverable generator; memorisation = the negation | 4–6 | FINDINGS5_P6, FINDINGS6_P6 | untrained; label permutation | concentration (removed: memorisers concentrate as much) | registered on the transformer (Run 6); CNN: 3 instruments (depth slope dead after null subtraction) |
| The generator is accepted at or before the largest accuracy jump on 5/6 seeds (leads +0, +0, +1, +4, +0; up to 4 checkpoints early) and one checkpoint after it on 1/6 (−1); never on a memoriser (0 of 4 swept memoriser runs; 138 spec-compliant checkpoints, descriptive, no p-value). Coincident marker, not a predictor | 5–6 | results/run{5,6}/sweep_* | memorisers 0/4 runs (138 ckpts, descriptive) | — | registered (P7.a Run 5 PASS); persistence 0.87–1.00 (Run 7 ≥ 0.85 PASS) |
| Finite (Jacobian-free) tower: decoys quotient the hint from resid_mid on; memorisers never | 5 | results/run5/z113_finals.json | linearised tower (artefacts) | — | registered PASS |
| Orbit coherence marks unknown inputs where the group was learned (Z113 OOV 0.99) and not where it is built in (CNN shifts: below softmax) | 6 | results/run6 | p′=97 (valid inputs) 0.50 | max-softmax, entropy, ODIN, Golan-72 | K.2 OOV PASS; K.3 FAIL |
| Learned symmetry = architecture-subtracted invariance, λ = (q_untr − q_tr)/q_untr ≥ 0.5; on augmented CNNs exactly brightness, contrast, gamma; memorisers none; a Lie-algebra search finds brightness unaided | 7 | results/run7/X.json | untrained seeds; memorisers | library vs contrastive | registered PASS (i, iii, iv); built-in rule wrong for CNN shift / right for Z113 swap (ii, v FAIL) |
| Learned symmetries lift the CNN's knowledge boundary on SVHN (+0.28, +0.31) and noise, not on CIFAR-100 | 7 | results/run7/K.json | non-learned families; untrained (fails: cosine artefact) | max-softmax, entropy, ODIN, Golan-72, Run-6 built-in | K′.2 FAIL by the letter (SVHN half holds) |
| Natural failure: orbit coherence predicts held-out errors on the transformer (AUROC 0.98–1.00 vs softmax 0.56–0.90); on the CNN it does not (0.67–0.69 vs 0.84) | 7 | results/run7/P5_*.json | random transformations | max-softmax, entropy, ODIN | Z113 PASS 3/3; ResNet FAIL 0/2 |
| The predictor is rebuilt from the generator and five characters with 11 weights at 1.000 held-out accuracy (Stage 7–8); the network's own circuit is a different, less redundant implementation (ablation asymmetry) | 7 | results/run7/R.json | MP null on the residual, γ = 0.00 | — | accuracy PASS 6/6; ablation symmetry FAIL 0/6; γ PASS |
| Nothing derived from the representation beats the softmax margin at predicting the adversarial failure radius (three instruments, six runs) | 1–6 | CLAIMS runs 1–6 | shuffled; random transformations | softmax margin | negative, settled |
| Every version-1 headline (2,900× TC collapse; universality; bias precedence) | v1 → Aug 2026 | archive | feature shuffle, label permutation | — | retired with mechanisms |
