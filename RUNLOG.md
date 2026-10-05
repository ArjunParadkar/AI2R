# RUNLOG — ten-hour spec run

Spec: `docs/WHITEBOX_Final_10hr_Specification.pdf`. Clock 2026-09-19 23:33 CDT → deadline 2026-09-20 09:33.
This file is the hand-off between sessions: what ran, what it said, what is still running.

## How to resume
1. `systemctl --user list-units 'wb-*'` — what is alive. Logs in `logs/`, results in `results/` and `runs/*/`.
2. Launch jobs only as `systemd-run --user --unit=wb-<x> --collect scripts/unit.sh wb-<x> logs/<x>.log analysis/<x>.py`
   (survives a closed tab; caps BLAS threads at `WB_THREADS`, default 4).
3. Session 1 (23:33–00:56) died with its tab. Session 2 resumed 01:07 from its transcript.

## Rule for reporting
§6.3 criteria are scored by the letter. Anything introduced after seeing data is labelled POST-HOC and reported
next to the pre-registered number, never instead of it.

## State of each cell (update in place)

| cell | pre-registered result | status |
|---|---|---|
| P1 Z113 | 5/5 top frequencies r>0.9 via isotypic route (symmetry-informed: uses the known shift a→a+1). Raw unsupervised Rips: 207 unnamed classes = FAIL | done |
| P1 ring | Euclidean Rips r=0.78–0.80 FAIL. POST-HOC g-geodesic Rips: one class (ratio 20), r=0.979 (rank 0.996); output's own r=0.983; criterion 0.99 → FAIL by the letter, at the model's ceiling | s0 done, s1/s2 running (`wb-ringpb2`) |
| P1 rot-MNIST | penultimate: 0/10 FAIL. conv1/conv2 10/10 but pixel + untrained controls pending (`wb-rotctl`) | running |
| P6 Z113 | G jump 0.14 / 0.27 (<0.4) FAIL; scrambled G=0.60 (not <0.2) FAIL. POST-HOC G_interior: jump ≈0.45–0.6, scrambled 0.07 | clean_s0 re-sweep running (`wb-sweep2`) |
| P6 ResNet | clean G=1.00 pass, scrambled G=0.89 (not <0.3) FAIL, shuffle null 0.09 pass | depth analysis running (`wb-resdepth`) |
| P7 Z113 | births within ±3 ckpts of grokking: see sweep; cross-seed order τ: pending scoring | pending |
| P2 | IG r=1.000, margin CV 0.10 → PASS (constant-pattern segment is only 1.5% of the edge) | done |
| P3 | s0: detected step 500, one round λ=3, dec acc 0.998. s1: detected one checkpoint late, escalation to λ=300 collapsed the model (test 0.04). Baselines queued in `wb-p3` | running |
| P4 | Z113 FAIL on all 4 OOD sets; ResNet FAIL (d_g 0.64 vs softmax 0.82). Reason is structural: d_g pushes the displacement through J, which annihilates the directions the head ignores — exactly where OOD lives | done; tubular variant in `wb-resdepth` |
| P5 | Z113 ρ Spearman −0.02 FAIL (manifold is homogeneous: curv CV 0.15, Lip CV 0.08). ResNet +0.28 FAIL (wrong sign), softmax margin 0.93 | done |
| GPT-2 | exploratory, `wb-gpt2b` | running |
| Speed | Z113 static open(): 25 s | timing table pending |

## Caveats that must appear in CLAIMS.md
* The P3 penalty that works is a KL consistency loss across hint values on random inputs. It needs the nuisance
  variable to be known. The *detector* is topological; the *intervention* is not structure-derived.
* The isotypic decomposition is given the group parameter. It is symmetry-informed, not unsupervised.
* The manifold score m moved 0.99 → 0.45 on the same grokked model when its neighbourhood size changed. It is
  not yet a stable instrument.

## 01:35 notes
* Load average hit 76: jobs launched before `scripts/unit.sh` capped BLAS threads oversubscribe 20 cores. P3 interrupt
  runs are at step ~8000/25000 after 45 min. SPARE baseline launched in parallel as `wb-spare{0,1}` → `runs/p3b_spare_s*`.
  `scripts/run_p3.sh` will try to start its own spare/none stage when the interrupt runs end: stop `wb-p3` at that point
  (after `runs/p3_interrupt_s*/history.json` exist). No-intervention baseline = the existing `runs/z113_shortcut_s{0,1}`.
* rot-MNIST controls: pixels 10/10, untrained net 10/10 at conv1/conv2 → early-layer circle is inherited from the input.
  Learned fact is the destruction of the circle: GAP r 0.57 (untrained 0.79), fc1 0.29 (untrained 0.57).
* P3 detection scan: 0 flags in 153 clean checkpoints; shortcut first flagged at 500 / 1000, true onset 500 / 500.
  Harder null training now: `runs/z113_decoy_s{0,1}` (hint present, uninformative).
* Post-grok G oscillates with period 3 checkpoints; it is anti-correlated with train loss (ρ=−0.54), which swings
  1e-7 ↔ 1e-2 (slingshots) while accuracy stays 1.00.
* Ring g-geodesic persistence: s0 r=0.979, s1 r=0.993 (output's own r: 0.983, 0.993). One class, ratio ≈20, null 0.01.
* 01:33 `wb-resdepth` was OOM-killed (30 GB box, 4 GB VM resident, swap full, ~10 python/CUDA procs at 1.4 GB each).
  Rule from here: heavy CPU jobs go through ONE sequential queue (`scripts/queue_a.sh`, unit `wb-queue-a`), gated on PIDs.

## 02:10 notes
* Test suite written by a sub-agent: tests/ — 143 pass, 1 documented xfail (TwoNN at k=15 over-reads a noisy circle: 2.28).
  It found three real library bugs, all fixed: `significant_classes` β0 (torus read 624), `p5_failure` hard-coded tangent
  dim 4 (curvature was round-off on low-dim manifolds, sign arbitrary), single-shuffle noise band (50% FPR on noise;
  now 1.5 × max over two shuffles). Old outputs kept as `static_v1_dimcap4.json`, `results/resnet_v1_dimcap4.json`.
* P5 Z113 after the fix, 3 seeds: ρ = −0.12 / −0.14 / −0.23 (p<1e-4, null ≈0) vs softmax margin +0.21/+0.30/+0.30. Right sign,
  real, small. Still FAIL (< −0.5 and beat margin).
* GPT-2 L6 (exploratory): m=0.06 (LOW), d≈12–16 of 768. Weekday circle via isotypic route out-of-sample r=0.992 (ratio 25,
  null 0.005); months plane r=0.97 but H1 coordinate 0.47; unsupervised Rips finds neither (0.11, 0.01).
  P4: spec d_g AUC 0.40/0.38 (below chance), tubular 0.79/0.61, softmax 0.92/0.81, entropy 0.96/0.88 → FAIL.
* Ring P7: error drops at ckpt 1; certified circle (r>0.9 & ratio>3) first at ckpt 3 in all seeds; r tracks error through
  training (Spearman −0.82/−0.60/−0.55).
* New P3 arm `--mode structure` (units wb-struct0/1, 8000 steps): the spec's L_c taken literally — one step along the
  DISCOVERED circle at resid_mid, symmetric KL on the output, no knowledge of which input token is the nuisance.
* Queues: wb-queue-a (cleanctrl re-sweep → score → grokking fig → resnet_depth) → wb-queue-b (z113_unlabeled_G) →
  wb-queue-c (resnet_analysis v2 with curvature fix). Figures so far: figures/{grokking,inventory,p4_p5,interrupt}.png.
* 02:40 Tried a continuous manifold score (local explained variance of top-d̂ PCs, normalised by the feature-shuffle).
  Not better: on a circle with noise ≈ sample spacing it reads d̂=4, m_ev=0.23 (thresholded m: 0.54). Both scores inherit
  the same weak link — the local dimension estimate at the kNN scale. Removed; m stays as is, flagged as fragile.
* GPT-2 P5: ρ +0.04 (null +0.05), softmax margin +0.72 → negative.

## 03:05 notes
* Two more OOM kills (02:50 `wb-resdepth2` 11.8 GB peak on the 8192-d conv layer; `wb-queue-b` 7 GB). Fixes: wide layers
  PCA-reduced to 256 dims before the graph; perturbation searches chunked (100 imgs); explicit frees in z113_unlabeled_G.
  All three now run strictly sequentially in `wb-queue-d`, gated on the structure-run PIDs. Clean-model depth results from
  run 1 are safe in `results/resnet_depth_run1_clean_only.json`.
* P6/P7 Z113 scored (results/p6_p7_score.json): G endpoint jump 0.06 / 0.16 / 0.81, G_int 0.17 / 0.35 / 0.59 (s0/s1/ctrl);
  ctrl groks at ckpt 1 so its "jump" is from the untrained init. Births within ±3 of grok: 4/5, 5/5, 3/4. Cross-seed
  birth ORDER is ill-posed: seeds use different frequencies (only k=51 shared by s0,s1).
* ResNet depth, clean, unlabeled held-out nodes in the cloud: G_int 0.988–1.000 at every layer, label-permute null 0.09.
  Scrambled (partial, before the OOM): layer1 G_int 0.54, layer2 0.55 (b0 8–9).
* **P3 structure-derived penalty (spec's L_c taken literally): FAILS and exposes a detector blind spot.** s0 at λ=300:
  the circle is destroyed (detector: 0 spurious classes) while shortcut reliance stays 1.000 and decorrelated accuracy
  0.014. The network re-encoded the hint non-circularly. Detection-by-H¹ says "clean" on a fully shortcut-dependent model.
* z113_unlabeled_G (partial): post-grok acc 1.00 but G_int_unl only 0.2–0.45; softmax confidence tracks accuracy better.
* 03:20 stopped wb-struct0/1 at steps 5000/5500 (conclusive: reliance 1.0 / 0.91 at λ=300); logs/p3_structure_s*.log are the record (no history.json).

## 03:35 notes — ResNet through depth (results/resnet_depth.json; wide layers PCA→256, variance kept 0.54–0.87)
* G_int with unlabeled held-out path nodes, clean vs scrambled: layer1 0.988 vs 0.542; layer2 0.994 vs 0.552; layer3 0.997 vs
  0.600; layer4 1.000 vs 0.863; penultimate 1.000 vs 0.877. Label-permute null 0.09–0.11 everywhere. Fiber β0: 1.0–1.2 vs 7.9–9.1.
* Reading: the amendment does not rescue the <0.3 criterion. Under an output-derived metric, neighbours are same-output points,
  so same-output geodesics are nearly tautological; separation is real (largest early, shrinking with depth) but it is a
  gap, not "chance". β0 of the fibers separates the two models more cleanly than G.
* P5 variant (manifold margin vs PGD radius): clean +0.84 vs softmax margin +0.93; scrambled +0.31 vs +0.94 → loses.
* P4 ResNet clean: d_g 0.63/0.72, tubular 0.52/0.78, Euclid 0.68/0.89, softmax 0.83/0.71 → FAIL.

## 04:10 notes
* **Z113 answer-fibers are intervals, not circles — and that is the correct answer.** P1's criterion "answer-fibers are
  circles (β1=1)" fails (0 certified circles in 10 fibers × 2 seeds; raw β1 3–9). Reason: the network is commutative at
  resid_mid — ||h(a,c−a) − h(c−a,a)|| is 0.3–0.8% of the fiber radius on 3 seeds — so within a fiber each frequency
  carries e^{ika}+e^{ik(c−a)} = 2e^{ikc/2}cos(k(a−c/2)): a standing wave. Axis ratio of the fiber's image in each
  frequency plane: 0.016–0.025 (segment); same planes on the b=0 slice: 0.996–0.999 (circle). The fiber is
  S¹/(a~c−a) = an interval. The spec's ground truth was wrong. `results/z113_fiber_symmetry.json`, `results/z113_fibers.json`.
* ResNet P5 after curvature fix: ρ +0.32 (wrong sign, anti-predictive of errors AUC 0.35); margin +0.93.
* Label-free G (unlabeled held-out path nodes) vs mean confidence as trackers of test accuracy over 68 checkpoints:
  Spearman 0.837 vs 0.836; Pearson 0.85 vs 0.65; MAE as an estimator 0.38 vs 0.27. A tie at best.
* Demo: open() 7.1 s (Z113) and 12.9 s (ResNet-18) with a shared GPU → Speed PASS. Z113 P5 on the FIBER probe is
  +0.02 (p=0.57) vs −0.12..−0.23 on the representative probe: the weak P5 effect is probe-dependent.
* `open_model(symmetry=...)` + `p1_symmetry_inventory` added (the route that actually names Z113's circles).
* 04:05 queued wb-queue-f (after decoys): interrupt seeds 2,3,4 (pre-registered schedule, 6000 steps) + POST-HOC gentle schedule on seed 1 (×3, check 250, ≤5 rounds).
* 04:30 decoy null (results/p3_decoy.json): 0/26 + 0/26 checkpoints with a hint-tracking circle, ablation effect 0.000; shortcut models 25/26 each, ablation effect 0.35–0.36.
* 04:50 P3 five seeds of the pre-registered interrupt: success on s0,s2,s3,s4 (one round, λ=3 → dec acc ≥0.999); s1 fails under
  both the ×10 and the post-hoc ×3 schedule. Detection 0–4% late on all five. s2 was detected later than s1 and still
  recovered, so lateness is not what breaks s1. Unexplained.
* 05:00 Unsupervised P1 on Z113 under the g-geodesic metric (the ring's fix): null. No significant H¹ class under either
  metric (results/z113_unsupervised_g.json). The multi-frequency curve is sampled too sparsely for Rips; only the
  symmetry-informed route names Z113's circles. Also fixed a flaw in my own ratio-gap fallback (it fired among tiny bars:
  β1=2180); now only the 20 longest bars compete. Tests 143/1xfail.
* Reference accuracies on the decorrelated set (results/p3_reference_dec_acc.json): clean 0.9997–0.9998, decoy 1.000,
  no-intervention 0.012–0.016 (worst group 0.005–0.008).
* 05:15 P2 boundary ground truth tested: geodesics never end at c±1; destinations are the spectrum-predicted shifts (top-3 capture 25–52% vs 5% chance). results/z113_p2_boundary.json

## 05:45 — all experiments finished
* P3 final: interrupt s0 0.998 (25k steps), s2/s3/s4 1.000 (6k), s1 0.008 (both schedules); SPARE 0.014/0.014; none 0.016/0.012;
  structure-derived 0.014/0.008. `wb-p3` stopped before its redundant second stage (its just-created empty dirs removed).
* Quiet-machine demo: open() 4.6 s (Z113), 8.3 s (ResNet-18). Tests 143 pass / 1 xfail. All figures regenerated.
* No wb-* units are running. Nothing is committed to git (no commit was requested); snapshots in ~/blindinglight_backups/.
* Header times in this log before 03:00 are estimates and run ~30–60 min fast; file mtimes are authoritative.
