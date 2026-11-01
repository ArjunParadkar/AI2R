# PAUSE 3 (Run 3) — does symmetry breaking catch what shape missed?  · 2026-09-20 15:12

**Plain English.** It catches it — and it catches too much. The breaking score b_z (the share of the network's input-pullback geometry that
lies in the hint slot) fires on every one of the 122 checkpoints where the run-2 functional floor fires, including all 7 checkpoints where
the H¹ shape detector was evaded. But it fails two of the three registered clauses: it tracks the floor only loosely (Spearman 0.68, needed
> 0.9) and it false-alarms on 39 of 102 decoy checkpoints and on every model at step 0. P3.1 is NOT MET. P3.3 is NOT SCORED (A-12).

**Contradicting the brief's framing, first.**
1. **"Symmetry breaking = spurious structure" conflates memorisation with shortcut use.** On the decoy models the hint is uninformative, yet
   b_z is 0.44–0.48 during the memorisation phase and collapses to 0.0004 exactly when the model groks (decoy_s0: 0.204 → 0.077 → 0.008 as test
   accuracy goes 0.44 → 0.77 → 1.00). A memorising network keys on every token of every example, the nuisance included. That is a genuinely
   broken symmetry and it is not the planted shortcut. The run-2 floor (finite, decorrelated-probe dependence) has 0 false alarms there.
2. **A randomly initialised network has b_z = 1/3** (three input slots, by symmetry). The registered threshold (0.17) is below it, so "zero
   firings on clean checkpoints" was unreachable, and in P3.3 the trigger fired at step 0 on 5/5 seeds.
3. **b_z is scale-free and infinitesimal, and both bite.** A collapsed model with reliance 0.000 scores 0.943; a model with finite reliance
   0.354 scores 0.0002 because the local derivative is flat. An infinitesimal statistic can be driven to zero while finite dependence
   remains — the same direction as run 2's evasion lesson, seen once.
4. The depth profile is flat: the breaking is fully present after attention (0.96–0.99 at resid_mid, resid_post and logits alike).

| cell | pre-registered | measured | met? |
|---|---|---|---|
| Spearman(b_z, floor), pooled over 383 checkpoints | > 0.9 | 0.682 | **NOT MET** |
| fires wherever the floor fires | all | 122 / 122 | met |
| all 7 evasion-window checkpoints | fire | 7 / 7 (b_z 0.80–0.98) | met |
| zero firings on 153 clean + 102 decoy | 0 | clean 3 (all step 0); decoy 39 (the memorisation phase) | **NOT MET** |
| **P3.1** | | | **FAIL** |
| P3.3 interrupt | 5/5 > 0.98 at step 6000, no re-firing to 12000, test ≥ 0.99 | stopped at step ~700 (4.5–5.3 s/step under GPU contention) | **NOT SCORED** |
| P3.4 (exploratory) | — | hint direction is rank 1 among eigenvectors of E[g] on both planted models (hint share of trace 0.95–0.96; 0.79 already at step 500) | — |

**What is now claimable.** Told which input slot is the nuisance, a Jacobian block-norm share is never silent when the functional floor fires,
including where the shape monitor was evaded. It is not a shortcut detector: it fires on memorisation and at initialisation, ignores output
scale, and being infinitesimal can miss finite dependence. The run-2 floor remains the better statistic. Exploratory: on a planted model the
nuisance is the *top* eigen-direction of the averaged input-pullback metric, with z hidden — the first unsupervised handle on the nuisance in
three runs, on two models, with no null beyond the clean models (hint share < 1% of trace) and no criterion registered.
Neighbouring work: LieGG (Moskalev et al., NeurIPS 2022). Full write-up: `FINDINGS_P3.md`.
