# FINDINGS (Run 6) P6 — closing the definition

**Plain English.** With the memoriser clause of slot collapse written in the right shape, four of the five instruments pass on all eleven
fresh models, and the fifth passes on the transformer and fails on the CNN exactly as SPEC6 §7 said it would. The block fails by the letter
(P6.c on ResNet); the transformer-only reading of the definition passes 4/4 instruments + group on 3 + 3 fresh models. Persistence at 0.94
fails on one seed (0.87). Cited: Golan & El-Yaniv 2018; Hendrycks & Gimpel 2017; Liang et al. 2018; Azulay & Weiss 2019; Zhang 2019; LieGG.

| instrument | registered | clean 12, 13, 14 / ResNet+aug 5, 6 | memorisers 15, 16, 17 / ResNet 5, 6, 7 | verdict |
|---|---|---|---|---|
| P6.a q(swap) | < 0.10 / > 0.30 | 0.005, 0.005, 0.005 | 0.627, 0.572, 0.515 | **PASS** |
| P6.b shift consistency | ≥ 0.75 / ≤ 0.35 | 0.989, 0.989 | 0.189, 0.178, 0.169 | **PASS** |
| P6.c Δ = slope − label-permuted | < −0.25 / > −0.10 | Z −0.411, −0.389, −0.447; **R −0.123, −0.118** | Z −0.057, −0.047, −0.049; **R −0.119, −0.111, −0.133** | **FAIL** (transformer-only: PASS 6/6) |
| P6.d slot shares | clean a, b ≥ 0.40, hint ≤ 0.05; mem hint > 0.10 or max(a, b) < 0.40 | (0.489, 0.496, 0.015) (0.486, 0.496, 0.019) (0.495, 0.492, 0.014) | (0.364, 0.327, 0.309) (0.292, 0.289, 0.419) (0.245, 0.322, 0.432) — untrained (0.33, 0.33, 0.33) reads memoriser-side 6/6 | **PASS** |
| P6.e full 113-cycle | found / none | 3/3 | 0/3 | **PASS** |
| block | all five | | | **FAIL** (P6.c on ResNet) |
| P7.d persistence ≥ 0.94 | 3/3 | 0.871 (missing 33, 36, 41, 46), 1.000, 0.970 | | **FAIL** (seed 12) |

**The ResNet depth slope, settled.** Null-subtracted, the augmented clean nets read −0.12 and the memorisers −0.11 to −0.13: after the
label-permutation null is removed there is *no difference* between a CNN that generalises and one that memorises. What Run 5 measured as
"fibres tighten with depth" on ResNets was the whole cloud tightening. On the transformer the null-subtracted slopes are −0.39 to −0.45
against −0.05 for memorisers, a clean separation on six fresh models. The instrument is transformer-only, as registered for that case.

**Run 5 → Run 6 on the memoriser side of P6.d.** Run 5's ceiling (max slot ≤ 0.42) failed on a memoriser at 0.445; the re-registered clause
(hint > 0.10 or no task slot ≥ 0.40) passes 3/3 here with hint shares 0.31, 0.42, 0.43, and the untrained network reads memoriser-side as
required. Over Runs 4–6 that is 7/7 memorisers and 10/10 generalisers on the re-registered clause (post-hoc on Runs 4–5, registered here).

**Persistence.** Seed 12 accepts the generator first at checkpoint 19 (test 0.23), four checkpoints before its jump at 23, then loses it at
four of 31 later checkpoints. Seeds 13 and 14: lead 0 and −1. Across Runs 5–6 (six dense-swept seeds): leads +0, +0, +1, +4, +0, −1.

Status: the definition — slot collapse, preserved architectural symmetry, generator recoverable from outputs, and (transformer only) fibres
tightening with depth — holds on every generaliser and fails on every memoriser across Runs 4–6 (17 fresh matched models). It is registered
on the transformer; on the CNN it has three instruments, not four.
