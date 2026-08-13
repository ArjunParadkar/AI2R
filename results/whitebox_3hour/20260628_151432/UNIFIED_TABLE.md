# WHITEBOX — Six Black-Box Problems (3-hour run, honest statuses)

| Problem | Prior State | WHITEBOX Result | Status |
|---|---|---|---|
| P1 Concept crystallization | ρ≈0.59 order; no robustness measure | top-3 FPL ↔ per-class accuracy ρ=−0.84 (p=0.002); cryst↔distinct ρ=0.32 (p=0.36 n.s.) | Deepened — significant topological robustness measure |
| P3 Bias intervention | Hypothesis only | 70.8% gender-bias removal @ +1.9% ppl in CONTROLLED generative-LM setting; does NOT transfer to real-world classifiers — COMPAS gap ≤7%, toxicity identity-gap 0.158→0.219 (worse); precedence window null | Partially solved — controlled only |
| P4 Failure neighborhoods | Untested | layer3 TC: correct 0.044 vs misclassified 0.037, Mann–Whitney p=8.2e-6 (β1 n.s.) | Supported — significant |
| P5 Fragility under shift | Untested | residual SV ↔ noise accuracy-drop ρ=0.05 (p=0.88) | Negative — clean |
| P6 Memorization signature | ResNet only | GPT-2 124M: clean TC=607.7 vs 100%-noise TC=0.211 (~2900×), β1 14→1, threshold ~25% | Solved — cross-architecture |
| P7 Cross-modal universality | Confounded | matched from-scratch: ViT SV peak t=0.03 vs TinyGPT t=0.60 — do NOT align (ViT peak = ep1 transient) | Negative / inconclusive |

Scorecard (after 1-hr tightening): 2 strong DIAGNOSTIC positives (P4 failure-detection p=8e-6, P6 memorization ~2900× cross-arch), 1 significant new measure (P1 FPL ρ=−0.84), 1 controlled-only intervention (P3: 71% in-vitro, fails to transfer to real classifiers across COMPAS + toxicity), 2 honest negatives (P5 fragility tested with 2 predictors, P7 refutes the paper's own H5 universality claim).
WHITEBOX is a strong DIAGNOSTIC instrument (reads internal geometry to detect memorization, failures, concept robustness); its prescriptive/universality claims (intervention transfer, fragility prediction, timing universality) are NOT supported in real-world/matched tests.
See FINAL_SUMMARY.md for plain-English + breakthrough paragraph.
