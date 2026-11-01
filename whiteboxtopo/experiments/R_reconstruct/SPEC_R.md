# Block R — reconstruction and verification (pre-registration, 2026-09-14, from WHITEBOX_Final_Specification §7; written BEFORE any R compute)

_Horses: R1 Tosen Jordan, R2 Air Shakur, R3 Marvelous Sunday, R4 Matikane Tannhauser, R5 Biko Pegasus, R6 Cheval Grand, R7 Verxina, R8 Vivlos._

| run | what | pass criterion (frozen) |
|---|---|---|
| R1 Tosen Jordan | rebuild the T1 group model from certified irreps only (`reconstruct.from_irreps`; no weights copied) | accuracy ≥ f_θ − 0.5 %; logit correlation > 0.98 |
| R2 Air Shakur | rebuild the ring attractor from the circular coordinate + fitted tangent dynamics only | 100-step integration error within 10 % of the original; same drift statistics |
| R3 Marvelous Sunday | T2: certified circuit with complement mean-ablated | faithfulness ≥ 0.9; report γ_cov |
| R4 Matikane Tannhauser | residual r_t = h_t − ĥ_t; MP null (K block's, verbatim); γ_cov | T1: γ_cov < 0.05 with synthetic-noise null < 0.02; T2–T4 report |
| R5 Biko Pegasus | behavioural equivalence battery: accuracy, logit corr, Gaussian curves at 6 radii, PGD curves, OOD agreement | all curves within pre-registered bands: |Δacc| ≤ 0.5 pt, logit r > 0.98, per-radius agreement gap ≤ 5 pt, OOD agreement ≥ 90 % |
| R6 Cheval Grand | ablation symmetry: ablate each certified object in f and f̂ | effect-vector r > 0.9 per object; any failure = mis-mapped object, reported |
| R7 Verxina | negative control: rebuild from artifact-class objects only; random certified subsets | artifact-only ≈ chance; degradation proportional to κ removed (Spearman > 0.8) |
| R8 Vivlos | residue-guided second pass if R5 fails: top MP-outlier directions become new extraction targets, steps 3–8 once | report γ_cov before/after |

"Complete" is used only where γ_cov < 0.05 and R5–R7 pass (Constitution §12.9); everywhere else "opened to degree 1 − γ_cov".
