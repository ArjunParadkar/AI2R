# FINDINGS (Run 6) F — P5 as equivariance defect

**Plain English.** The equivariance defect does not predict where a network will fail. On the transformer it is barely correlated with the
failure radius (ρ = −0.03, −0.02, −0.12) and a *random* set of token substitutions does as well or better; on the CNN it reaches −0.41 / −0.43,
but the softmax margin reaches +0.93, and random non-group transformations reach −0.37 / −0.38. The flipping direction is not the gradient
of the defect (cosine > 0.5 on 15–16% of inputs, median 0.22). By the brief's literal clause F.2 "passes" because a negative number is more
negative than a positive one (A-15); by the |ρ| reading it fails 0/5. Cited: Golan & El-Yaniv 2018; Azulay & Weiss 2019; Zhang 2019; LieGG.

| cell | registered | Z113 clean 12, 13, 14 | ResNet+aug 5, 6 | verdict |
|---|---|---|---|---|
| F.2 |ρ(ε, radius)| > |ρ(margin, radius)|, ρ(ε) < 0 (A-15) | ε −0.031, −0.022, −0.125 (ε̄ −0.08, −0.07, −0.17); margin +0.529, +0.461, +0.363 | ε −0.409, −0.432 (ε̄ −0.47, −0.48); margin +0.926, +0.935 | **FAIL** 0/5 |
| F.2 brief's literal clause | ρ(ε) < ρ(margin) | true on all five, vacuously | | reported |
| F.2 direction | cos(d, ∇ε̄) > 0.5 on ≥ 70%, both ResNets | n/a (tokens) | 15%, 16% above 0.5; 78%, 79% above 0; median 0.22, 0.24 | **FAIL** |
| F.3 shuffled ε | ρ ≈ 0 | −0.02, +0.03, +0.05 | +0.02, −0.02 | met |
| F.3 non-group ε (7 random token substitutions / 13 random translations + patch permutations) | group ε must be more negative | **random −0.04, −0.14, −0.18** vs group −0.03, −0.02, −0.13 | random −0.37, −0.38 vs group −0.41, −0.43 | **FAIL** (Z113: the group is not doing the work) |
| Gaussian six-radius test (reported) | | — | ε −0.29 / −0.29; margin +0.63 / +0.65 | — |

**What the numbers say.** On the CNN, ε is a mediocre predictor of the PGD radius (|ρ| ≈ 0.42) that random pixel transformations nearly
match (0.37): "how much does the output move under *some* input change" is a smoothness proxy, and the group adds ≈ 0.05 to it. The softmax
margin is the quantity the PGD radius is *defined* from (steepest descent on the margin), so ρ = 0.93 is close to tautological — Runs 1 and 4
already said this ("the last layer is linear"). On the transformer, an in-distribution input has κ = 1 on 1000/1000 points and ε is the
residual size of an already-correct orbit; that residual carries no information about how far the resid_mid activation can move.

Status: P5 stays descriptive. Six runs, three instruments (curvature × Lipschitz, Morse–Smale separatrix distance, equivariance defect), none
beats the softmax margin, and the reason is the same each time.
Files: `results/run6/*_finals.json`, `raw_*.json`, `figures/run6_failure.png`.
