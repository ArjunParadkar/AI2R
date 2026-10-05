# FINDINGS (Run 7) P5′ — where it fails, with natural failure as the ground truth

**Plain English.** With the adversarial radius replaced by what actually goes wrong — held-out errors — the transformer half of P5 is answered:
orbit coherence under the discovered generator predicts which held-out pairs an early-stopped model gets wrong at AUROC 0.98–1.00, against
max-softmax's 0.56–0.90, and a random set of token substitutions gives 0.44–0.49. On the CNN it does not: κ_s over the three learned
photometric symmetries predicts test errors at 0.67–0.69 and corruption errors at 0.66–0.67, against max-softmax's 0.84 and 0.76. It beats
the non-learned-family null on every cell, so the learned families carry what signal there is; there is not enough. Cited: Golan & El-Yaniv
2018; Hendrycks & Gimpel 2017; Liang et al. 2018; LieGG.

| cell | registered | measured | verdict |
|---|---|---|---|
| Z113 seeds 18, 19, 20 — held-out errors, κ vs max-softmax | AUROC(κ) > AUROC(msp), 3/3 | s18: test acc **0.09** (never reached the 0.70–0.90 window in 12,000 steps; closest checkpoint used, as SPEC7 registers), κ 0.983 vs msp 0.563; s19: acc 0.785, κ **1.000** vs 0.852; s20: acc 0.880, κ **1.000** vs 0.900. κ_s 0.60–0.69; entropy 0.58–0.90 | **PASS** 3/3 (s18 outside the window; 2/2 inside it) |
| Z113 null: κ under 7 random token substitutions | worse than the group | 0.452, 0.442, 0.492 | met |
| ResNet (a) CIFAR-10 test errors (22–24%) | κ_s^learned > msp on (a) or (b), both seeds | κ_s 0.693 / 0.666; msp 0.844 / 0.844; entropy 0.846 / 0.845; ODIN 0.817 / 0.807 | fails |
| ResNet (b) 25 generated corruption cells pooled (error 42%) | | κ_s 0.670 / 0.664; msp 0.758 / 0.765; entropy 0.762 / 0.769; ODIN 0.756 / 0.759 | fails |
| **P5′ ResNet** | | | **FAIL** 0/2 |
| ResNet null: κ_s over non-learned families (λ < 0.2) | worse than learned on every scored cell | (a) 0.657 / 0.655; (b) 0.668 / 0.652 — below learned on 4/4 | met |

**On the transformer.** On seeds 19 and 20 the errors are *exactly* the inputs whose orbit is incoherent: mean κ on correct pairs 0.86 / 0.92,
on errors 0.10 / 0.08, and the AUROC is 1.000. Seed 18 had not grokked (test 0.09) — it is reported as the register says, and its 0.983 is a
statement about a memorising network's few correct answers being the coherent ones. This is P2's orbit explanation read as a failure
predictor: where the learned rule holds, the answer is right; where it does not, it is wrong. Softmax confidence carries much less
(0.56–0.90). Note the generator was not *accepted* by the automorphism search on seeds 18 and 19 at those checkpoints (u = 1 was used, which
is a generator of Z₁₁₃ for any learned frequency set); it was on seed 20.

**On the CNN.** The learned group is three photometric families. A CIFAR-10 test error is mostly a semantic confusion (cat/dog, truck/car)
on which brightness invariance holds as well as on a correct answer, so κ_s sees only the 2–3× blind-spot enrichment of K′.3, not the error
itself. The corruption cells include photometric corruptions (brightness, contrast) on which the learned invariances are *exactly* what is
being violated — and still κ_s reaches only 0.67. Max-softmax is a statement about the decision; κ_s is a statement about three invariances;
on a ten-class natural-image task the decision is the better predictor of its own error.

**Status of P5 after seven runs.** On the transformer, answered: natural failure = orbit incoherence (registered, 3/3, with the null met).
On the CNN: descriptive. Adversarial failure: nothing beats the margin anywhere (Runs 1–6).
Files: `results/run7/P5_z113.json`, `P5_resnet.json`, `figures/run7_roc.png`.
