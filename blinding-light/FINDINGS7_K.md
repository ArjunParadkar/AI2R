# FINDINGS (Run 7) K′ — P4 on the CNN with learned symmetries

**Plain English.** Replacing the architecture's shifts and flips by the three learned photometric symmetries moves orbit coherence on SVHN
from below chance (0.35, 0.42) to 0.63 and 0.73 — a gain of +0.28 and +0.31, above the registered +0.20 — and on Gaussian noise from
0.60/0.43 to 0.95/0.74, beating max-softmax there (0.80). On CIFAR-100 it gains nothing (0.69 → 0.69, 0.68 → 0.69): a photometric
invariance is as true of a CIFAR-100 image as of a CIFAR-10 one. And the untrained null fails: the untrained network's κ_s over the learned
families is 0.997, because its logits barely depend on the input and a cosine between two near-identical vectors is 1 whatever the
transformation. K′.2 fails by the letter on the CIFAR-100 half and on the null; the SVHN half holds. Cited: Golan & El-Yaniv 2018;
Hendrycks & Gimpel 2017; Liang et al. 2018 (ODIN); Bouchacourt et al. 2021; Immer et al. 2022; LieGG.

| unknown set | κ_s learned (s5 / s6) | κ_s built-in, Run 6 | gain | max-softmax | entropy | ODIN | Golan-72 | κ_s non-learned families (null) |
|---|---|---|---|---|---|---|---|---|
| SVHN | **0.631 / 0.728** | 0.351 / 0.418 | +0.28 / +0.31 ✓ | 0.809 / 0.738 | 0.821 / 0.742 | 0.811 / 0.750 | 0.113 / 0.199 | 0.529 / 0.541 |
| CIFAR-100 | 0.689 / 0.685 | 0.692 / 0.679 | −0.00 / +0.01 ✗ | 0.710 / 0.710 | 0.717 / 0.715 | 0.742 / 0.738 | 0.453 / 0.445 | 0.654 / 0.638 |
| Gaussian noise (reported) | **0.945 / 0.741** | 0.602 / 0.431 | +0.34 / +0.31 | 0.798 / 0.803 | 0.797 / 0.803 | 0.713 / 0.783 | 0.006 / 0.002 | 0.786 / 0.622 |
| untrained-network null: κ_s^learned on CIFAR-10 | 0.997 / 0.996 (registered < 0.9) | | | | | | | |

**K′.2: FAIL** (SVHN clause met 2/2; CIFAR-100 clause 0/2; untrained null 0/2). Beats max-softmax: noise only (s5). Beats Golan-72: 6/6.
Beats the non-learned-family null: 6/6 — the learned families carry the signal that exists.

**What the learned group can and cannot see.** SVHN digits and Gaussian noise are photometrically unlike CIFAR-10 — flat backgrounds,
different contrast statistics — and a network that learned to ignore brightness, contrast and gamma *on natural images* does not ignore
them on those inputs: coherence drops. CIFAR-100 images are natural images with the same photometric statistics; the learned invariances
hold on them exactly as on CIFAR-10, and coherence cannot tell the two apart. Orbit coherence marks inputs on which the network's learned
rule breaks; a semantically unknown image on which the rule still holds is not marked. That is the honest boundary of P4 by this definition.

**The untrained null, twice.** Run 6's untrained κ_s of 0.996 under shifts and Run 7's 0.997 under photometric families are the same
artefact: cosine similarity of an untrained network's logits is ≈ 1 for any pair of inputs. The Euclidean q of the extractor (normalised by
random-pair distance) does not have this problem (untrained q_brightness 0.26–1.40) — which is why the extractor separates trained from
untrained and κ_s does not. A κ built from the extractor's normalised displacement instead of a cosine would be the right next instrument; it is not registered here.

**K′.3.** The 16 lowest-κ_s CIFAR-10 test images: 75% / 44% misclassified (test error 22%) — a blind-spot enrichment of 2–3×.
Files: `results/run7/K.json`, `rawK_*.json`, `figures/run7_roc.png`.
