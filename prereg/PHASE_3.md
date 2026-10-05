# PRE-REGISTRATION — WHITEBOX v2, Phase 3 (vision, reduced scope)
Authors: Arjun Paradkar and Reva Paradkar (independent; no university affiliation).  Written 2026-10-05, before any Phase-3 run of these scripts.  Tag `prereg-phase-3` (annotated, unsigned: no GPG key).
Scope actually registered: rotated MNIST; ResNet-18 on CIFAR-10 with and without BatchNorm.  NOT registered / NOT done (stated up front): ImageNet-subset, ResNet-50, ViT-S, CIFAR-100 training, Waterbirds and CelebA with JTT / SPARE / CLP, the Bouchacourt / Kvinge invariance measures, Bahat consistency beyond the 5-transformation TTA below.  Seeds: 5 per condition (the 10-seed rule is for confirmatory claims on small models; these claims are therefore labelled 5-seed and exact permutation p-values are limited to 1/252).
Disclosure: `bench/` smoke tests of these scripts were limited to checking that they execute (train 1 epoch; no results examined).  The Killing solver is not used (it failed its Phase-1 gate for non-constant-coefficient fields); the foliation test below uses only the metric.

## H3.1 rotated MNIST — learned foliation vs rotation (access level B)
Chart p = (t1, t2, a) around 12 fixed test digits (40 random points each, |t| <= 0.15, a uniform in [0, 2 pi)); model metric g = Fisher-Rao pullback in the chart (exact forward-mode Jacobian).  Per model: r_a = mean g_aa / tr g; kernel alignment = mean squared component along e_a of the smallest-eigenvalue eigenvector of g.  Conditions x seeds 1-5: rotaug (full-SO(2) augmentation, 8 epochs), noaug, scrambled (random labels, 3000 images, rotation-augmented, 80 epochs).
**H3.1a** rotaug vs noaug: AUROC(r_a lower in rotaug) >= 0.9 (exact permutation p reported).  **H3.1b** rotaug vs scrambled: AUROC(kernel alignment higher in rotaug) >= 0.9.  **H3.1c** the trivial baseline rotated-test-accuracy is reported for the same contrast; if it separates the conditions equally well the report says that the instrument adds nothing over test accuracy there.  A foliation 'aligned with rotation orbits' is claimed only if H3.1a and H3.1b both pass.

## H3.2 the BatchNorm confound (ResNet-18, CIFAR-10; conditions bn / nonorm x clean / scrambled x seeds 1-5)
Extractor lambda (Run-1 definition: (q_untrained - q_trained)/q_untrained at the penultimate layer, mean over 5 magnitudes) for brightness, contrast, gamma, against two untrained nulls (5 untrained seeds each): BN in train mode (batch statistics) and BN in eval mode.  CIs by bootstrap over the 5 seeds.
**H3.2a** nonorm clean: lambda CI lower bound > 0 for >= 2 of 3 families (photometric invariance is learned without any normalisation layer).  **H3.2b** the confound: for bn clean, lambda against the train-mode null is lower than against the eval-mode null (paired over seeds, CI of the mean difference lower bound > 0) in >= 2 of 3 families.  **H3.2c** scrambled (both norms): |lambda| < 0.1 for all three families.
## H3.3 knowledge boundary baselines (clean models, 2 norms x 5 seeds, 2000 test images, SVHN and CIFAR-100 as OOD)
kappa = hard orbit coherence under 6 photometric transformations (brightness +-0.15, log-contrast +-0.3, log-gamma +-0.3); baselines MSP, energy, ODIN (T=1000, eps=0.0014), TTA (hflip + 4 two-pixel shifts).  Targets: misclassified test images, SVHN, CIFAR-100.  PASS per target if the bootstrap-over-models 95% CI lower bound of AUROC(kappa) - AUROC(best baseline on that target) is > 0; overall PASS if >= 2 of 3 targets.  Prior expectation (v1, ResNet): kappa did not beat MSP everywhere; a fail is a result.

## Frozen artefacts
| file | sha256 |
|---|---|
| `analysis/phase3_rotmnist.py` | `d604a71e50e706b5595cf54698ae797675e8460b13993dc6a8f027e46215951a` |
| `analysis/phase3_resnet.py` | `57aad3ca048f07ab9656f5763bc2f1f8c005f18d0029e846d8235b8dcfabf39e` |
| `bench/stats.py` | `625298b69c694b85af152cb342de2bddb12ee93ebcdf9e1e7e224968a271550e` |
| `whiteboxtopo/extractor.py` | `ab9a1a6031ffbfead5d8ec4a7e5b7ab66c1783d1ca64178685217f58bd1ef856` |
| `whiteboxtopo/knowledge.py` | `01cb067ed167cf499ace90932f48b0d273729cad3834922e0378cb8e20bb00c6` |

Amendment 2026-10-05 to PHASE_3 (before any Phase-3 result existed): analysis/phase3_rotmnist.py uses reverse-mode (jacrev) instead of forward-mode (jacfwd) for the chart Jacobian because grid_sample has no forward-mode rule; same exact Jacobian, no metric or threshold change.  New sha256: 8873c242c926835bbbe35c2e864804a5d728d59b5651b54e6b5982232b6721de
