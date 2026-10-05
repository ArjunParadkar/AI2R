# FINDINGS (Run 4) P6 — understanding vs memorisation on TRUE memorisers: five instruments

**Plain English.** The specimens are now real: both ResNet memorisers fit random labels on 5,000 images to 99.6% / 99.4% and test at chance
(10.4% / 10.2%); both Z113 memorisers are at 100% train, 1% test; every matched clean model is ≥ 99% train. **By the letter the block fails:
2 of 5 instruments pass on all their matched pairs (needed 4).** Slot collapse (P6.2) and group presence (P6.5) pass. Concentration (P6.1)
fails outright — memorisers are as concentrated as generalisers. Architectural symmetry (P6.3) and the depth slope (P6.4) separate every pair
in the predicted direction and miss a threshold on one side each.
Neighbouring work: LieGG (Moskalev, Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022). Shift consistency as a property of CNNs: Zhang 2019;
Azulay & Weiss 2019; consistency regularisation: Laine & Aila 2017 — used here as a memorisation diagnostic on training data.

**Contradicting the brief, first.**
1. **"Its metric is spread rather than concentrated" is false for a true memoriser.** C = 1 − PR/PR_untrained: Z113 clean 0.71 / 0.54 vs memoriser
   0.69 / 0.79; ResNet clean 0.70 / 0.61 vs memoriser 0.65 / 0.64. Differences +0.02, −0.25, +0.05, −0.03 (needed > +0.25). Training of any kind
   concentrates ḡ; memorising concentrates it as much. The run-3 number the brief relied on (5.7 vs 7.8) was the per-input PR on an under-trained
   "memoriser"; on true memorisers the per-input version gives +0.20 / +0.16 — right direction, still under 0.25. The instrument's own null
   also fails on ResNet: a second untrained seed scores C = +0.24 and −0.17 (should be ≈ 0), so PR of ḡ is seed-noisy at that scale.
2. **A clean ResNet trained without augmentation does not keep its architecture's symmetry.** Shift consistency: untrained 0.99 / 0.90, clean
   0.68 / 0.67 (needed ≥ 0.9), memoriser 0.19 / 0.17. Everyone breaks the symmetry the convolution offers; the memoriser breaks it far more. The
   clean models here generalise only partly (58% test on 5,000 images), which is itself the honest reading: they sit between.
3. **Group presence does not certify "no shortcut".** Both planted-shortcut models also carry a full 113-cycle (they do compute a+b off the
   leaked residues). P6.5 separates generalisers from memorisers; it says nothing about spurious structure.

| instrument | pre-registered | measured (clean vs memoriser, per matched pair) | verdict |
|---|---|---|---|
| gate | memorisers ≥ 99% train, test ≤ 15% / 2% | ResNet 99.6/10.4, 99.4/10.2; Z113 100/1.0, 100/0.8; clean ≥ 99.1% train (ResNet test 58.8, 57.7; Z113 100) | all specimens admitted |
| P6.1 concentration | C_clean − C_mem > 0.25 on 4 pairs | +0.02, −0.25 (Z113); +0.05, −0.03 (ResNet) | **FAIL** |
| P6.2 slot collapse (Z113) | clean a+b ≥ 0.9, each ≥ 0.3; memoriser max slot ≤ 0.5 | clean (0.49, 0.50, 0.01) ×4 within ±0.01; memorisers (0.35, 0.38, 0.28), (0.28, 0.39, 0.33) | **PASS** |
| P6.3 architectural symmetry | Z113 q(swap) < 0.05 vs > 0.3; CNN shift consistency ≥ 0.9 vs ≤ 0.6 | Z113 0.003 vs 0.556, 0.005 vs 0.583 ✓; ResNet 0.68 vs 0.19, 0.67 vs 0.17 (clean below 0.9) | **FAIL** (clean-ResNet side) |
| P6.4 depth slope of fiber connection scale | clean < −0.3; memoriser > −0.1 | Z113 −0.55 vs 0.00, −0.51 vs −0.01 ✓; ResNet −0.59 vs −0.21 ✗, −0.49 vs −0.09 ✓ | **FAIL** (one memoriser at −0.21) |
| P6.5 group presence | full 113-cycle on 4 clean, none on 2 memorisers | 4/4 and 0/2 | **PASS** |
| **block** | ≥ 4 of 5 | 2 of 5 | **FAIL** |

**What the five instruments do show, read without thresholds.** On every one of the eight matched comparisons in P6.2–P6.5 the generaliser
and the true memoriser fall on the predicted sides of each other, with no overlap: swap quotient 0.003–0.005 vs 0.56–0.58; shift consistency
0.67–0.68 vs 0.17–0.19; depth slope −0.49…−0.59 vs 0.00…−0.21; hint-slot share 0.01–0.03 vs 0.28–0.33; generator 4/4 vs 0/2. The ResNet tower
says the same thing by depth: a 2-pixel shift moves the memoriser's representation 0.74–0.77 of a typical inter-image distance at the logits
and the clean model's 0.37–0.38, and the two only diverge from layer3 on. Four instruments, one direction; one instrument (concentration) null.
"Memorisation is the refusal to quotient" survives as a description of P6.2–P6.5 and not as the registered five-instrument definition.

Timings: Z113 instruments 20–190 s per model on CPU; ResNet ~60 s per model (500 Jacobians + shift test + depth profile).
Files: `results/run4/z113_finals.json`, `resnet.json`, `score.json`; `figures/run4_p6_instruments.png`, `run4_tower.png`.
