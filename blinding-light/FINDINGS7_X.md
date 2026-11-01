# FINDINGS (Run 7) X — the learned-symmetry extractor

**Plain English.** Subtracting the untrained network works as a definition: on both augmented clean ResNets the extractor returns exactly
three learned symmetries at the penultimate layer — **brightness, contrast, gamma** — the photometric prediction, none of which was ever
augmented; the true memorisers have none; and a gradient search over a 10-dimensional Lie algebra, told nothing about the library, converges
to brightness on both seeds (|cos| 0.95–0.99). Two registered clauses fail for the same reason: the untrained CNN is *not* blind to shifts
and flips by the Euclidean q (q_untrained 0.17–0.30, not < 0.05), and the untrained transformer *is* blind to swap (0.01–0.03), so the
"built-in" rule excludes the wrong one of the two. Cited: LieGG (Moskalev et al. 2022); Bouchacourt et al. 2021; Immer et al. 2022.

| clause | registered | measured | verdict |
|---|---|---|---|
| (i) ≥ 3 learned families at the penultimate layer, both clean ResNets | λ ≥ 0.5 | s5: brightness +0.72, contrast +0.69, gamma +0.70; s6: +0.72, +0.70, +0.72 — exactly three each; next: translation-beyond-padding +0.47 / +0.45, shift +0.35, flip +0.27, saturation +0.21 / +0.29 | **PASS** |
| (ii) shift and flip excluded as built-in | q_untrained < 0.05 at the penultimate layer | shift 0.17–0.29 (five magnitudes), flip 0.30: the untrained ResNet-18 moves 20–30% of a random-pair distance under a 1–5 px shift; they read as *partially learned* (λ +0.35, +0.27) — by augmentation | **FAIL** |
| (iii) contrastive search recovers a learned library family, 2/2 | |cos| ≥ 0.8, library λ ≥ 0.5 | s5: brightness, cos −0.95/−0.96/−0.96, λ +0.79 (all top-3); s6: brightness, cos +0.99/+0.98/−0.96, λ +0.76 | **PASS** |
| (iv) memorisers < 2 learned symmetries | | 0, 0, 0 at the penultimate layer (contrast +0.48–0.50 is the nearest); geometric families λ −1.7 to −2.3: a memoriser is 2–3× *more* sensitive than an untrained net to rotation, scale and shear | **PASS** |
| (v) Z113: orbit translation λ_post < 0.2 on clean 9–14; swap λ_post ≥ 0.5 on clean, < 0.2 on memorisers | | orbit translation +0.09, −0.90, +0.09, −0.40, −0.88, −0.78 ✓ (a symmetry, not an invariance); swap +0.64, +0.40, +0.53, +0.73, +0.62, +0.47 (2 of 6 below 0.5) — and by the extractor's own rule swap is **built-in** (untrained q 0.010–0.026 at resid_post); memorisers −76, −59, −53 ✓ | **FAIL** (swap) |
| block | all five | | **FAIL** (3 of 5) |

**The inventory, read as P1 on the CNN.** Invariance depth (first layer with λ ≥ 0.5): brightness at layer1 (s5) / layer2 (s6); gamma at
layer2; contrast only at the penultimate layer (s5) / layer4 (s6). Translation beyond the padding and circular shift become learned only at
the logits. Hue, saturation, cutout, elastic: not learned (λ −0.3 to +0.3). **Noise, blur and JPEG have λ −0.6 to −1.9: training made the
network one and a half to three times more sensitive to high-frequency perturbation than it was at initialisation** — the same fact as the
adversarial-radius results of Runs 1–6, seen from the extractor's side. Geometric families (rotation, scale, shear) read λ −0.3 to −0.5 on
clean nets and −1.7 to −2.3 on memorisers: nobody learned them, and memorising makes a network fragile to them.

**Why (ii) and (v) fail together.** "Built-in" was defined by q_untrained < 0.05. The untrained transformer's attention pools the two operand
slots symmetrically enough that swap barely moves resid_post (q 0.01–0.03), so swap is architectural there and its λ is a ratio of two small
numbers (0.40–0.73 on generalisers, −50 to −76 on memorisers, which *break* it — Runs 4–6 said "present at initialisation, broken during
memorisation, restored at grokking", and the extractor says exactly that). The untrained CNN, with stride-2 convolutions and no pooling before
layer4, is not shift-blind by this q — Run 6's κ_s ≈ 0.996 on the untrained net was a cosine of near-constant logits, not shift invariance.
So the threshold 0.05 is right for the transformer's swap and wrong for the CNN's shift, and the brief's (ii)/(v) had them the other way round.
Post-hoc, labelled, upgrading nothing: with "built-in" read as q_untrained < 0.3, shift and flip are excluded on both ResNets and (ii) reads 2/2.

Files: `results/run7/X.json`, `X_z113.json`, `figures/run7_inventory.png`, `whiteboxtopo/extractor.py`.
