# Phase 3 — `representation_correspondence`: pre-registered test of Sub-H4

Method: RTD (Barannikov et al. 2022), reimplemented from the reference code
(`whitebox/signals/correspondence.py`); score = 1 − RTD(data, repr) /
RTD(data, re-paired repr), clipped to [0, 1], bootstrap over batches. The
null keeps the representation's own topology and destroys only the
point-wise correspondence, so the score measures *structure-preserving
mapping*, exactly the Sub-H4 notion.

Pre-registered hypotheses (April proposal): correspondence is (a) high for
generalising models, (b) near zero for memorising models, (c) predictive of
held-out generalisation without held-out data.

## Step 1 — is the data-manifold persistence recovered? (`phase3_synthetic_manifold_check.csv`)

600 points, embedded in R²⁰ with σ = 0.03 noise, 250 maxmin landmarks, H0–H2 exact.

| manifold | expected | top H1 lifetimes | top H2 lifetimes | verdict |
|---|---|---|---|---|
| circle | β₁=1 | **1.44**, 0.03, 0.03 | 0.07, 0.04 | unambiguous |
| torus (R=2, r=0.7) | β₁=2, β₂=1 | **1.75, 0.71**, 0.47, 0.43 | **0.30**, 0.14, 0.11 | second loop only 1.5× the noise tail; β₂ 2× — recoverable but not clean |
| sphere | β₂=1 | 0.23, 0.23 (noise) | **1.02**, 0.005 | unambiguous |

Data-manifold PH is reliable when the manifold's features are well separated
in scale; the torus's small loop shows the limit at n = 600. On real images
there is no such check: the "data manifold" is pixel-space Euclidean
distance, which is a poor proxy for the image manifold. That is the crux the
brief asked to spend rigour on, and it is exactly where the measurement is
weakest — see Step 3.

## Step 2 — synthetic models (`phase3_synthetic.csv`, `phase3_synthetic.png`)

Per manifold × 3 seeds: `iso` (rotated/scaled data), `untrained` MLP hidden
layer, `general` (MLP regressing a smooth function of the coordinates,
held-out R² ≈ 1.00), `memorize` (same MLP fitting i.i.d. random targets,
train R² 0.96–1.00, held-out R² < −1.5).

| manifold | iso | untrained | general | memorize | (a)/(b) |
|---|---:|---:|---:|---:|---|
| circle | 1.000 | 0.80 | **0.39** | **0.46** | reversed |
| torus | 1.000 | 0.95 | **0.90** | **0.64** | holds |
| sphere | 1.000 | 0.91 | **0.74** | **0.73** | no separation |

The instrument itself is right (`iso` = 1.000 in every case, untrained
networks ≈ 0.8–0.95 because a random tanh net is nearly an embedding). The
*hypothesis* is wrong: a model that generalises preserves the input topology
only as far as its task needs it. On the circle the smooth target is a
1-D function of the angle, so the optimal representation folds the loop
onto its level sets and correspondence drops to 0.39 — lower than the
memoriser, which keeps a mangled but loop-bearing copy of the input. On the
torus the target depends on all three coordinates and the ordering holds.

## Step 3 — real data (`phase3_real.csv`, `phase3_real.png`)

Penultimate layer vs pixel space, 5 batches of 300.

| dataset / models | generalising | memorising (≥ 50 % noise) | AUC (gen vs mem) | ρ(score, test acc) |
|---|---:|---:|---:|---:|
| CIFAR-10, 21 ResNet-18 (Phase 2 battery) | 0.05–0.09 (full+aug 0.047–0.057) | 0.07–0.09 | 0.62 | −0.20 (p = 0.38) |
| MNIST, 12 MLPs (label noise 0/25/50/100 %) | 0.125 ± 0.005 | 0.24–0.26 | 1.00 **in the wrong direction** | −0.41 (p = 0.18) |

(a) fails: the best generalisers have the *lowest* correspondence with pixel
space. (b) fails: memorisers are not near zero and on MNIST are the most
input-faithful models. (c) fails: no significant relation to test accuracy
in either direction.

## Verdict

Correspondence, as pre-registered, is **not** a generalisation signal. It is
a valid and now well-tested measurement of something else: *how much of the
input's neighbourhood structure a layer retains*. Generalising classifiers
retain little of it by design (they collapse classes), memorisers retain
more (they need every input's identity). That reading is consistent across
synthetic and real data and is the reason the sign flips.

Ships as an **instrument** with this documentation, not as a signal a lab
should act on: `representation_correspondence` answers "does this layer
still look like the input?", which is useful for choosing layers to probe or
for tracking when a representation departs from the input (the untrained →
trained drop), and useless for judging whether the model understands. The
honest scope statement the brief asked for: clean and validated on synthetic
manifolds; on images it measures faithfulness to pixel-space geometry, and
the pre-registered hypotheses are refuted there.

## Cost

~1–2 s per cross-barcode at 300–400 points on 20 threads; a full
`correspondence` call (5 batches × 2 directions × {real, null}) ≈ 30–60 s.
Layers with many long-lived features (e.g. ResNet layer3) cost ~35 s per
cross-barcode; keep batches ≤ 400.
