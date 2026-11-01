# WhiteBOX — Blinding Light

Started 2026-09-19. Clean slate.

The point is to open the box: use mathematics and algorithms to find out what is
actually true about cognition in these systems, and to follow what the math says
even when it says nothing interesting.

What this is not: defending a claim, dressing an instrument up as a finding, or
hunting novelty for its own sake. If the result is that there is no structure, the
result is that there is no structure.

Nothing carries over automatically. Anything pulled from
`../WhiteBOX - Archive` has to earn its way back in on its own merits.

---

## The ten-hour run (2026-09-19 → 20)

Spec in `docs/`. **Read `CLAIMS.md` first** — the scorecard by the letter of the pre-registration, what the run actually
established, and where every number lives. `RUNLOG.md` is the chronological log. `figures/grid.png` is the one-page view.

```
whiteboxtopo/     the library
  manifold/       knn_graph (euclidean | g | secant), tangent_pca, intrinsic_dim, manifold_score, geodesics, fit_manifold
  topology/       persistence, circular_coordinates (DSMVJ), circular correlations, isotypic_decomposition, fibers, matching
  geometry/       second_fundamental_form, lipschitz_field, reach, ignorance (spec d_g), tubular_distance, manifold_margin, saddles
  problems/       p1_inventory (metric='euclidean'|'g'), p1_symmetry_inventory, p2_explain, p3_spurious, p4, p5, p6, p7
  nulls/          feature_shuffle, label_permute, shuffle_predictor
  open.py         open_model(layers, holdout, ood, failure_radius, nuisance, symmetry, ...) -> Report(.summary(), .to_json())
systems/          Z113 transformer (clean | shortcut | scrambled | decoy), ring RNN, ResNet-18/CIFAR-10, rotated-MNIST CNN
analysis/         one script per experiment; fig_*.py and make_grid.py draw figures/ from results/
tests/            143 tests on synthetic manifolds with analytic ground truth (1 documented xfail); timings in results/
scripts/unit.sh   launch anything as a systemd user unit with capped BLAS threads (survives the terminal)
```

```bash
source .venv/bin/activate
python -m pytest tests -q                 # ~30 s
python analysis/demo.py                   # open() on Z113 and ResNet-18: the seven answers + wall time (7 s / 13 s)
```

Minimal use:

```python
from whiteboxtopo import open_model
from whiteboxtopo.problems import LayerData
ld = LayerData(name='layer', X=acts, J=jacobians, logits=logits, labels=y)      # J: (n, d_out, d_layer)
print(open_model({'layer': ld}, holdout=held_out_acts, ood={'set': (ood_acts, ood_logits)}).summary())
```
