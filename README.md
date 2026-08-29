# whitebox-topo

Topological interpretability of neural-network training. Persistent-homology
instruments (SV, TC, β₁, FPL, BiasTop) with calibrated signals built on them.

**Status: Phase 0 (skeleton + audited instruments).** See `AUDIT.md` for what
this library is and is not built on. Full quickstart arrives in Phase 5.

```
whitebox/
  core/       reduce.py (fixed PCA), persistence.py (ripser / giotto-ph), film.py
  metrics/    sv.py tc.py betti.py fpl.py biastop.py
  signals/    memorization (Phase 2), correspondence (Phase 3), failure (Phase 4)
  viz/
  activations.py   torch forward-hook extraction with the original pooling rules
tests/
benchmarks/        Phase 4
```

Dev install: `uv venv --python 3.12 .venv && uv pip install -e '.[dev]'` then `pytest`.
