# benchmarks/ — index

Every claim in `CLAIMS.md` and `VALIDATION.md` is sourced from a script and
a results file here. This index maps phase doc → what it validates → the
script(s) that regenerate it → where the output lands → measured runtime
(from the committed `.log` files, hardware RTX 5070 Laptop 8 GB / 20 CPU
threads unless noted). Run scripts from the repo root with the `[dev]`
extra installed (`pip install -e ".[dev]"`).

| Phase doc | Validates | Regenerate | Output | Runtime |
|---|---|---|---|---|
| [`PHASE1.md`](PHASE1.md) | Core PH pipeline is fast/exact enough at benchmark scale; giotto-ph vs ripser backend agreement; landmark-approximation error | `phase1_wall.py`, `phase1_gpt2_bench.py` (timing tables); `phase1_figures.py` → `phase1_timing.png`; `phase1_validate_approx.py` → `phase1_approx_vs_exact.png` | `results/phase1_wall.csv`, `results/phase1_gpt2_timing.csv`, `results/phase1_approx_{summary,raw}.csv`, 2 PNGs (300 dpi) | GPT-2 XL activation extraction 106 s + one frame 16 s (`phase1_gpt2_bench.log`); exact PH sweep n=150→8000 is 0.02–39 s per point (table in doc) |
| [`PHASE2.md`](PHASE2.md) | `memorization_signal` (both families): label-noise battery, LOSO AUC, cross-family sign flip, training-trajectory limitation | `phase2_battery_cifar.py`, `phase2_battery_gpt2.py` (train + extract battery); `phase2_calibrate.py` → `phase2_roc.png`; `phase2_trajectory.py` → `phase2_trajectory.png` | `results/battery_cifar/*.{json,npz}`, `results/battery_gpt2/*.{json,npz}`, `results/phase2_{calibration_table,features}.csv`, `results/phase2_report.json`, `results/phase2_trajectory.csv`, 2 PNGs | Full CIFAR+GPT-2 battery (31 models) ≈ 74 min total, ~96–212 s/model (`phase2_battery.log`); `phase2_calibrate.py` < 1 min |
| [`PHASE3.md`](PHASE3.md) | `representation_correspondence` / RTD: synthetic-manifold ground truth, real-model Sub-H4 test (refuted) | `phase3_synthetic.py` → `phase3_synthetic.png` + manifold-check CSV; `phase3_real.py` → `phase3_real.png` (needs `benchmarks/data/`: CIFAR-10, MNIST, a GPT-2 sentence corpus) | `results/phase3_synthetic*.csv`, `results/phase3_real*.{csv,json}`, 2 PNGs | Synthetic step: seconds (600-pt clouds, 250 landmarks); real step: dominated by activation extraction over 33 models, not separately timed |
| [`PHASE6.md`](PHASE6.md) | `invariants` (E2 port): orbit-CV audit, invariant-quantity table; `legibility` (E4 port): profile validation, `legibility_score` refusal numbers; `memorization_signal(standardize=True)` cost | `phase6_standardize_memorization.py` (recomputes Phase 2 features standardised); `phase6_figures.py` → `phase6_invariance_cv.png`, `phase6_legibility_phase.png`, `phase6_standardize_auc.png`; E2/E4 tables themselves are copied from the `whitebox-v2` experiment tree, not regenerated here | `results/phase6/*.csv`, `results/phase6/*.png`, `results/phase6/phase6_standardize_report.json` | `phase6_standardize_memorization.py`: 62 (model, layer) rows from cached activations, ≈4 s/row ≈ 4 min (`phase6_standardize.log`) |
| [`DARKMATTER.md`](DARKMATTER.md) | `darkmatter.residual_structure` (E6 port): MP/TW spectral null + topology-with-shuffle-null on SAE reconstruction error | **Not regenerable from this repo** — the battery (GPT-2 small, `gpt2-small-res-jb`, 3 layers, 65k tokens/layer) ran in the `whitebox-v2` experiment tree (`E6/run_*.py`); its CSVs are copied verbatim into `results/darkmatter/`. `tests/test_darkmatter.py` reproduces the headline numbers directly from those CSVs (that reproduction *is* the regression test for this doc) | `results/darkmatter/e6_*.csv` | E6's own runtime is logged in `E6/` of the whitebox-v2 tree, not here; the local reproduction test (`tests/test_darkmatter.py`) runs in the full `pytest -q` pass, ~28 s total |
| [`GLUING.md`](GLUING.md) | `gluing.explanation_obstruction` (E3 port): sheaf-obstruction Ω, shuffle-null band, control-failure result | **Not regenerable from this repo** — the battery (GPT-2 small, IOI vs induction, 2400 prompts/task, 32/491-edge EAP graph) ran in the `whitebox-v2` experiment tree (`E3/run_*.py`); its CSVs are copied verbatim into `results/gluing/`. `tests/test_gluing.py` reproduces the two summary-config headline rows and the top IOI blocking pair directly from those CSVs | `results/gluing/{summary,pairs}.csv` | Same as above — E3's own runtime is in the whitebox-v2 tree; reproduction runs inside `pytest -q` |

## Not indexed above (flagged, not fixed)

`benchmarks/phase4_validation.py` / `phase4_figures.py` produce
`results/phase4/{phase4_report.json,phase4_summary.csv,phase4_validation.png}`
(the P4 failure-neighbourhood permutation-null benchmark referenced in
`AUDIT.md` Bin 1 row 3 and `VALIDATION.md`'s "Candidates that did NOT pass"
table) but there is no `PHASE4.md` write-up in this directory — the
negative result (direction reproduces, permutation p=0.36/0.09, not
shipped) is documented in `AUDIT.md` and `VALIDATION.md` instead. Left as-is
during the Stage 3 rename/release pass rather than authoring a new phase
doc under a mechanical-rename task.

## Full test suite

```bash
pytest -q          # 96 tests, ~28 s, no GPU required (torch/HF-dependent
                    # tests are skipped via pytest.importorskip if the
                    # optional extras aren't installed)
```
