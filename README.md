# WHITEBOX

One question, pursued in four programmes from June to October 2026: **can mathematics open the box** — can topology and algebra say what a neural network actually learned, when, and whether a result is real or an instrument artefact? Each programme below is a complete git history merged in with its original commit dates. Read the newest first.

| programme | dates | where | what it found |
|---|---|---|---|
| **Structure of Understanding (E&E)** | Oct 2026 → | [`structure-of-understanding/`](structure-of-understanding/) | current programme: Phase 0+1 done, Phase 2 (K1/K2) pre-registered — `CLAUDE.md` is the protocol, `docs/CLAIMS.md` the claim ledger |
| **Blinding Light** | 19 Sep – 5 Oct 2026 | [`blinding-light/`](blinding-light/) | runs 1–7: knowledge is a group; group generator + frequencies recovered from outputs alone on unseen seeds; generator leads grokking; most pre-registered predictive cells FAIL by the letter — `CLAIMS.md`, `report/` |
| **whiteboxtopo library + sprint** | Aug – Sep 2026 | [`whiteboxtopo/`](whiteboxtopo/) | the instrument library (invariants, legibility, darkmatter, gluing), HPCC/Colab job sheets, `CLAIMS.md`, `VALIDATION.md`; experiment caches stripped |
| **v2 programme** | Aug 2026 | [`v2-programme/`](v2-programme/) | E1–E6 + stages 2–3: every predictive claim failed pre-registration; four instruments shipped |
| **v1: developmental interpretability** | Jun – Aug 2026 | this directory (root) | the original AI2R work below: persistent homology across training events |

## Run it in 60 seconds

The instrument library runs on plain numpy arrays — no trained model, no GPU, no checkpoint:

```bash
cd whiteboxtopo
pip install -e ".[legibility]" matplotlib
python demo/whitebox_demo.py          # ~6 s on a laptop CPU
pytest -q                             # the library's own tests, ~30 s
```

`demo/whitebox_demo.py` runs all four instruments on synthetic cases and prints one verdict line each
(real output, 2026-11-01, in [`whiteboxtopo/demo/whitebox_demo_output.txt`](whiteboxtopo/demo/whitebox_demo_output.txt)):

```
[1 invariants]  'top neuron activation': verdict=not_invariant, broken_by=perm+scale, CV=1.73   |   'network output mean': verdict=exact, CV=4.3e-16
[2 gluing]      Omega=0.267 (CI 0..0.267), shuffle null [0, 0.4], p_null=0.45, significant_vs_null=False, nerve edges=4, frustrated=1
[3 darkmatter]  verdict='neither', MP outliers=0/60 (CI (0.0, 0.0)), lambda+=1.5, TC ratio=1.06
[4 legibility]  modularity Q=0.749 (4 communities found, 4 planted), treewidth bound=8
```

i.e. a raw-neuron reading is exposed as a statement about one weight representative (not the function), a
planted two-regime explanation does **not** clear the shuffle null, pure-noise residuals are correctly called
unstructured, and a planted 4-block attribution graph is recovered.  It also writes
[`whiteboxtopo/demo/whitebox_demo.png`](whiteboxtopo/demo/whitebox_demo.png) (2×2 panel) — 25-second walkthrough:
[`whiteboxtopo/demo/whitebox_demo.mp4`](whiteboxtopo/demo/whitebox_demo.mp4).

Papers, boards and specs across all eras: [`papers/`](papers/) (competition papers, NeurIPS drafts, science-fair board + logbook, FINAL_SPEC, HANDBOOK, mathematical foundations).

Nulls are results here. Nothing in this repository upgrades a verdict that the pre-registration did not grant.

---

## v1 (original README)

A topological framework for developmental interpretability of neural network
learning — tracking how representations reorganize *during* training using
persistent homology, rather than probing a finished model.

The paper is in [`paper/`](paper/): `WhiteBox_ArjunParadkar_v2.pdf` (10 pp),
with its LaTeX source under `paper/WhiteBox_src/main.tex` and the figures and
tables it pulls from in `paper/WHITEBOX_regenerated_figures_tables/`. The
`\graphicspath` in `main.tex` is relative, so the paper compiles with
`pdflatex main.tex` from inside `paper/WhiteBox_src/`.

Arjun Paradkar and Reva Paradkar (independent).

## What is measured

Layer activations are reduced (PCA→30, fit at epoch 0 and held fixed), then
run through Vietoris–Rips persistent homology. Three quantities are tracked
across training:

- **SV** — 2-Wasserstein distance between successive persistence diagrams;
  spikes mark representational phase transitions.
- **TC** — total persistence, `Σ(d_i − b_i)²`.
- **β₁** — Betti number at the median pairwise distance.

## Layout

```
whitebox.py              core topological machinery
exp_vision.py            ResNet-18 / CIFAR-10 trajectories
exp_language.py          TinyGPT language trajectories
exp_memorization.py      label-noise memorization signature
generate_results.py      original figures + tables

phase1_intervention.py   topological bias intervention (the main result)
phase1_figures.py          --blocks --topk --alphas --epochs
phase2_scaling.py        from-scratch precedence window across model sizes
phase2_figures.py
phase3_driver.py         H1 (n=10), H4 noise curve, H2 100-epoch run
phase4_compas.py         COMPAS race-erasure (data auto-downloads)
phase4b_toxicity.py
phase4_figures.py
generate_final_report.py

expA_memorization.py     P6 memorization at GPT-2 124M
expB_crystallization.py  P1 + FPL robustness (saves a ResNet checkpoint)
expC_failure.py          P4/P5, consumes expB's checkpoint
expD_crossmodal.py       P7, tiny-ViT-from-scratch vs TinyGPT
gen_3h_outputs.py        figures + unified table
run_3h_chain.sh <dir>    orchestrates expA–expD

regen_paper_assets.py    rebuilds the paper's figures and tables
results/                 figures, metrics JSON, tables, logs, reports
data/                    corpora and tabular datasets
```

`results/` holds the scientific record and is tracked. Model checkpoints are
not — they are regenerated by the scripts that wrote them.

## Environment

This runs on a single RTX 5070 Laptop (8 GB VRAM), which constrains most
design choices here — the experiments are meant to fit that budget, and the
3-hour chain is run **serially** because 8 GB cannot parallelize it.

The environment has some sharp edges worth knowing before you try to
reproduce it:

- Python 3.14 with torch 2.11+cu130.
- `transformers` 5.12.1 is installed `--no-deps` into a local `site_pkgs/`
  and loaded via `PYTHONPATH`, rather than into the venv.
- The `tokenizers` pin in `site_pkgs/transformers/dependency_versions_table.py`
  has to be relaxed to `<=0.24.0` — only tokenizers 0.23.1 ships cp314 wheels.
- `accelerate` is deliberately **not** installed; it drags in a duplicate torch.

Neither `venv` nor `site_pkgs/` is tracked. CIFAR-10 is not tracked either
(it is public and re-downloaded on demand); the Gutenberg texts, the cached
general corpus, COMPAS and the toxicity set are small enough to keep, so the
language and fairness experiments stay reproducible from a clone.

## Findings, stated honestly

- **Memorization signature (P6) replicates at GPT-2 124M.** Clean TC = 607.7
  vs 0.211 at 100% label noise — roughly 2900× separation, wider than
  ResNet's ~1000×. β₁ collapses 14→1. Critical noise threshold ≈ 25%. This is
  the strongest cross-architecture result here.
- **The bias intervention is surgical, not blunt.** Projecting the gender
  direction out of the residual stream — *refreshed every epoch*, because the
  direction rotates — cuts bias ~43% for ~+3% perplexity at α=0.3. Two details
  carry that number: erase at blocks {1,2,3} and **not** block 0 (erasing raw
  token embeddings caused nearly all the capability damage), and train against
  a large general-prose mix, or the control LM collapses on its own and
  "surgery vs amputation" stops meaning anything. A one-time fixed-direction
  head ablation does essentially nothing.
- **Topological precedence (H3) is a from-scratch phenomenon.** In a
  pretrained GPT-2, gender is already encoded, so layer-0 BiasTop is frozen
  and deep-layer BiasTop starts high and flat. The precedence window that
  from-scratch TinyGPT shows (t_bias=2 < t_output=4) does **not** cleanly hold
  for a pretrained model. The paper reports this as mixed rather than
  supported.
- Of the five original hypotheses, H1 and H4 are decisive, H2 is supported,
  H3 is mixed, and H5 is partial — both modalities show the peak-then-decay
  shape but peak at different normalized times.

Results were reported as they came out; the 40% / <2% targets in the original
plan were not tuned toward.
