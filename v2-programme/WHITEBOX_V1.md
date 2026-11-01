# WHITEBOX v1 — the record (everything done on WHITEBOX up to 2026-08-29)

"v1" = the WHITEBOX paper (Paradkar 2026), the original experiment clone, and
the `whitebox-topo` library (Phases 0–5). This file is the v2 repo's pointer
back to all of it. Nothing here is a new claim; it is the ledger v2 builds on.

## Where it lives

| Thing | Path | Status |
|---|---|---|
| Original experiments + paper (v2 pdf under `paper/`) | `/home/god/whitebox_experiments_clone` — GitHub `ArjunParadkar/AI2R` (commit 95e7a5b) | untouched reference |
| Library `whitebox-topo` (import name `whitebox`) | `/home/god/whitebox-lib` — Phases 0–5 done 2026-08-29, commit `829f18a`, **local only, not pushed** | frozen; v2 imports it as a dependency |
| Audit of every paper claim | `whitebox-lib/AUDIT.md` (bins: robust / shaky / instruments) | authoritative |
| Validation ledger | `whitebox-lib/VALIDATION.md`, `benchmarks/PHASE1–4.md`, `benchmarks/results/` (CSVs + 300 dpi figs) | authoritative |
| Env | `whitebox-lib/.venv` (Py3.12, torch 2.11 cu128, giotto-ph, ripser, persim, networkx 3.6) | reuse for v2 |

## What v1 built (instruments — claim-independent, tested)

* `compute_diagrams` / `topological_film` / `FilmRecorder`: VR persistence (H0, H1) of a fixed-PCA activation probe cloud per (layer, epoch); ripser + giotto-ph backends (bit-identical); paired subsampling CIs; landmark/GPU-distance approximations validated (landmark keeps TC *ordering* ρ 0.98 but destroys β₁ — not used by any signal).
* Diagram functionals: structural velocity (W₂ between consecutive diagrams), topological complexity TC = Σ(d−b)², persistent entropy (Rucco), β_k at scale (two scale conventions — must be stated), top-k feature-persistence lifetime, BiasTop + permutation test (FPR 0/30, underpowered below n≈300/group).
* RTD correspondence (Barannikov ICML 2022) reimplemented exactly; `iso` control = 1.000 on synthetic manifolds.
* Timing: 0.11 s at n=500, 39 s at n=8000 (≈ n^2.3). TC and β₁ are probe-size dependent — compare at fixed n only.

## What v1 killed (the important part)

| Paper claim | What the battery showed |
|---|---|
| P6/H4 memorisation = TC/β₁ collapse, ~1000× (ResNet) / **2,900× (GPT-2)**, architecture-universal | The collapse came from 100 %-noise models that **never fit the noise** (ResNet train acc 0.10; GPT-2 fit 39 % of targets) — a failure-to-learn artefact. Genuine memorisers show no collapse. The real signature's **sign flips between CNN and LM: cross-family AUC 0.00** (CNN calibration applied to GPT-2 gives 0.67, LM applied to CNN 0.00). Universality refuted. |
| Sub-H4 / representation-correspondence hypotheses (generalisers retain more input topology, correspondence tracks test accuracy) | Refuted on real data: generalisers score *lowest* on CIFAR (0.05), AUC gen-vs-mem 0.62, ρ(score, test acc) −0.20/−0.41 n.s. RTD measures "how much input topology a layer retains", not understanding. Instrument only. |
| P4 failure neighbourhoods (misclassified inputs in sparser layer-3 regions, Mann-Whitney p 8e-6) | The paper's test was pseudo-replicated over bootstrap draws. Under a **label-permutation null p = 0.36 / 0.09**. Per-sample AUC 0.54/0.53 vs softmax confidence 0.90. Not shipped. |
| P1 top-3 FPL ↔ per-class accuracy (ρ −0.84) | Did not replicate on full-data models: ρ −0.53 (p 0.12), −0.27 (p 0.45); n = 10 classes. Not shipped. |
| P5 fragility under shift | Null (ρ ≤ 0.56, all p > 0.08), as in the paper's own run. |
| H3 bias precedence, P3 intervention transfer, H5 cross-modal universality | Already negative/mixed in the original record (precedence is a from-scratch-only phenomenon; COMPAS/toxicity transfer failed; ViT "peak" was an epoch-1 transient). No API. |

## What survived

1. `memorization_signal(model, probe, family=...)` — **per-family, converged checkpoints only**: leave-one-seed-out AUC **0.98** (21 ResNet-18/CIFAR-10, label noise 0–100 %) and **1.00** (GPT-2 124M LM-target-noise battery, tiny: 2 seeds, 8 labelled models, all fine-tunes overfit). `family` is mandatory; no auto-detect; other families must run the battery first.
2. The SV phase-transition peak (H1) reproduces across seeds (ρ 0.93) — descriptive instrument, no decision rule.
3. **The methodology itself**: pre-registered hypotheses, shuffle/permutation/random nulls, bootstrap CIs, cross-family tests, kill attempts, negatives written up with the same care. This is what v2 inherits as its operating rules (see `PROGRAM.md`).

## v1 gotchas carried forward

* The distribution name `whitebox` collides on PyPI → v2 Stage 3 renames the import to `whiteboxtopo`.
* Two β₁ scale conventions and two FPL definitions exist in the v1 record; always state which.
* TC is scale-dependent (activation norm, PCA dim, threshold) — E2's scaling-invariance question is real, not a formality.
* Bootstrap draws are not independent samples; never run a rank test over them.
