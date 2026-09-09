# HPCC jobs (Arkansas cluster) — written locally, run over SSH after the local blocks

All jobs: checkpoint per unit of work, `--resume` flag, log to `logs/`, results as CSV/JSON under `results/<job>/`.
Environment recipe: `hpcc/env.sh` (to be written with the cluster's module names; needs the TL venv recipe from
`~/Whitebox/v2-programme/.venv-tl` — transformer_lens + sae_lens — for the K jobs, plus `pip install -e ~/Whitebox/whitebox-v2[dev]`).
Every script is import-checked locally (`python -c "import ast; ast.parse(open(f).read())"`) but **not executed** — no local
GPU holds an 8–9B model; first cluster run of each job is a smoke run (`--n-seq 8`, `--n 2`, one layer / width / revision).

| Job | Handbook | Needs | Status (2026-09-09) |
|---|---|---|---|
| `f12_frontier_audit.py` | F12 | Llama-3-8B or Gemma-2-9B (HF token), `whiteboxtopo.symmetry.llama` backend | WRITTEN; verify identity on the cluster first (`identity.json`) |
| `k7_width_scaling.py` | K7 | Gemma-2-2B + Gemma Scope width ladder 16k→1M (HF token), TL venv | WRITTEN — **labelled a replication of Engels et al. 2410.14670** (NOVELTY_BOARD #4) |
| `k8_frontier_residue.py` | K8 | Gemma-2-9B + Gemma Scope 9b-pt-res canonical, TL venv, ≥ 40 GB | WRITTEN; produces the K1 cache format and reruns `run_K2_K6` k2/k3/k4_k5 unchanged |
| `m4_cross_family.py` | M4 | cached CNN + LM batteries (`benchmarks/results/battery_*`, rsync them), trains a ViT-S/8 battery (18 models, ~1 GPU-h) | WRITTEN; reports the full cross-family AUC matrix |
| `c4_scale.py` | C4 | Gemma-2-9B, spaCy `en_core_web_sm`, ≥ 40 GB | WRITTEN; k-sparse MLP-hidden dictionary (a residual Gemma Scope SAE cannot be gauge-audited — stated in the docstring) |
| `b5_family_calibration.py` | B5 | GPT-2, Pythia-410M, Gemma-2-2B; `experiments/B_bias` | WRITTEN; shift cells = FPR, stretch cells = power (B AMENDMENTS #1), n requirement per family |
| `v1_ensemble_large.py` | V1 (large arms) | `experiments/V_legibility/run_V1.py` (reused), GPU per shard | WRITTEN; widths 1024/2048, tf 6/8 layers, 3 seeds, sharded |
| `v7_pythia_drift.py` | V7 | Pythia-410M / 1B, 154 revisions | WRITTEN; head/MLP-level EAP graph around the mid layer, fixed 2 000-token probe |
| `l7_end_to_end.py` | L7 | Gemma-2-9B, `benchmarks/l6_end_to_end_gpt2.py` (reused), K8 cache optional | WRITTEN; writes `L7_timing.json` incl. a `does_not_scale` list |

Run order on the cluster: F12 (identity first) → K8 → K7 → C4 → B5 → M4 → V1-large (sharded) → V7 → L7.
