# HPCC jobs (Arkansas cluster) — written locally, run over SSH after the local blocks

All jobs: checkpoint per unit of work, `--resume` flag, log to `logs/`, results as CSV.
Environment recipe: `hpcc/env.sh` (to be written with the cluster's module names).

| Job | Handbook | Needs | Status |
|---|---|---|---|
| `f12_frontier_audit.py` | F12 | Llama-3-8B or Gemma-2-9B (HF token), `whiteboxtopo.symmetry.llama` backend | WRITTEN (backend + checkpointed script); verify identity on the cluster first (`identity.json`) |
| `k7_width_scaling.py` | K7 | Gemma Scope width ladder 16k→1M (HF token) | TODO |
| `k8_frontier_residue.py` | K8 | Gemma-2-9B + Gemma Scope | TODO |
| `m4_cross_family.py` | M4 | v1 battery + ViT family | TODO |
| `c4_scale.py` | C4 | Gemma-2-9B + Gemma Scope | TODO |
| `b5_family_calibration.py` | B5 | GPT-2, Pythia-410M, Gemma-2-2B | TODO |
| `v1_ensemble_large.py` / `v7_pythia_drift.py` | V1 (large arms), V7 | Pythia-410M/1B 154 checkpoints | TODO |
| `l7_end_to_end.py` | L7 | Gemma-2-9B | TODO |
