# HANDOFF (living document; update at every milestone)

Last update: 2026-10-23 ~02:15 CDT, by the lead research agent.

## Where we are

- **Phase 0:** approved. Arjun's approval is quoted verbatim in `reports/phase0.md` §10 and `docs/record.md`.
- **Phase 1:** in progress.
  - Estimators are written: `whitebox_ee/`.
  - Planted systems V1–V5 plus formation: `validation/planted.py`.
  - Harness: `validation/run_phase1.py`; summariser `validation/summarise_phase1.py`.
  - Tests: 41 pass (`.venv/bin/python -m pytest -q tests/`).
  - Propositions 2.7.1–2.7.3 are written in `proofs/P7–P9`, with theory amendments M1–M6 in `theory/core.tex` §5. The PDF compiles.
- **Phase 1 dev runs** (seeds 0–2, `runs/phase1/dev/`):
  - Exact: V1 one-hot and noise, V3, V4, V5, all nulls (0 accepted).
  - V2: AUROC 1.0.
  - V1 Fourier fails; it is a reported stress variant, not gated.
  - Sensitivity: N ≤ 500 recovers nothing on V1.
  - FORMATION dev was re-running seeds 0–1 with the per-pair detector labels removed.
- **Next Phase 1 steps:**
  1. Confirm the FORMATION dev results.
  2. Commit, then tag `prereg-phase-1`:
     `git tag -a prereg-phase-1 -m "..."`, then `sha256sum prereg/PHASE_1.md`. Record both in `docs/record.md`.
  3. Run validation seeds 100–104:
     `for c in V1 NULL V2 V3 V4 V5 FORMATION SENS; do OMP_NUM_THREADS=2 nohup .venv/bin/python -m validation.run_phase1 --seeds 100 101 102 103 104 --cells $c --out runs/phase1/val > runs/phase1/val/log_$c.txt 2>&1 & done`
  4. Fill `docs/record.md` rows P1a–P6 by the letter of `prereg/PHASE_1.md`.
  5. Write `reports/phase1.md` and self-gate.
- **Phase 2:**
  - `prereg/PHASE_2.md` is a DRAFT; not tagged.
  - Cell runner: `bench/k12.py`. Slurm template: `runs/slurm/k12.sbatch`.
  - LLM wrapper `whitebox_ee/llm.py` (HFSystem), tasks in `bench/tasks.py`, baselines in `whitebox_ee/baselines.py` (tested in `tests/test_llm.py`).
  - Dev run of the full pipeline on Qwen2.5-1.5B (development model, excluded from prereg lists): `bench/dev/llm_dev.py`, log `runs/dev_llm.log`. The HF download was slow and stalled once.

## Cluster (Pinnacle)

- The ControlMaster died at about 01:20 on 2026-10-23. The coordinator ordered **no ssh until it sends "PINNACLE BACK"**.
- Submitted before the drop:
  - env build attempt 1: job 1364346, FAILED (bitsandbytes 0.45 wheel unavailable for glibc 2.17);
  - attempt 2: job 1364364 (bitsandbytes 0.42.0). Status unknown.
- When Pinnacle is back:
  1. `sacct -j 1364364`; check `/scrfs/storage/paradkar/envs/wb-ee/.wb_ok`.
  2. Sync the code: `git archive HEAD | ssh pinnacle 'mkdir -p ~/wb-ee && tar xf - -C ~/wb-ee'` and `git rev-parse HEAD > COMMIT`. There is no rsync on the laptop.
  3. Run a smoke test of `bench.k12` on pythia-1.4b on gpu72.
  4. Submit Phase 2 cells after the tag.
  5. Append a `[wb]` line to `~/research_status.md`.
- Share limits:
  - GPU: ≤ 96 gpu-QOS cores and ≤ 8 submitted GPU jobs;
  - CPU: ≤ 64 comp cores;
  - wait if the account is above 80% of any limit.
- Namespace: jobs `wb-*`; data under `/scrfs/storage/paradkar/wb/`.

## Key design decisions (all recorded in theory §5 and in prereg/PHASE_1.md)

- **M1:** parents are interventional direct effects, with a within-fibre swap as the null.
- **M2:** edge activity, normalised by the variation available.
- **M3:** leaf rule (a node counts on x only if C(x) ∈ Rg(C)).
- **M4:** canonical realisers:
  - delta-mode ridge probes;
  - coarsening and separable-composite rules;
  - members' centroids refitted to the node partition;
  - interventions at the primary site only.
- **M5:** fixed point, with suspended nodes and pending causal use.
- **M6:** norm-matched random nulls, per-behaviour criterion (i), κ_max = 0.6.
- **LLM sites:** causal order = layer (`probe.site_layer`); delta mode uses `prev_site` (same role, previous layer).
- **Not gated:** the unsupervised proposers (SAE, ISA, pullback). They do not yet recover factorial categorical codes.

## Do not

- Do not edit `CLAUDE.md`.
- Do not push.
- Do not send partner letters.
- Do not download gated or credentialed data.
