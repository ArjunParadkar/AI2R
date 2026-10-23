# HANDOFF (living document; update at every milestone)

Last update: 2026-10-23 ~05:00 CDT, by the lead research agent.

## Where we are

- **Phase 0:** approved. The approval is quoted verbatim in `reports/phase0.md` §10.
- **Phase 1:** COMPLETE (`reports/phase1.md`).
  - Tag `prereg-phase-1` at commit bc6036a; SHA-256 recorded in `docs/record.md`.
  - Validation seeds 100–104 are in `runs/phase1/val/`; verdicts in `runs/phase1/val/VERDICTS.md` (computed by `validation/verdicts_phase1.py`).
  - All gated criteria pass except **P6b** (formation; answer-funnel limitation, declared before the run). Formation is blocked for K14, and the unsupervised proposers are not gated.
  - Record rows are in `docs/record.md`, cut entries in `docs/cut.md`, function statuses in `docs/CLAIMS.md`.
  - Theory amendments M1–M8 are in `theory/core.tex` §5 (PDF built). Propositions 2.7.1–2.7.3 are in `proofs/P7–P9`.
- **Phase 2:** `prereg/PHASE_2.md` is a DRAFT, not tagged; the answer-funnel diagnostic is in §4.3.
  - Cell runner: `bench/k12.py` (K1 + K2 + funnel diagnostic). It still needs an end-to-end test on a tiny model; do that before the tag.
  - LLM dev run (Qwen2.5-1.5B, add2, development model): `runs/dev_llm.log`. It was still in the quotient stage at 05:00. The cost is dominated by quotient tests: 258 candidates × (20 random realisers × 128 + 1024) rows.
  - **Before freezing PHASE_2:** set LLM-specific discovery settings (`n_random_realisers`, `n_pairs`) from the dev timing, test `k12` end to end, then tag and record the hash.
- **Run models that fit locally** (laptop GPU, 8 GB): pythia-1.4b and -2.8b in bf16. Everything ≥ 7B goes to Pinnacle.
- **Thread oversubscription:** always set `OMP_NUM_THREADS`/`MKL_NUM_THREADS` (4 for a single LLM process, 2 when running many planted processes).

## State at 2026-10-23 ~09:40 (Pinnacle down again since 08:44; wait for "PINNACLE BACK")

- **Laptop rule (Arjun):** nothing heavier than a sub-minute single-process test runs on the laptop. Everything else goes to Pinnacle through `runs/slurm/cmd_array.sbatch`, which runs line i of a command file.
- **Env:** `/scrfs/storage/paradkar/envs/wb-ee` is built (job 1364678: torch 2.5.1, transformers 4.49, bnb 0.42).
- **Amendment 1:** P2 FAILED on seed 105 (recorded).
- **Amendment 2 (M9 step-level leaf rule; dual-mode realisers; coarsening rule removed; V2D/FORMD):** drafted in `prereg/PHASE_1_amend_2.md`, not tagged.
- **Queued when Pinnacle is back:**
  1. Fetch `/scrfs/storage/paradkar/wb/results/phase1/dev_a2` (job 1364717, old code; informational only) and the LLM diagnostics (job 1364727 logs).
  2. Sync the code: `git archive HEAD | ssh pinnacle "tar xf - -C ~/wb-ee"` and write COMMIT.
  3. Submit `runs/cells/dev_a3.txt` (24 lines) on cloud72 (`-c 4 --array=1-24%10`) and `runs/cells/llmdev_a2.txt` on agpu06 (`-c 24 --gres=gpu:1 -t 06:00:00`).
  4. If dev_a3 is clean, tag `prereg-phase-1-amend-2`, record the hash, then submit `runs/cells/val_a2.txt` (50 lines) and compute verdicts with `validation/verdicts_phase1.py DIR 110,111,112,113,114` (extend the script for V2D/FORMD first).
  5. Then finalise and tag `prereg/PHASE_2.md`, and submit the K1/K2 cells (`runs/slurm/k12.sbatch`).

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
