# HANDOFF (living document; update at every milestone)

Last update: 2026-10-23 ~11:05 CDT, by the lead research agent (resumed session).

## STATE AT 11:05 (READ THIS FIRST; supersedes the sections below where they disagree)

- **Amendment-2 validation: P1f FAILED** (seed 114, untrained V4 null: 4/160 = 0.025 > 0.02). P1a–P1e, P3, P4, P5 PASS. P2 (V2D) and P6 (FORMD) are still pending. See `reports/phase1.md` §8 and `docs/record.md`.
  - By prereg/PHASE_1.md §4 the amended estimator may not touch a real model, and **Phase 2 does not run** (`reports/phase2.md`). K2 has no verdict; the kill rule was not triggered.
- **Phase 2 GPU jobs CANCELLED** (1364954 unstarted; lanes 1364995–1364998). Their outputs are discarded unanalysed; nothing is running on GPU.
- **Still running** (CPU, comp QOS, ≤ 64 cores): the long amend-2 cells on comp06, lanes `wb-p1-cA` (1364993: FORMD ×5, V2D 110/111, SENS 110–112) and `wb-p1-cB` (1364994: FORMATION ×5, V2D 112–114, SENS 113/114).
  - They are pending on Priority and estimated to start ~13:00.
  - Formation cells cache per checkpoint in `.../val_a2/_ckpt`. If a job hits its 6 h limit, resubmit the same line: `sbatch --export=NONE,CMDS=runs/cells/val_a2_long_A.txt,PAR=8,NT=4 -J wb-p1-cA -p comp06 -q comp -c 32 -t 6:00:00 --dependency=singleton runs/slurm/pack.sbatch` (and the same for B). Finished cells are skipped.
  - When done: fetch, run `validation/verdicts_phase1.py runs/phase1/val_a2 110,111,112,113,114 > runs/phase1/val_a2/VERDICTS.md`, and replace the three PENDING rows in `docs/record.md`. These results are for the record only; they cannot change the gate.
- 1364799 (wb-p1dev3, cloud, 12 cores: FORMD dev seeds 0–2) is informational; let it finish.
- **Next scientific step: amendment 3** (a new development cycle). The post-hoc hypothesis: removing the coarsening rule lets single-input coarsenings (g0–g2 = pos mod λ) pass on untrained copies. Fix the cause, check EVERY development null before tagging (the dev_a3 nulls already showed this and were not inspected), then tag `prereg-phase-1-amend-3` with new seeds (115–119) and re-validate. Phase 2 may need its own amendment if the estimator changes (`reports/phase2.md`).
- **Queue facts (2026-10-23):** the 72 h partitions are days deep for this account (priority 2: gpu72 → 10-25, agpu72 → 10-27, comp72 → 10-30). The 6 h partitions start within hours (agpu06 immediately at 10:40, comp06 ~2 h). comp01 (1 h) turns over fast.
  - Use `runs/slurm/k12_lane.sbatch`: singleton lanes that resubmit themselves and resume by stage, so at most 4 GPU jobs are ever submitted. Arrays count every task toward the 8-job share, so avoid them.
  - `bench/k12.py --gate_exit` stops gate-failing cells after stage 1.
- The Phase 2 analysis-code issues found before any data are listed in `reports/phase2.md`: the pooling unit, the K1 untrained-null condition, and the family anchor. Fix them in a Phase 2 amendment before any run.


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

## State at 2026-10-23 ~10:50 (READ THIS FIRST)

**Tags**
- `prereg-phase-1` (bc6036a)
- `prereg-phase-1-amend-1` (9a57d93)
- `prereg-phase-1-amend-2` (95a65da)
- `prereg-phase-2` (25b772e)

Hashes and SHA-256 are in `docs/record.md`; the OSF texts are in `prereg/osf/` (Arjun uploads).

**Phase 1 history**
- The original validation passed everything except P6b.
- Amendment 1 (principal-subspace realisers) failed P2 (recorded).
- Amendment 2 adds M9 (step-level leaf rule), dual-mode delta+plain realisers, removal of the coarsening rule, and V2D/FORMD as the gated memorization and formation systems (V2/FORM become controls).
  - Its dev check (job 1364799, seeds 0–2) was clean: V1, V3, V4 and V5 exact, NULL 0, V2D AUROC 0.994/0.999. FORMD dev was still running at the tag.
  - **Its validation (seeds 110–114) is running.** Results go to `/scrfs/storage/paradkar/wb/results/phase1/val_a2`.
    - 1364949 (long cells: V2D, FORMD, FORMATION, SENS) on gpu72, packed, 32 cores.
    - 1364950/1364951 (short cells) on comp01.
    - 1364952/1364953 are HELD to respect the 64-core CPU share. Release with `scontrol release 1364952` when 1364950 ends, then 1364953.
  - When everything is done:
    1. Fetch with `ssh pinnacle 'cd .../val_a2 && tar cf - *.json' | tar xf - -C runs/phase1/val_a2`.
    2. Run `.venv/bin/python validation/verdicts_phase1.py runs/phase1/val_a2 110,111,112,113,114 > runs/phase1/val_a2/VERDICTS.md`.
    3. Add the record rows (same parser as before), and update `reports/phase1.md` §8 and `docs/CLAIMS.md`.

**Phase 2**
- Pre-registration frozen in `prereg/PHASE_2.md`. Cells: `runs/cells/phase2.txt` (39).
- Running as array 1364954 (`wb-k12`, agpu72, `%2`, 24 cores each). Results go to `/scrfs/storage/paradkar/wb/results/k12/<cell>/{k1.json,graph.pkl,k2.pkl,log.txt}`; logs to `/scrfs/storage/paradkar/wb/logs/wb-k12-1364954_<i>.out`.
- Do NOT analyse before the amend-2 validation verdict (see the note in `docs/record.md`).
- Analysis: `.venv/bin/python -m bench.analyse_k12 runs/k12 --out reports/phase2_tables.json`. Exclude the pythia-160m smoke cell, which is a development model.
- If K2 FAILS, the programme stops: write `reports/phase2.md` with everything and start no Phase 3–8 runs.
- Raise `%2` to `%4` if the gpu QOS has room. My share is ≤ 96 cores: the val-long job uses 32 while it runs.

**Rules from Arjun and the coordinator (2026-10-23)**
- The laptop is only for editing, git and sub-minute single-process tests.
- QOS cloud ≤ 16 cores at once.
- Bulk CPU work goes to comp72/comp01 as packed 32-core jobs (`runs/slurm/pack.sbatch`; `CMDS`, `PAR`, `NT`), within ≤ 64 comp cores.
- GPU ≤ 96 gpu-QOS cores and ≤ 8 submitted GPU jobs.
- Poll at most every 10 minutes. ≤ 2 concurrent ssh, each wrapped in `timeout`.
- If ssh hangs, stop, wait 10 minutes and retry once. The ControlMaster lasts until about 20:00 on 2026-10-23.

**Runners**
- `runs/slurm/cmd_array.sbatch`: an array that runs line i of `$CMDS`.
- `runs/slurm/pack.sbatch`: packs many lines into one node.
- `runs/slurm/k12.sbatch`: an array of `bench.k12` argument lines.
- Sync code with `git archive HEAD | ssh pinnacle "tar xf - -C ~/wb-ee"` and write `COMMIT`.

**Key development facts for LLMs** (Qwen2.5-1.5B, development model)
- Addition is computed in a narrow late band (about L22–26 of 28) at the answer positions. Operand positions stay causal until then.
- Plain principal-subspace realisers work there (s0: A = 0.36 against η = 0.02); delta-mode realisers mostly do not.
- The full pipeline found s0 at L25@pre_ans_d0 ← {a0, b0} (IIA 0.97) with out0 ← s0. For out1 it found no intermediate (c0 and s1 missing), so the realisers capture the column sum but not yet the carry.

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
