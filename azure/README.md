# Azure launcher — the whole remaining sprint compute in one day

Runs the SAME job commands as `hpcc/sbatch/*.sbatch` (one GPU VM per job) and writes into the same `hpcc/results/` namespace, so
PAUSE reports, `hpcc/RUNBOOK.md` result paths and `sync_down`-style merging all stay valid. Nothing here changes any protocol,
cap or budget except `F2_VALID_BUDGET` (300 M on the 56 GB T4 VM vs 150 M laptop / 500 M cluster — a memory limit, F AMENDMENTS #12).

## 0. What Arjun supplies
1. **Access** — either `az login` (device code) in this shell, or a service principal with Contributor on the subscription:
   `az login --service-principal -u APP_ID -p SECRET --tenant TENANT_ID`.
2. **GPU quota** (the real critical path; new subscriptions have 0). In the portal: *Quotas → Compute → region* request
   - `Standard NCASv3_T4 Family vCPUs`: **200** (20 F2 shards × 8 + 15 small jobs × 8, launched in two waves)
   - `Standard NCADS_A100_v4 Family vCPUs`: **96** (4 × NC24ads_A100_v4 for the gemma-2-9b jobs K8/C4/F12/L7)
   in ONE region (default `southcentralus`; `eastus`/`westus3` are fine — set `AZ_LOC`). Small requests are often auto-approved in minutes.
3. **`HF_TOKEN`** with the Gemma licence accepted (K7/K8/C4/F12/L7).

## 1. Run
```bash
cd ~/Whitebox/whitebox-v2
export HF_TOKEN=hf_...            # never commit
bash azure/provision.sh --with-ckpt   # rg + storage + repo tarball (+2.2 GB V1 ckpts for v5); writes azure/config.local.env (gitignored)
bash azure/launch.sh f2               # wave 1: 20 F2 shards (T4)
bash azure/launch.sh k7,m4,b5,v7,v1,v5,k8,c4,f12,l7   # wave 2 (or `all` if quota allows)
bash azure/status.sh                  # job board; `bash azure/status.sh log f2-s3` tails a job log
bash azure/collect.sh                 # pull results+logs, merge F2 shards into experiments/F_gauge/results, delete finished VMs (repeatable)
bash azure/teardown.sh                # after the final collect: deletes the whole resource group
```
Every job is `--resume`-able; `launch.sh` skips jobs with a done marker or a live VM, so re-running it after evictions/quota bumps is safe.
Spot VMs: `AZ_SPOT=1 bash azure/launch.sh …` (60–80 % cheaper; a results snapshot goes to blob every 10 min).

## 2. How it works
`provision.sh` uploads the checkout (same excludes as `hpcc/sync_up.sh`; the laptop's F2 status CSV is inside, so cloud shards skip cells
the laptop already finished) and mints a 48 h container SAS. `launch.sh` renders `cloud-init.yaml` per job (`render.py`, literal
substitution) → VM (Ubuntu 24.04 + NVIDIA driver extension) → systemd `wb-job` runs `worker.sh`: wait for driver → fetch tarball → venv per
`hpcc/env.sh` recipe (torch cu128) → run the command (3 attempts) → tar `hpcc/results` to blob → `markers/<job>.done|failed` → shutdown.
Logs stream to `logs/<job>.log` in blob every 2 min. `collect.sh` overlays the tarballs onto `hpcc/results/` and merges F2 shards by
specimen (dedup against the laptop's rows, all four CSVs).

## 3. Timing / cost (on-demand list prices, 2026)
| wave | VMs | wall | cost |
|---|---|---|---|
| F2 tail, 20 × NC8as_T4_v3 | 20 | ≤ 12 h (worst cell seen: 8 h) | ~$180 |
| small jobs (k7 m4 b5 v7×2 v1×8 v5) | 14 × T4 | ≤ 12 h (v5 longest) | ~$130 |
| gemma-2-9b jobs (k8 c4 f12 l7) | 4 × A100 80 GB | ≤ 12 h | ~$180 |
Total well under $1 000; Spot roughly a third of that. Wall clock ≈ quota approval + ~15 min VM/venv setup + the longest single job.

## 4. Laptop overlap
The laptop tail (`wb-gpu-queue`) keeps running; stop it once the cloud shards are up (`systemctl --user stop wb-gpu-queue`) to avoid
duplicate work — the merge dedups by specimen either way.
