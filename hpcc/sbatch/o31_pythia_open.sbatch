#!/bin/bash
#SBATCH --job-name=wb-o31_pythia_open
#SBATCH --partition=__PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --gres=__GRES__
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=hpcc/logs/%x-%A_%a.out
# O3.1 Hishi Akebono — T3 Pythia P7/P3 over all 154 public checkpoints (experiments/O_opener/SPEC_O3.md).
# Cost (laptop-measured compute + download): per-revision compute with the frozen settings is 23 s on a shared RTX 5070
# for pythia-410m (18.6-27.4 s over two revisions, weights already on disk) and 11 s for pythia-70m; each pythia-410m
# revision is 1.6 GB on disk and each pythia-1b revision ~4 GB, so the 154 revisions are 246 GB / ~616 GB of HF
# downloads and the download, not the GPU, sets the wall time. At a 100 MB/s link: 410m ~2.5 h, 1b ~4 h; at 30 MB/s:
# ~4 h and ~8 h. --time is set to 12 h to cover the slow-link case; the job is resumable (one trajectory-CSV row per
# revision + a cloud cache), so a time-out is RE-SUBMITTED, never restarted.
# --purge deletes each revision's HF snapshot after use, so only one checkpoint (<= 4 GB) is ever on scratch. Do NOT
# run this concurrently with v7_pythia_drift on the same HF_HOME: V7 reads the same snapshots and --purge would pull
# them out from under it. If the two array tasks saturate the link, throttle with `sbatch --array=0-1%1`.
cd "${SLURM_SUBMIT_DIR:?submit from the whitebox-v2 checkout root}"; source hpcc/env.sh; set -x
M=(EleutherAI/pythia-410m EleutherAI/pythia-1b)
python experiments/O_opener/hishi_akebono.py --model ${M[$SLURM_ARRAY_TASK_ID]} --out hpcc/results/o31 --resume --purge
# after BOTH array tasks finish (run once, no GPU needed):
#   python experiments/O_opener/hishi_akebono.py --aggregate --out hpcc/results/o31
