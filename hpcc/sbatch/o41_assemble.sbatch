#!/bin/bash
# O4.1 Bamboo Memory, step 9: read F12 / C4 / K8 / L7 / O4.1-P0 and write one scorecard per T4 model.  CPU only — no
# model is loaded; the cost is np.linalg.eigvalsh on the (20 000 x d) K8 residual and its two MP nulls, 3 layers x 3
# decompositions (minutes).  __PARTITION__ is filled by hpcc/configure.sh with the GPU partition; change it by hand to a
# CPU queue if the allocation prefers one (this job requests no GRES).
#
# DEPENDENCY.  The job ids are only known at submission, so the dependency is written on the command line, not here:
#
#   f12=$(sbatch --parsable hpcc/sbatch/f12_frontier_audit.sbatch)
#   c4=$(sbatch  --parsable hpcc/sbatch/c4_scale.sbatch)
#   k8=$(sbatch  --parsable hpcc/sbatch/k8_frontier_residue.sbatch)
#   l7=$(sbatch  --parsable hpcc/sbatch/l7_end_to_end.sbatch)
#   p0=$(sbatch  --parsable hpcc/sbatch/o41_frontier_p0.sbatch)
#   sbatch --dependency=afterany:$f12:$c4:$k8:$p0 hpcc/sbatch/o41_assemble.sbatch      # add :$l7 to wait for L7 too
#
# afterany, never afterok: a scorecard whose inputs failed or timed out is still a deliverable — every missing input
# becomes a not_reached cell carrying the reason (Operating Constitution #4).  Re-run it any time more inputs land; it
# overwrites its own two files and touches nothing else.
#SBATCH --job-name=wb-o41_assemble
#SBATCH --partition=__PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=hpcc/logs/%x-%A_%a.out
cd "${SLURM_SUBMIT_DIR:?submit from the whitebox-v2 checkout root}"; source hpcc/env.sh; set -x
for m in meta-llama/Meta-Llama-3-8B google/gemma-2-9b; do
  python experiments/O_opener/bamboo_memory.py --assemble --model "$m" --results hpcc/results --out hpcc/results/o41
done
