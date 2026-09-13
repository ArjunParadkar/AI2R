#!/usr/bin/env bash
# Runs ON the VM (installed by cloud-init as /opt/wb/worker.sh, started by systemd unit wb-job).
# Env (from /opt/wb/job.env): JOB, CMD, TARBALL_URL, CKPT_URL (optional), BLOB_BASE (container URL), SAS (query string), HF_TOKEN
set -uo pipefail; source /opt/wb/job.env; cd /opt/wb; mkdir -p logs
LOG=/opt/wb/logs/$JOB.log; exec >>"$LOG" 2>&1
put() { curl -sS -X PUT -H "x-ms-blob-type: BlockBlob" --data-binary "@$2" "$BLOB_BASE/$1?$SAS" -o /dev/null || echo "put $1 failed"; }
mark() { echo "$2" > /tmp/m; put "markers/$JOB.$1" /tmp/m; }
pushlog() { while true; do put "logs/$JOB.log" "$LOG"; sleep 120; done; }
echo "=== $JOB start $(date -u +%FT%TZ) on $(hostname) ==="; mark started "$(date -u +%FT%TZ)"; pushlog & PL=$!
# 1. wait for the NVIDIA driver extension (installed by `az vm extension set` in launch.sh; can land after cloud-init)
for i in $(seq 1 90); do nvidia-smi -L >/dev/null 2>&1 && break; sleep 20; done; nvidia-smi -L || { mark failed "no GPU driver after 30 min"; kill $PL; shutdown -h now; exit 1; }
# 2. payload
[ -d whitebox-v2 ] || { curl -sSfL "$TARBALL_URL" -o repo.tgz && mkdir whitebox-v2 && tar xzf repo.tgz -C whitebox-v2 && rm repo.tgz; } || { mark failed "tarball"; kill $PL; shutdown -h now; exit 1; }
if [ -n "${CKPT_URL:-}" ] && [ ! -d whitebox-v2/experiments/V_legibility/ckpt ]; then curl -sSfL "$CKPT_URL" -o ckpt.tgz && tar xzf ckpt.tgz -C whitebox-v2 && rm ckpt.tgz; fi
cd whitebox-v2
# 3. environment (same recipe as hpcc/env.sh; cu128 wheels need driver >= 570, which the extension ships)
export HF_TOKEN HF_HOME=/opt/wb/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ ! -d .venv-hpcc ]; then
  python3 -m venv .venv-hpcc && .venv-hpcc/bin/pip install -q -U pip \
  && .venv-hpcc/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 \
  && .venv-hpcc/bin/pip install -q -e ".[dev]" transformers accelerate transformer_lens sae_lens spacy pyarrow \
  && .venv-hpcc/bin/python -m spacy download -q en_core_web_sm || { mark failed "venv"; kill $PL; shutdown -h now; exit 1; }
fi
export PATH=/opt/wb/whitebox-v2/.venv-hpcc/bin:$PATH; mkdir -p hpcc/logs hpcc/results
.venv-hpcc/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 4. the job (identical command line to the sbatch), results snapshot to blob every 10 min so an eviction loses <= 10 min
( while true; do sleep 600; tar czf /tmp/res.tgz hpcc/results 2>/dev/null && put "results/$JOB.tgz" /tmp/res.tgz; done ) & RS=$!
for attempt in 1 2 3; do echo "+ [$attempt/3] $CMD"; bash -c "$CMD"; RC=$?; [ $RC -eq 0 ] && break; echo "rc=$RC, retry in 60 s (jobs are --resume-able)"; sleep 60; done
kill $RS 2>/dev/null; tar czf /tmp/res.tgz hpcc/results && put "results/$JOB.tgz" /tmp/res.tgz
echo "=== $JOB end rc=$RC $(date -u +%FT%TZ) ==="; put "logs/$JOB.log" "$LOG"
[ $RC -eq 0 ] && mark done "$(date -u +%FT%TZ)" || mark failed "rc=$RC"
kill $PL 2>/dev/null; shutdown -h now   # collect.sh deletes the VM once its done marker exists; a failed VM stays stopped (deallocate manually) for inspection
