# Block M — amendments

**#1 (2026-09-07, before the M1 run; after a timing smoke on one cloud).** M1 renaming budget
reduced from 100 to 30 per kind (perm / scale / combined) — each feature evaluation carries its own
3-shuffle null and the full budget was ~20 min per cloud for 228 clouds. Verdict thresholds
unchanged (CV < 0.01 on the median component, per feature).

**#2 (2026-09-09 12:20) M1 restarted for CPU oversubscription.** The cpu-queue M1 stage (6 pool workers) ran with default
BLAS threading: each worker held ~40 threads, load average 93 on 20 cores, 10 of 216 clouds in 33 min (≈ 12 h projected)
while slowing V6 and V2/V3. `haru_urara.py` made resumable (append + skip finished (model, layer) pairs); the stage was killed
(its 10–14 finished rows are kept) and relaunched as unit `wb-m1` with `OMP/MKL/OPENBLAS_NUM_THREADS=3`. The cpu queue
records M1 as failed and proceeds to Z4/Z5; `logs/M1.done` is touched by the standalone unit.
