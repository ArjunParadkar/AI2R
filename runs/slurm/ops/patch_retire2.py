import sys
p = sys.argv[1]; s = open(p).read()
a = '  now=$(date +%s); (( END - now < MIN_START )) && { echo "[wb] lane time nearly used"; break; }\n'
b = a + '  [[ -e $ST/RETIRE_${SLURM_JOB_NAME#wb-k12-} ]] && { echo "[wb] retire flag set: no new cell"; break; }\n'
assert s.count(a) == 1; s = s.replace(a, b)
open(p, "w").write(s); print("patched")
