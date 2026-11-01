import sys
p = sys.argv[1]; s = open(p).read()
a = 'left=0; for i in $(cat "$ORDER"); do eligible $i && { left=1; break; }; done\n'
b = a + ('LANE=${SLURM_JOB_NAME#wb-k12-}\n'
         'if [[ -e $ST/RETIRE_$LANE ]]; then left=0; echo "[wb] lane retired on purpose ($ST/RETIRE_$LANE)"; fi   # lane swap S->L keeps 4 GPU lanes\n')
assert s.count(a) == 1; s = s.replace(a, b)
open(p, "w").write(s); print("patched")
