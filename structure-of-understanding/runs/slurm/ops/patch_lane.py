import sys
p = sys.argv[1]; s = open(p).read()
a = 'export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1\n'
b = a + ('# bitsandbytes 0.42 (pre-registered) has no cuda124 binary and cannot find libcudart.so.12 (torch cu124 wheels):\n'
         '# load its cuda123 build against the CUDA 12 runtime torch already ships (job 1368092, cell 31). Ops only.\n'
         'NV=/scrfs/storage/paradkar/envs/wb-ee/lib/python3.11/site-packages/nvidia\n'
         'export BNB_CUDA_VERSION=123\n'
         'export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cusparse/lib:$NV/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n')
assert s.count(a) == 1; s = s.replace(a, b)
c = '  [[ -f $ST/failed/$i ]] && return 1          # crashed (not a timeout): needs a look, never retried blindly\n'
d = ('  if [[ -f $ST/failed/$i ]]; then            # crashed (not a timeout): needs a look, never retried blindly\n'
     '    grep -q "^hold-bnb" $ST/failed/$i 2>/dev/null || return 1   # hold-bnb: 8-bit cell held for lanes with the bnb fix (pre-fix lanes skip it)\n'
     '  fi\n')
assert s.count(c) == 1; s = s.replace(c, d)
open(p, "w").write(s)
print("patched")
