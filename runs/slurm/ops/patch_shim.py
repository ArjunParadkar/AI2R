import sys
p = sys.argv[1]; s = open(p).read()
a = '  timeout --signal=TERM --kill-after=60 $((END - $(date +%s))) python -m bench.k12 $line --gate_exit --out $R\n'
b = ('  RUN="-m bench.k12"   # 8-bit cells only: loader shim (transformers 4.49 + bnb 0.42 dispatch .to; verified job 1368749)\n'
     '  [[ "$line" == *--load_in_8bit* ]] && RUN=/scrfs/storage/paradkar/wb/shim/bnb8_k12.py && echo "[wb] 8-bit cell: via $RUN"\n'
     '  timeout --signal=TERM --kill-after=60 $((END - $(date +%s))) python $RUN $line --gate_exit --out $R\n')
assert s.count(a) == 1; s = s.replace(a, b); open(p, "w").write(s); print("patched")
