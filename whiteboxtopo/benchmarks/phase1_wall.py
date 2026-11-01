"""Where is the PH wall? Exact PH (giotto-ph, no collapse, thresh 95, PCA->30)
on GPT-2 124M layer-12 activations for growing probe sizes, 20 threads."""
import csv, json, sys, time, os, resource
import numpy as np, torch, whiteboxtopo as wb
from common import probe_sentences, gpt2_activations, RESULTS
ns = [int(x) for x in sys.argv[1:]] or [2000, 3000, 4000, 6000, 8000]
sents = probe_sentences(max(ns), offset=0)
_, acts = gpt2_activations("gpt2", sents, [12]); A = acts[12]; red = wb.fit_reducer(A[:500], 30)
rows = []
for n in ns:
    t0 = time.time(); D = wb.compute_diagrams(A[:n], red, backend="giotto-ph", distance_device="cuda"); dt = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    rows.append(dict(n=n, seconds=dt, TC=wb.topological_complexity(D[1]), n_bars=len(D[1]), peak_rss_gb=rss))
    print(rows[-1], flush=True)
    with open(f"{RESULTS}/phase1_wall.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
