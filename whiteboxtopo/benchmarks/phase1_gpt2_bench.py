"""Phase 1 deliverable check: topological_film on real transformers, timed.

1. GPT-2 124M: PH wall-clock vs probe size n in {150,300,500,1000,2000} for
   ripser / giotto-ph(+collapse) / landmark200, layer 12, PCA->30, thresh 95.
2. Full film frame with uncertainty (5 layers, n=500, 20 bootstrap subsamples
   of 250): the cost a lab pays per epoch.
3. Scale-up: gpt2-medium (355M), gpt2-large (774M), gpt2-xl (1.5B) fp16 —
   extraction + one 5-layer frame at n=500.
CSV + JSON in benchmarks/results.
"""
from __future__ import annotations
import csv, json, os, time
import numpy as np, torch
import whiteboxtopo as wb
from common import probe_sentences, gpt2_activations, RESULTS

os.makedirs(RESULTS, exist_ok=True)
rows, report = [], {}
sents = probe_sentences(2000)

t0 = time.time(); model, acts = gpt2_activations("gpt2", sents, [0, 3, 6, 9, 12]); t_ex = time.time() - t0
report["gpt2_extract_2000_seconds"] = t_ex; print(f"extract 2000 x 5 layers: {t_ex:.1f}s", flush=True)
A = acts[12]; red = wb.fit_reducer(A[:500], 30)

for n in (150, 300, 500, 1000, 2000):
    X = A[:n]
    for backend, kw in (("ripser", {}), ("giotto-ph", {}), ("giotto-ph-nocollapse", dict(collapse_edges=False)), ("landmark", dict(n_landmarks=200))):
        if backend == "ripser" and n > 1000: continue
        b = "giotto-ph" if backend.startswith("giotto") else backend
        t0 = time.time(); D = wb.compute_diagrams(X, red, backend=b, distance_device="cuda", **kw); dt = time.time() - t0
        rows.append(dict(model="gpt2", n=n, backend=backend, seconds=dt, TC=wb.topological_complexity(D[1]), n_bars=len(D[1])))
        print(f"n={n:5d} {backend:22s} {dt:7.2f}s TC={rows[-1]['TC']:.3g} bars={len(D[1])}", flush=True)

# full frame with uncertainty
sub = {l: acts[l][:500] for l in acts}
t0 = time.time(); film = wb.film_from_activations([sub], n_boot=20, subsample=250, distance_device="cuda"); dt = time.time() - t0
report["gpt2_frame_5layers_n500_boot20_seconds"] = dt
report["gpt2_frame_tc"] = {str(l): repr(film.tc(l)[0]) for l in film.layers}
print(f"5-layer frame n=500 + 20 boots: {dt:.1f}s", flush=True); print(report["gpt2_frame_tc"], flush=True)
del model; torch.cuda.empty_cache()

for name in ("gpt2-medium", "gpt2-large", "gpt2-xl"):
    try:
        t0 = time.time(); model, ac = gpt2_activations(name, sents[:500], None or [0, 6, 12, 18, 24], dtype=torch.float16, batch=16)
        nl = model.config.n_layer; t_ex = time.time() - t0
        t0 = time.time(); film = wb.film_from_activations([ac], n_boot=20, subsample=250, distance_device="cuda"); t_ph = time.time() - t0
        report[name] = dict(n_params=sum(p.numel() for p in model.parameters()), extract_seconds=t_ex, frame_seconds=t_ph,
                            vram_gb=torch.cuda.max_memory_allocated() / 1e9, tc={str(l): film.tc(l)[0].point for l in film.layers})
        print(name, json.dumps(report[name]), flush=True)
        del model; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        report[name] = {"error": repr(e)}; print(name, "FAILED", e, flush=True)
        torch.cuda.empty_cache()

with open(f"{RESULTS}/phase1_gpt2_timing.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(report, open(f"{RESULTS}/phase1_gpt2_report.json", "w"), indent=1)
print("done")
