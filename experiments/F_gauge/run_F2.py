"""F2 + F3 + F4(within) — enumerate explanations, hidden symmetries, canonical orbit keys.
Writes results/explanations.csv (one row per minimal circuit x criterion), results/symmetry.csv,
results/enumeration_status.csv. Resumable per specimen. --kill runs the F2/F4 planted kill tests."""
import os, sys, csv, time, argparse, json
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from flib.models import MLP
from flib.functions import truth_table
from flib.enumerate import enumerate_valid, interpretation_count
from flib.minimal import minimal_ids_fast
from flib.interpret import handbook_interpretation, algorithm_string
from flib.gauge import canonical_form, functional_key
from flib.enumerate import circuit_layers
from whiteboxtopo import symmetry as S

R = os.path.join(os.path.dirname(__file__), "results")
EPS = (1e-6, 1e-4)
CAP = 1800.0   # SPEC F2: 30 min per network


def _valid(model, X, y, scheme, use_gpu):
    if use_gpu:
        from flib.gpu_enum import enumerate_valid_torch
        return enumerate_valid_torch(model, X, y, scheme, chunk=1 << 18, time_cap=CAP)
    return enumerate_valid(model, X, y, scheme)


def analyse(model, X, y, spec, use_gpu=False, meloux=True):
    """Returns (explanation rows, status row)."""
    rows, status = [], {"specimen": spec, "n_hidden": sum(model.widths)}
    t0 = time.time()
    for scheme in ("zero", "mean"):
        valid = _valid(model, X, y, scheme, use_gpu)
        if valid is None:
            status[f"enumerated_{scheme}"] = 0; continue
        status[f"enumerated_{scheme}"] = 1
        mins = minimal_ids_fast(valid)
        status[f"n_valid_{scheme}"] = int(len(valid)); status[f"n_minimal_{scheme}"] = int(len(mins))
        for mid in mins:
            mid = int(mid)
            ok, gates, out_table = handbook_interpretation(model, X, mid, scheme)
            nm = int(interpretation_count(model, X, mid, scheme)) if meloux else -1
            keys = {eps: canonical_form(model, mid, eps)[0].hex() for eps in EPS}
            rows.append(dict(specimen=spec, criterion=scheme, circuit_id=mid, n_units=bin(mid).count("1"),
                             interpretable=int(ok), n_meloux=nm, key6=keys[1e-6], key4=keys[1e-4],
                             fkey=hash(functional_key(model, X, mid, scheme)),
                             algorithm=algorithm_string(gates, out_table, circuit_layers(mid, model.widths)) if ok else ""))
    status["wall"] = round(time.time() - t0, 1)
    return rows, status


def job(args):
    spec, path, fn, use_gpu = args
    import torch; torch.set_num_threads(1)
    model = MLP.load(os.path.join(R, path)); X, y = truth_table(fn)
    rows, status = analyse(model, X, y, spec, use_gpu)
    # F3
    h = S.hidden(model, inputs=X, seed=0, keep_directions=False)
    sym = dict(specimen=spec, n_params=h.n_params, rank=h.rank, fiber_dim=h.corank, known_dim=h.known_dim,
               excess=h.excess, dead=h.identified["dead"], duplicate=h.identified["duplicate"],
               constant=h.identified["constant"], unexplained=h.unexplained, G_incomplete=int(h.G_incomplete),
               dead_units=len(h.dead_units), constant_units=len(h.constant_units))
    return rows, status, sym


FIELDS = ["specimen", "criterion", "circuit_id", "n_units", "interpretable", "n_meloux", "key6", "key4", "fkey", "algorithm"]
SFIELDS = ["specimen", "n_hidden", "enumerated_zero", "n_valid_zero", "n_minimal_zero", "enumerated_mean", "n_valid_mean",
           "n_minimal_mean", "wall"]
YFIELDS = ["specimen", "n_params", "rank", "fiber_dim", "known_dim", "excess", "dead", "duplicate", "constant", "unexplained",
           "G_incomplete", "dead_units", "constant_units"]


def _append(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new: w.writeheader()
        for r in rows: w.writerow(r)


def kill_tests():
    from flib.planted import xor3_single, planted_k
    X, y = truth_table("XOR3"); out = {}
    m = xor3_single()
    rows, st = analyse(m, X, y, "planted_xor_single", use_gpu=False)
    out["F2_kill"] = {s: dict(n_minimal=st[f"n_minimal_{s}"], circuits=[r["circuit_id"] for r in rows if r["criterion"] == s],
                             interpretable=[r["interpretable"] for r in rows if r["criterion"] == s],
                             full_set_id=(1 << sum(m.widths)) - 1) for s in ("zero", "mean")}
    for k in (1, 2, 3):
        m, nc = planted_k(k)
        gpu = sum(m.widths) > 20
        rows, st = analyse(m, X, y, f"planted_k{k}", use_gpu=gpu, meloux=False)
        res = {}
        for s in ("zero", "mean"):
            rr = [r for r in rows if r["criterion"] == s and r["interpretable"]]
            res[s] = dict(N=len(rr), K=len(set(r["key6"] for r in rr)), n_minimal=st.get(f"n_minimal_{s}"),
                          enumerated=st.get(f"enumerated_{s}"), wall=st["wall"])
        out[f"F4_kill_k{k}"] = dict(expected_K=k, expected_N=2 * k, **res)
        print(k, res, flush=True)
    with open(os.path.join(R, "kill_tests.json"), "w") as f:
        json.dump(out, f, indent=1, default=int)
    print(json.dumps(out, indent=1, default=int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12); ap.add_argument("--max-hidden", type=int, default=20)
    ap.add_argument("--min-hidden", type=int, default=0); ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--kill", action="store_true")
    a = ap.parse_args()
    if a.kill:
        return kill_tests()
    spec = list(csv.DictReader(open(os.path.join(R, "specimens.csv"))))
    done = set()
    sp = os.path.join(R, "enumeration_status.csv")
    if os.path.exists(sp):
        done = {r["specimen"] for r in csv.DictReader(open(sp))}
    jobs = []
    for r in spec:
        if not int(r["converged"]): continue
        name = f'{r["function"]}_d{r["depth"]}_w{r["width"]}_s{r["seed"]}'
        nh = int(r["depth"]) * int(r["width"])
        if name in done or nh > a.max_hidden or nh < a.min_hidden: continue
        jobs.append((name, r["path"], r["function"], a.gpu))
    print(f"{len(jobs)} specimens ({len(done)} done)", flush=True)
    ep, yp = os.path.join(R, "explanations.csv"), os.path.join(R, "symmetry.csv")
    if a.gpu:
        it = map(job, jobs)
    else:
        pool = Pool(a.procs); it = pool.imap_unordered(job, jobs)
    for i, (rows, status, sym) in enumerate(it):
        _append(ep, FIELDS, rows); _append(sp, SFIELDS, [status]); _append(yp, YFIELDS, [sym])
        if i % 50 == 0: print(f"{i}/{len(jobs)} {status}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
