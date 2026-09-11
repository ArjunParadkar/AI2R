"""F2 + F3 + F4(within) — enumerate explanations, hidden symmetries, canonical orbit keys.
Writes results/explanations.csv (one row per minimal circuit x criterion), results/symmetry_f2.csv (pre-Amendment-7 accounting; results/symmetry.csv is owned by grass_wonder.py),
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


MELOUX_CAP = 60   # AMENDMENTS #4


def _count_capped(model, X, mid, scheme, cap=MELOUX_CAP):
    import signal
    def _h(sig, frm): raise TimeoutError
    old = signal.signal(signal.SIGALRM, _h); signal.alarm(cap)
    try:
        return int(interpretation_count(model, X, mid, scheme))
    except TimeoutError:
        return -1
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)


def circuit_row(model, X, y, spec, scheme, mid, meloux=True):
    mid = int(mid)
    ok, gates, out_table = handbook_interpretation(model, X, mid, scheme)
    nm = _count_capped(model, X, mid, scheme) if meloux else -1
    keys = {eps: canonical_form(model, mid, eps)[0].hex() for eps in EPS}
    return dict(specimen=spec, criterion=scheme, circuit_id=mid, n_units=bin(mid).count("1"),
                interpretable=int(ok), n_meloux=nm, key6=keys[1e-6], key4=keys[1e-4],
                fkey=hash(functional_key(model, X, mid, scheme)),
                algorithm=algorithm_string(gates, out_table, circuit_layers(mid, model.widths)) if ok else "")


def _circuit_job(args):
    path, fn, spec, scheme, mid, meloux = args
    import torch; torch.set_num_threads(1)
    model = MLP.load(os.path.join(R, path)); X, y = truth_table(fn)
    return circuit_row(model, X, y, spec, scheme, mid, meloux)


def analyse(model, X, y, spec, use_gpu=False, meloux=True, pool=None, path=None, fn=None):
    """Returns (explanation rows, status row). With ``pool`` (GPU mode) the per-circuit analysis is
    farmed out to worker processes."""
    rows, status = [], {"specimen": spec, "n_hidden": sum(model.widths)}
    t0 = time.time()
    for scheme in ("zero", "mean"):
        valid = _valid(model, X, y, scheme, use_gpu)
        if valid is None:
            status[f"enumerated_{scheme}"] = 0; continue
        status[f"enumerated_{scheme}"] = 1
        if len(valid) > VALID_BUDGET:      # AMENDMENTS #12: np.unique/sort over >~2e8 int64 ids needs > 20 GB; count, record, skip minimality
            status[f"n_valid_{scheme}"] = int(len(valid)); status[f"n_minimal_{scheme}"] = -1; status[f"enumerated_{scheme}"] = 2
            _append(os.path.join(OUT_DIR, "enumeration_excluded.csv"), ["specimen", "reason", "excluded_at"],
                    [dict(specimen=f"{spec}:{scheme}", reason=f"valid set {len(valid)} > VALID_BUDGET {VALID_BUDGET} (minimality skipped; count kept)", excluded_at=time.strftime("%F %T"))])
            del valid; continue
        mins = minimal_ids_fast(valid)
        status[f"n_valid_{scheme}"] = int(len(valid)); status[f"n_minimal_{scheme}"] = int(len(mins))
        if pool is not None and len(mins) > 4:
            rows += pool.map(_circuit_job, [(path, fn, spec, scheme, int(mid), meloux) for mid in mins], chunksize=4)
        else:
            rows += [circuit_row(model, X, y, spec, scheme, mid, meloux) for mid in mins]
    status["wall"] = round(time.time() - t0, 1)
    return rows, status


_POOL = None
VALID_BUDGET = 150_000_000   # AMENDMENTS #12 (2026-09-11): largest valid set that fits the 20 GB unit cap; --valid-budget overrides
OUT_DIR = R


def job(args):
    spec, path, fn, use_gpu = args
    import torch; torch.set_num_threads(1)
    model = MLP.load(os.path.join(R, path)); X, y = truth_table(fn)
    rows, status = analyse(model, X, y, spec, use_gpu, pool=_POOL, path=path, fn=fn)
    # F3
    h = S.hidden(model, inputs=X, seed=0, keep_directions=False)
    sym = dict(specimen=spec, n_params=h.n_params, rank=h.rank, fiber_dim=h.corank, known_dim=h.known_dim,
               excess=h.excess, dead=h.identified.get("dead", ""), duplicate=h.identified.get("duplicate", ""),
               constant=h.identified.get("constant", ""), unexplained=h.unexplained,   # keys changed in Amendment 7; F3 owns the real accounting G_incomplete=int(h.G_incomplete),
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
    from flib.planted import xor3_single, planted_k, with_bias_unit
    from flib.enumerate import mask_from_id
    X, y = truth_table("XOR3"); out = {}
    m = xor3_single()
    rows, st = analyse(m, X, y, "planted_xor_single", use_gpu=False)
    out["F2_kill"] = {s: dict(n_minimal=st[f"n_minimal_{s}"], circuits=[r["circuit_id"] for r in rows if r["criterion"] == s],
                             interpretable=[r["interpretable"] for r in rows if r["criterion"] == s],
                             full_set_id=(1 << sum(m.widths)) - 1) for s in ("zero", "mean")}
    print("F2 kill", out["F2_kill"], flush=True)
    for k in (1, 2, 3):
        m, blocks = planted_k(k); nc = len(blocks)
        n_hidden = sum(m.widths); off = np.cumsum([0] + list(m.widths))
        planted = []
        for j in range(nc):
            mid = 0
            for l, (lo, hi) in enumerate(blocks[j]):
                for u in range(lo, hi):
                    mid |= 1 << (off[l] + u)
            planted.append(mid)
        res = {"n_hidden": n_hidden, "expected_K": k, "expected_N": 2 * k}
        means = [a.mean(0) for a in m.hidden_acts(X)]
        from flib.enumerate import ablated_forward
        for s in ("zero", "mean"):
            r = {}
            # validity of each planted circuit
            masks = np.stack([mask_from_id(mid, n_hidden) for mid in planted])
            logits, _ = ablated_forward(m, X, masks, s, means)
            r["planted_valid"] = [bool(((logits[i] > 0).astype(np.int8) == y).all()) for i in range(nc)]
            # minimality: enumerate all subsets of each planted circuit
            r["planted_minimal"] = []
            for mid in planted:
                units = np.flatnonzero(mask_from_id(mid, n_hidden)); sub_valid = 0
                for sub in range(1, (1 << len(units)) - 1):
                    sm = np.zeros(n_hidden, bool); sm[units[[i for i in range(len(units)) if (sub >> i) & 1]]] = True
                    lg, _ = ablated_forward(m, X, sm[None], s, means)
                    if ((lg[0] > 0).astype(np.int8) == y).all(): sub_valid += 1
                r["planted_minimal"].append(sub_valid == 0)
            keys = [canonical_form(m, mid, 1e-6)[0].hex() for mid in planted]
            interp = [handbook_interpretation(m, X, mid, s)[0] for mid in planted]
            r["planted_interpretable"] = interp
            r["K_planted"] = len(set(keys)); r["N_planted"] = nc
            if n_hidden <= 28:
                t0 = time.time()
                valid = _valid(m, X, y, s, use_gpu=n_hidden > 20)
                if valid is not None:
                    mins = minimal_ids_fast(valid)
                    ok = [handbook_interpretation(m, X, int(x), s)[0] for x in mins]
                    ks = [canonical_form(m, int(x), 1e-6)[0].hex() for x in mins]
                    r["exhaustive"] = dict(n_valid=int(len(valid)), n_minimal=int(len(mins)), N=int(sum(ok)),
                                           K=len(set(kk for kk, o in zip(ks, ok) if o)),
                                           planted_among_minimal=[int(p) in set(int(x) for x in mins) for p in planted],
                                           wall=round(time.time() - t0, 1))
            res[s] = r
            print(k, s, r, flush=True)
        out[f"F4_kill_k{k}"] = res
    # F4-kill-b: gauge consistency of the orbit keys on trained specimens
    from flib.gauge import random_gauge
    spec = [r for r in csv.DictReader(open(os.path.join(R, "specimens.csv"))) if int(r["converged"]) and int(r["depth"]) * int(r["width"]) <= 16]
    rng = np.random.default_rng(0); sel = rng.choice(len(spec), 100, replace=False)
    kb = {"n": 0, "identical_zero": 0, "identical_mean": 0, "failures": []}
    for i in sel:
        r = spec[i]; m = MLP.load(os.path.join(R, r["path"])); Xs, ys = truth_table(r["function"])
        mg, _, _ = random_gauge(m, rng, log_scale_sd=1.0)
        for s in ("zero", "mean"):
            k0 = sorted(canonical_form(m, int(x), 1e-6)[0].hex() for x in minimal_ids_fast(enumerate_valid(m, Xs, ys, s)))
            k1 = sorted(canonical_form(mg, int(x), 1e-6)[0].hex() for x in minimal_ids_fast(enumerate_valid(mg, Xs, ys, s)))
            same = k0 == k1
            kb[f"identical_{s}"] += int(same)
            if not same: kb["failures"].append((r["path"], s, len(k0), len(k1), len(set(k0) & set(k1))))
        kb["n"] += 1
    out["F4_kill_b"] = kb
    print("F4-kill-b", kb, flush=True)
    with open(os.path.join(R, "kill_tests.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else int(o))
    print(json.dumps(out, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else int(o)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12); ap.add_argument("--max-hidden", type=int, default=20)
    ap.add_argument("--min-hidden", type=int, default=0); ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--kill", action="store_true")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("I", "N"), help="HPCC: take every N-th pending specimen starting at I")
    ap.add_argument("--out", default=None, help="HPCC: write explanations/status/symmetry CSVs under this directory instead of results/")
    ap.add_argument("--valid-budget", type=int, default=VALID_BUDGET, help="AMENDMENTS #12: skip minimality (count kept) when a scheme's valid set exceeds this")
    a = ap.parse_args()
    global VALID_BUDGET, OUT_DIR; VALID_BUDGET = a.valid_budget
    if a.kill:
        return kill_tests()
    spec = list(csv.DictReader(open(os.path.join(R, "specimens.csv"))))
    done = set()
    OUT = a.out or R; os.makedirs(OUT, exist_ok=True); OUT_DIR = OUT
    sp = os.path.join(OUT, "enumeration_status.csv")
    if a.out and os.path.exists(os.path.join(R, "enumeration_status.csv")):      # specimens already enumerated locally count as done
        done |= {r["specimen"] for r in csv.DictReader(open(os.path.join(R, "enumeration_status.csv")))}
    if os.path.exists(sp):
        done = {r["specimen"] for r in csv.DictReader(open(sp))}
    jobs = []
    for r in spec:
        if not int(r["converged"]): continue
        name = f'{r["function"]}_d{r["depth"]}_w{r["width"]}_s{r["seed"]}'
        nh = int(r["depth"]) * int(r["width"])
        if name in done or nh > a.max_hidden or nh < a.min_hidden: continue
        jobs.append((name, r["path"], r["function"], a.gpu))
    jobs = jobs[a.shard[0]::a.shard[1]]
    print(f"{len(jobs)} specimens ({len(done)} done; shard {a.shard[0]}/{a.shard[1]})", flush=True)
    ep, yp = os.path.join(OUT, "explanations.csv"), os.path.join(OUT, "symmetry_f2.csv")  # AMENDMENTS #9
    if a.gpu:
        global _POOL
        _POOL = Pool(a.procs)
        it = map(job, jobs)
    else:
        pool = Pool(a.procs); it = pool.imap_unordered(job, jobs)
    for i, (rows, status, sym) in enumerate(it):
        _append(ep, FIELDS, rows); _append(sp, SFIELDS, [status]); _append(yp, YFIELDS, [sym])
        if i % 50 == 0: print(f"{i}/{len(jobs)} {status}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
