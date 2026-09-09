"""F4b — approximate stabilisers within a network via the gauge residual (SPEC_F4_reformulated §F4b).
Writes results/F4b_eps.csv, F4b_summary.csv, F4b_verdicts.json, F4b_kill.json."""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flib.models import MLP
from flib.functions import truth_table
from flib.crossseed import effective_subimpl, residual, size_signature
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EPS = (1e-6, 1e-3, 1e-2, 0.05, 0.1, 0.2)


def classes(subs, eps_list):
    N = len(subs); by_sig = {}
    for i, s in enumerate(subs):
        if s is not None: by_sig.setdefault(size_signature(s), []).append(i)
    rs = {}
    for ids in by_sig.values():
        for i, j in itertools.combinations(ids, 2): rs[(i, j)] = residual(subs[i], subs[j])[0]
    out = {}
    for eps in eps_list:
        parent = list(range(N))
        def find(i):
            while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for (i, j), r in rs.items():
            if r < eps:
                a, b = find(i), find(j)
                if a != b: parent[a] = b
        out[eps] = len({find(i) for i in range(N)})
    return out, (min(rs.values()) if rs else np.nan)


def job(args):
    spec, recs, fn = args
    m = MLP.load(os.path.join(R, "networks", f"{spec}.npz")); X, y = truth_table(fn); rows = []
    for crit, mids in recs.items():
        subs = [effective_subimpl(m, X, int(mid), crit) for mid in mids]
        K, rmin = classes(subs, EPS)
        for eps in EPS: rows.append(dict(specimen=spec, criterion=crit, eps=eps, N=len(mids), K_within=K[eps], ratio=K[eps] / len(mids), r_min=rmin))
    return rows


def kill():
    from flib.planted import planted_k
    out = {}
    m, blocks = planted_k(1); off = np.cumsum([0] + list(m.widths)); X, y = truth_table("XOR3"); planted = []
    for j in range(len(blocks)):
        mid = 0
        for l, (lo, hi) in enumerate(blocks[j]):
            for u in range(lo, hi): mid |= 1 << (off[l] + u)
        planted.append(mid)
    r, _ = residual(effective_subimpl(m, X, planted[0], "zero"), effective_subimpl(m, X, planted[1], "zero"))
    out["planted_k1_copies"] = dict(r=r, pass_=bool(r < 1e-9))
    spec = "XOR3_d2_w6_s0"; t0 = MLP.load(os.path.join(R, "networks", f"{spec}.npz")); t = t0.copy()
    from flib.enumerate import enumerate_valid
    from flib.minimal import minimal_ids_fast
    m0 = minimal_ids_fast(enumerate_valid(t0, X, y, "zero"))
    # duplicate the first unit of the original minimal circuit into a spare (non-circuit) slot, split its outgoing weight
    from flib.enumerate import circuit_layers
    units0 = circuit_layers(int(m0[0]), t0.widths)[0]; spare = [u for u in range(t0.widths[0]) if u not in units0][0]; src = int(units0[0])
    t.Ws[0][:, spare] = t.Ws[0][:, src]; t.bs[0][spare] = t.bs[0][src]; t.Ws[1][src, :] = t.Ws[1][src, :] / 2; t.Ws[1][spare, :] = t.Ws[1][src, :]
    ok = bool(np.array_equal(t.predict(X), y)); mins = minimal_ids_fast(enumerate_valid(t, X, y, "zero"))
    both = int(m0[0]) | (1 << spare)                     # the original circuit with both copies (valid; not minimal if a half-copy suffices)
    from flib.enumerate import ablated_forward, mask_from_id
    lg, _ = ablated_forward(t, X, mask_from_id(both, sum(t.widths))[None], "zero", [a.mean(0) for a in t.hidden_acts(X)])
    valid_both = bool(((lg[0] > 0).astype(np.int8) == y).all())
    r_both = residual(effective_subimpl(t0, X, int(m0[0]), "zero"), effective_subimpl(t, X, both, "zero"))[0]
    rs = [residual(effective_subimpl(t0, X, int(m0[0]), "zero"), effective_subimpl(t, X, int(mid), "zero"))[0] for mid in mins]
    out["duplicated_unit_net"] = dict(still_correct=ok, N_orig=int(len(m0)), N_dup_minimal=int(len(mins)), both_copies_circuit_valid=valid_both,
                                      r_both_copies_vs_original=float(r_both), r_min_minimal_vs_original=float(np.nanmin(rs)) if rs else None,
                                      note="minimal circuits keep one half-weight copy when it suffices (a real weight difference, r>0); the both-copies circuit must collapse to the original",
                                      pass_=bool(valid_both and r_both < 1e-9))
    json.dump(out, open(os.path.join(R, "F4b_kill.json"), "w"), indent=1); print(json.dumps(out, indent=1))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--kill": return kill()
    E = pd.read_csv(os.path.join(R, "explanations.csv"), dtype={"key6": str, "key4": str}, usecols=["specimen", "criterion", "circuit_id"])
    jobs = [(spec, {c: g2.circuit_id.tolist() for c, g2 in g.groupby("criterion")}, spec.split("_d")[0]) for spec, g in E.groupby("specimen")]
    rows = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=8)):
            rows += r
            if i % 300 == 0: print(i, flush=True)
    D = pd.DataFrame(rows); D.to_csv(os.path.join(R, "F4b_eps.csv"), index=False)
    S = D.groupby(["criterion", "eps"]).agg(ratio_median=("ratio", "median"), ratio_mean=("ratio", "mean"), frac_specimens_collapsed=("ratio", lambda s: (s < 1).mean()), n=("specimen", "nunique")).reset_index()
    S.to_csv(os.path.join(R, "F4b_summary.csv"), index=False); print(S.to_string())
    v = {c: dict(median_ratio_eps_1e2=float(S[(S.criterion == c) & (S.eps == 1e-2)].ratio_median.iloc[0]), pass_=bool(S[(S.criterion == c) & (S.eps == 1e-2)].ratio_median.iloc[0] >= 0.95)) for c in ("zero", "mean")}
    json.dump(v, open(os.path.join(R, "F4b_verdicts.json"), "w"), indent=1); print(v)


if __name__ == "__main__":
    main()
