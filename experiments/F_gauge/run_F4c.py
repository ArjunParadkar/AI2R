"""F4c — interpretation-level gauge: do Méloux threshold interpretations collapse under per-unit negation?
(SPEC_F4_reformulated §F4c). Writes results/F4c_negation.csv, F4c_kill.json."""
import csv, os, sys, json, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flib.models import MLP
from flib.functions import truth_table
from flib.enumerate import circuit_layers, separations, _ablated_acts
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CAP = 20000


def negation_canonical(B):
    """Complement every unit column so that its first entry is 0 (per-unit negation gauge)."""
    B = np.asarray(B, dtype=np.int8)
    return (B ^ B[:1, :]).tobytes()


def enumerate_bits(model, X, mid, scheme, cap=CAP):
    """All Méloux-consistent interpretations of circuit ``mid`` as bit matrices (N, n_units); None if > cap."""
    acts, logits = _ablated_acts(model, X, mid, scheme); layers = circuit_layers(mid, model.widths); out_bits = (logits > 0).astype(np.int8)
    found = []

    def rec(l, prev_bits, cols):
        if len(found) > cap: return
        if l == len(layers):
            table = {}
            for r in range(len(X)):
                k = prev_bits[r].tobytes()
                if table.setdefault(k, int(out_bits[r])) != out_bits[r]: return
            found.append(np.concatenate(cols, axis=1) if cols else np.zeros((len(X), 0), np.int8)); return
        opts = [separations(prev_bits, acts[l][:, u]) for u in layers[l]]
        if any(len(o) == 0 for o in opts): return
        for combo in itertools.product(*opts):
            bits = np.stack([c[1] for c in combo], axis=1).astype(np.int8) if combo else np.zeros((len(X), 0), np.int8)
            rec(l + 1, bits, cols + [bits])
            if len(found) > cap: return
    rec(0, X.astype(np.int8), [])
    return None if len(found) > cap else found


def job(args):
    spec, crit, mid, fn = args
    m = MLP.load(os.path.join(R, "networks", f"{spec}.npz")); X, y = truth_table(fn)
    B = enumerate_bits(m, X, int(mid), crit)
    if B is None: return dict(specimen=spec, criterion=crit, circuit_id=mid, n_interp=-1, n_classes=-1, ratio=np.nan)
    classes = {negation_canonical(b) for b in B}
    return dict(specimen=spec, criterion=crit, circuit_id=mid, n_interp=len(B), n_classes=len(classes), ratio=len(classes) / len(B) if B else np.nan)


def kill():
    B1 = np.array([[0, 1], [1, 0], [0, 0], [1, 1]], np.int8); B2 = B1.copy(); B2[:, 0] ^= 1      # unit 0 negated
    same = negation_canonical(B1) == negation_canonical(B2)
    B3 = B1.copy(); B3[0, 0] ^= 1                                                                 # a genuinely different bit pattern
    diff = negation_canonical(B1) != negation_canonical(B3)
    out = dict(complement_pair_one_class=bool(same), different_pattern_two_classes=bool(diff), pass_=bool(same and diff))
    json.dump(out, open(os.path.join(R, "F4c_kill.json"), "w"), indent=1); print(out)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--kill": return kill()
    E = pd.read_csv(os.path.join(R, "explanations.csv"), dtype={"key6": str, "key4": str}, usecols=["specimen", "criterion", "circuit_id", "n_meloux"])
    E = E[(E.n_meloux > 1) & (E.n_meloux <= CAP)]
    E = E.sample(min(3000, len(E)), random_state=0)
    jobs = [(r.specimen, r.criterion, int(r.circuit_id), r.specimen.split("_d")[0]) for r in E.itertuples()]
    # resumable (AMENDMENTS #11 addendum): every finished circuit is appended to F4c_negation_partial.csv at once;
    # on restart the deterministic sample (random_state=0) is re-drawn and finished circuits are skipped.
    part = os.path.join(R, "F4c_negation_partial.csv"); rows = []
    if os.path.exists(part):
        rows = pd.read_csv(part).to_dict("records")
        done = {(r["specimen"], r["criterion"], int(r["circuit_id"])) for r in rows}
        jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
        print(f"resuming: {len(rows)} done, {len(jobs)} left", flush=True)
    with Pool(4) as pool, open(part, "a", newline="") as f:
        w = None
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=8)):
            rows.append(r)
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r))
                if f.tell() == 0: w.writeheader()
            w.writerow(r); f.flush()
            if i % 200 == 0: print(i, r, flush=True)
    D = pd.DataFrame(rows); D.to_csv(os.path.join(R, "F4c_negation.csv"), index=False)
    ok = D[D.n_interp > 0]
    v = dict(n_circuits=int(len(ok)), n_capped=int((D.n_interp < 0).sum()), ratio_median=float(ok.ratio.median()), ratio_mean=float(ok.ratio.mean()),
             frac_circuits_with_collapse=float((ok.ratio < 1).mean()), n_interp_matches_n_meloux=float((ok.n_interp.values == E.set_index(["specimen", "criterion", "circuit_id"]).loc[list(zip(ok.specimen, ok.criterion, ok.circuit_id))].n_meloux.values).mean()))
    json.dump(v, open(os.path.join(R, "F4c_verdicts.json"), "w"), indent=1); print(v)


if __name__ == "__main__":
    main()
