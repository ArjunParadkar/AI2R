"""Development diagnostic (amendment 3, dev seeds only): why does the untrained-copy V4 null
accept g0-g2 under amendment 2? Runs one null cell with the coarsening rule on or off and dumps,
for every accepted node, the candidate it came from and its quotient statistics.

  python -m validation.dev_null_diag --seed 0 --kind untrained --family V4 --coarsen 0 --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import whitebox_ee.discover as D
from validation import planted as PL
from validation import run_phase1 as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kind", default="untrained")
    ap.add_argument("--family", default="V4")
    ap.add_argument("--coarsen", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"diag_{a.family}_{a.kind}_s{a.seed}_c{a.coarsen}.json"
    if f.exists():
        return
    orig = D.discover

    def disc(*x, **k):
        k["coarsening_rule"] = bool(a.coarsen)
        return orig(*x, **k)
    R.discover = disc
    captured = {}
    orig_rd = R.run_discovery

    def rd(*x, **k):
        G, probe, rec, lab, n = orig_rd(*x, **k)
        captured.update(G=G, rec=rec, lab=lab, n=n)
        return G, probe, rec, lab, n
    R.run_discovery = rd
    t0 = time.time()
    r = R.cell_null(a.seed, a.kind, a.family)
    G, rec, lab = captured["G"], captured["rec"], captured["lab"]
    acc = []
    for c in G.concepts():
        nd = G.nodes[c]
        cand = nd.meta.get("candidate")
        cr = rec["candidates"].get(cand, {})
        acc.append(dict(node=c, label=lab.get(c), candidate=cand, merged=nd.meta.get("merged", []),
                        site=nd.primary.site, k=int(nd.primary.U.shape[1]), source=cr.get("source"),
                        delta_Q=cr.get("delta_Q"), A=cr.get("A"), eta=cr.get("eta"), kappa=cr.get("kappa"),
                        H=cr.get("H"), parents=G.parents.get(c), cert=G.certificates.get(c)))
    verdicts = {}
    for k_, v in rec["candidates"].items():
        verdicts[v.get("verdict", "?").split(" of ")[0]] = verdicts.get(v.get("verdict", "?").split(" of ")[0], 0) + 1
    res = dict(seed=a.seed, kind=a.kind, family=a.family, coarsen=a.coarsen, n_candidates=r["n_candidates"],
               n_accepted=r["n_accepted"], fp_rate=r["fp_rate"], accepted=acc, verdict_counts=verdicts,
               coarsened={k_: v["verdict"] for k_, v in rec["candidates"].items()
                          if str(v.get("verdict", "")).startswith("coarsening")},
               wall_s=time.time() - t0)
    f.write_text(json.dumps(R._jsonable(res), indent=1, default=str))
    print(f"[diag] {f.name} accepted {r['n_accepted']}/{r['n_candidates']} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
