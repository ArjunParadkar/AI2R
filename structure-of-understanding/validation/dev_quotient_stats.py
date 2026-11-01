"""Development diagnostic (amendment 3, dev seeds only): per-candidate quotient statistics
(per-behaviour swap effect A_b, random-realiser null effects, verdict, accepted or not) for one
Phase 1 cell, so that criterion (i) of the quotient test can be redesigned on development data.

  python -m validation.dev_quotient_stats --spec V4 0 --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from validation import run_phase1 as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True, help="cell spec, e.g. V1 0 onehot | NULL 0 untrained V4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = [a.spec[0], int(a.spec[1])] + [{"True": True, "False": False}.get(x, x) for x in a.spec[2:]]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = out / ("qs_" + "_".join(str(x) for x in spec) + ".json")
    if f.exists():
        return
    caps = []
    orig = R.run_discovery

    def rd(*x, **k):
        G, probe, rec, lab, n = orig(*x, **k)
        acc = {}
        for c in G.concepts():
            nd = G.nodes[c]
            for cand in [nd.meta.get("candidate")] + list(nd.meta.get("merged", [])):
                acc[cand] = dict(node=c, label=lab.get(c))
        caps.append(dict(n_behaviours=len(probe.sys.behaviours), behaviours=list(probe.sys.behaviours),
                         candidates={k_: dict(v, accepted=acc.get(k_)) for k_, v in rec["candidates"].items()
                                     if "A_b" in v}))
        return G, probe, rec, lab, n
    R.run_discovery = rd
    t0 = time.time()
    res = R.run_cell(tuple(spec))
    f.write_text(json.dumps(R._jsonable(dict(spec=spec, captures=caps, wall_s=time.time() - t0,
                                             summary={k: v for k, v in res.items() if isinstance(v, (int, float, str))})),
                            default=str))
    print(f"[qs] {f.name} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
