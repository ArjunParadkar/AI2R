#!/usr/bin/env python3
"""O5 Symboli Kris S — the grid (FINAL_SPEC §5, §8.2 O5): models × P0–P7 with the degree of opening 1 − γ_cov, one row per specimen
and one summary row per T1 condition, each cell linked to its evidence file.

  python symboli_kris_s.py            # -> results/grid.md, results/grid.csv, results/grid.json

Sources (results/open/): meisho_doto_<s>.json (P0, P1, P2, P6, P7, R1–R8, γ_cov), nishino_flower_<s>.json (P3),
fine_motion_<s>.json (P4, P5). Each horse's JSON carries a top-level "cells" dict keyed by cell name with status/value/criterion/
provenance. A cell no horse reached is "not_reached". HPCC tiers (T3, T4) are listed as pending rows.
"""
from __future__ import annotations
import csv, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results"); OPEN = os.path.join(RES, "open")
CELLS = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]
SOURCES = {"meisho_doto": ["P0", "P1", "P2", "P6", "P7"], "fine_motion": ["P4", "P5"]}


def p3_cell(spec):
    """P3 from O1.3's detections CSV (nishino_flower_detections.csv): a planted specimen passes only if some detector's onset is within 5 %
    AND that detector has zero false detections over the nulls; non-planted specimens carry no planted bias (they are the nulls)."""
    f = os.path.join(RES, "nishino_flower_detections.csv")
    if not os.path.exists(f): return None
    rows = list(csv.DictReader(open(f))); r = next((x for x in rows if x["specimen"] == spec), None)
    if r is None: return None
    if r.get("condition", "") != "planted":
        return dict(status="not_reached", provenance="no planted bias (null specimen for O1.3; false-detection counts in nishino_flower_detections.csv)")
    dets = ["zigzag_raw", "zigzag_diff", "eigen_gap_raw", "eigen_gap_diff", "probe_res8_raw", "probe_res8_diff", "probe_res8_v2_diff", "probe_res8_v3_diff"]
    fd = {d: sum(1 for x in rows if x.get(f"{d}_false_detection", "") in ("1", "True", "true")) for d in dets}
    best = None
    for d in dets:
        for gt in ("acc", "logit", "logitw"):
            p = r.get(f"{d}_pass_{gt}", ""); a = r.get(f"{d}_absdelta_{gt}", "")
            if p in ("1", "True", "true") and fd[d] == 0: return dict(status="certified", value=f"{d} |Δ| {a} vs {gt} GT, 0 false detections")
            try:
                a = float(a)
                if best is None or a < best[0]: best = (a, d, gt, fd[d])
            except ValueError: pass
    return dict(status="failed", value=None if best is None else f"best |Δ| {best[0]:.2f} ({best[1]} vs {best[2]} GT, {best[3]} false det.)")


def load(horse, spec):
    f = os.path.join(OPEN, f"{horse}_{spec}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def specimens():
    rows = list(csv.DictReader(open(os.path.join(RES, "specimens.csv"))))
    return [r for r in rows if r["included"] == "1"]


def main():
    grid = []
    for r in specimens():
        s = r["name"]; row = dict(specimen=s, kind=r["kind"], condition=r["condition"], seed=r["seed"]); links = {}
        gamma = None
        for horse, owned in SOURCES.items():
            d = load(horse, s)
            for c in owned:
                if d is None: row[c] = "not_reached"; links[c] = f"{horse} not run"; continue
                cc = d.get("cells", {}).get(c)
                row[c] = cc["status"] if cc else "not_reached"; links[c] = f"open/{horse}_{s}.json#{c}"
        c3 = p3_cell(s); row["P3"] = c3["status"] if c3 else "not_reached"; links["P3"] = "nishino_flower_detections.csv"
        for horse in ("meisho_doto",):
            d = load(horse, s)
            if horse == "meisho_doto" and d is not None:
                gamma = d.get("scorecard", {}).get("gamma_cov"); row["R"] = ",".join(f"{k}:{v['status'][:4]}" for k, v in d["cells"].items() if k.startswith("R"))
        row["gamma_cov"] = gamma; row["degree"] = None if gamma is None else round(1 - gamma, 4); row["links"] = links; grid.append(row)
    # summary rows per condition (pre-registered: cell certified on >= 4/5 clean seeds)
    summary = []
    for cond in sorted({g["condition"] for g in grid}):
        rows = [g for g in grid if g["condition"] == cond]; srow = dict(specimen=f"T1 {cond} ({len(rows)} seeds)", condition=cond)
        for c in CELLS:
            n = sum(g[c] == "certified" for g in rows); srow[c] = f"{n}/{len(rows)} certified"
        gam = [g["gamma_cov"] for g in rows if g["gamma_cov"] is not None]
        srow["gamma_cov"] = None if not gam else float(np.median(gam)); srow["degree"] = None if not gam else round(1 - float(np.median(gam)), 4); summary.append(srow)
    pending = [dict(specimen=t, condition="pending") for t in ("T2 GPT-2 small (O2.1/O2.2)", "T3 Pythia-410M/1B (O3.1, HPCC)", "T4 Llama-3-8B / Gemma-2-9B (O4.1, HPCC)")]
    json.dump(dict(rows=grid, summary=summary, pending=pending), open(os.path.join(RES, "grid.json"), "w"), indent=1, default=str)
    with open(os.path.join(RES, "grid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["specimen", "kind", "condition", "seed"] + CELLS + ["R", "gamma_cov", "degree"], extrasaction="ignore"); w.writeheader(); w.writerows(grid)
    L = ["# The grid — T1 (O5 Symboli Kris S)", "", "| model | " + " | ".join(CELLS) + " | R cells | γ_cov | 1 − γ_cov |", "|" + "---|" * (len(CELLS) + 4)]
    for g in grid:
        L.append(f"| {g['specimen']} | " + " | ".join(g[c] for c in CELLS) + f" | {g.get('R', '')} | {'—' if g['gamma_cov'] is None else f'{g['gamma_cov']:.4f}'} | {'—' if g['degree'] is None else g['degree']} |")
    L += ["", "| condition | " + " | ".join(CELLS) + " | median γ_cov | 1 − γ_cov |", "|" + "---|" * (len(CELLS) + 3)]
    for srow in summary:
        L.append(f"| {srow['specimen']} | " + " | ".join(srow[c] for c in CELLS) + f" | {'—' if srow['gamma_cov'] is None else f'{srow['gamma_cov']:.4f}'} | {'—' if srow['degree'] is None else srow['degree']} |")
    L += ["", "Pending rows: " + "; ".join(p["specimen"] for p in pending), "", "Evidence: results/open/<horse>_<specimen>.json (cell status, value, criterion, null, provenance)."]
    open(os.path.join(RES, "grid.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
