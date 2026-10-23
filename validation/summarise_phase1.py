"""Summarise Phase 1 cell JSONs into a table (one line per cell)."""
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
for f in sorted(d.glob("*.json")):
    r = json.loads(f.read_text())
    if "error" in r:
        print(f.name, "ERROR", r["error"][:100]); continue
    c = r["cell"]
    if c in ("V1", "SENS"):
        print(f.name, "nodeP %.2f nodeR %.2f edgeF1 %.2f decoy %s wrong_hits %d rho %.2f gMAE %.3f oracleF1 %.2f rt %.0fs" % (
            r["node_precision"], r["node_recall"], r["edge_f1"], r["decoy_verdicts"][:1], len(r["wrong_hits"]),
            r["spearman_vs_oracle"] if r["spearman_vs_oracle"] is not None else float("nan"),
            r.get("generality_mae_vs_oracle") if r.get("generality_mae_vs_oracle") is not None else float("nan"),
            r["oracle_edge_f1_vs_spec"], r["runtime_s"]))
    elif c == "NULL":
        print(f.name, "accepted %d / %d  fp %.3f %s" % (r["n_accepted"], r["n_candidates"], r["fp_rate"], r["accepted_labels"]))
    elif c == "V2":
        print(f.name, "AUROC(-mu) %s AUROC(-lam) %s memo@tau %.2f grounded_rule %.2f" % (
            [round(x, 3) for x in r["auroc_neg_mu"]], [round(x, 3) for x in r["auroc_neg_lambda"]], r["memorized_at_tau"], r["grounded_rule"]))
    elif c == "V3":
        print(f.name, "withheld move %.2f predc %.2f | seen move %.2f predc %.2f | nodes %s" % (
            r["withheld"]["move_acc"], r["withheld"]["pred_correct_acc"], r["seen"]["move_acc"], r["seen"]["pred_correct_acc"], r["node_labels"]))
    elif c == "V4":
        print(f.name, "AUROC %.3f scaffold_recall %.2f content_recall %.2f principles %s" % (
            r["auroc_scaffold_vs_content"], r["scaffold_recall"], r["content_recall"], r["principles_025"]))
    elif c == "V5":
        print(f.name, "valid", r["valid"], "status", {k: v for k, v in r["status"].items() if v != "relabelling"})
    elif c == "FORMATION":
        print(f.name, "events", r["events"], "Lambda", [round(x, 2) for x in r["Lambda"]], "sum principle from", r["sum_principle_from"])
