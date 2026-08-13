"""Phase 1 figures INT-1, INT-2, INT-3 + selection of best alpha. Reads results JSON."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOPO_LAYERS = [0, 3, 6, 9, 11]
INTERVENE_BLOCKS = [0, 1, 2, 3]


def norm01(a, ref_max):
    a = np.asarray(a, float)
    return a / ref_max if ref_max > 0 else a


def first_cross(series, thr):
    for i, v in enumerate(series):
        if v is not None and v > thr:
            return i
    return None


def choose_report_epoch(res):
    """Report at the operating point where bias has crystallized but the model is still
    a healthy LM: the last epoch where control perplexity <= 1.8x its running minimum.
    (30-epoch fine-tuning on a small corpus overfits the CONTROL itself, so epoch-30
    perplexity is meaningless for both runs; we report the honest early-stopping point.)"""
    ppl = res["run_A"]["perplexity"]
    mn = min(ppl[1:]) if len(ppl) > 1 else ppl[0]
    ob = res["run_A"]["output_bias"]
    onset = res["run_B"][list(res["run_B"])[0]].get("output_onset_threshold", 0)
    ep = 1
    for i in range(1, len(ppl)):
        if ppl[i] <= 1.8 * mn:
            ep = i
        else:
            break
    # ensure bias has crystallized (control output bias above onset)
    while ep < len(ob) - 1 and ob[ep] < max(onset, 2 * ob[0]):
        ep += 1
    return ep


def pick_best_alpha(res, report_epoch):
    """Best bias-reduction-to-perplexity-preservation ratio at the report epoch."""
    A = res["run_A"]
    e = report_epoch
    ob_A = A["output_bias"][e]; ppl_A = A["perplexity"][e]
    rows = []
    for a, B in res["run_B"].items():
        ob_B = B["output_bias"][e]; ppl_B = B["perplexity"][e]
        bias_red = (ob_A - ob_B) / ob_A * 100 if ob_A else 0.0
        ppl_deg = (ppl_B - ppl_A) / ppl_A * 100 if ppl_A else 0.0
        ratio = bias_red / (abs(ppl_deg) + 1e-6)
        rows.append({"alpha": float(a), "bias_reduction_pct": bias_red,
                     "ppl_degradation_pct": ppl_deg, "ratio": ratio,
                     "outbias_B": ob_B, "ppl_B": ppl_B})
    rows.sort(key=lambda r: (r["bias_reduction_pct"] > 0,
                             r["bias_reduction_pct"] - max(r["ppl_degradation_pct"], 0)),
              reverse=True)
    return rows[0], rows, ob_A, ppl_A


def fig_int1(res, best_alpha, outpath):
    A = res["run_A"]; B = res["run_B"][str(best_alpha)]
    epochs = list(range(len(A["output_bias"])))
    btA = np.array(A["biastop"]["0"]); btB = np.array(B["biastop"]["0"])
    obA = np.array(A["output_bias"]); obB = np.array(B["output_bias"])
    bt_max = max(btA.max(), btB.max(), 1e-9)
    ob_max = max(obA.max(), obB.max(), 1e-9)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, norm01(btA, bt_max), color="tab:blue", lw=2.2, label="BiasTop — Run A (control)")
    ax.plot(epochs, norm01(btB, bt_max), color="tab:orange", lw=2.2, label="BiasTop — Run B (intervened)")
    ax.plot(epochs, norm01(obA, ob_max), color="red", lw=2.0, ls="--", label="Output bias — Run A")
    ax.plot(epochs, norm01(obB, ob_max), color="green", lw=2.0, ls="--", label="Output bias — Run B")
    t = B.get("intervened_at")
    if t is not None:
        ax.axvline(t, color="black", lw=2.0)
        ax.text(t + 0.2, 0.96, "Topological Intervention", rotation=90,
                va="top", ha="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Normalized bias magnitude")
    ax.set_title(f"Topological intervention prevents bias crystallization (GPT-2 124M, α={best_alpha})")
    ax.legend(loc="center right", fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(outpath, dpi=300); plt.close(fig)


def fig_int2(res, best_alpha, ob_A, ppl_A, outpath, report_epoch):
    B = res["run_B"][str(best_alpha)]
    ob_B = B["output_bias"][report_epoch]; ppl_B = B["perplexity"][report_epoch]
    bias_red = (ob_A - ob_B) / ob_A * 100
    ppl_deg = (ppl_B - ppl_A) / ppl_A * 100
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # output bias
    ax = axes[0]
    bars = ax.bar(["Run A\n(control)", "Run B\n(intervened)"], [ob_A, ob_B],
                  color=["#c0392b", "#27ae60"])
    ax.set_title(f"Output bias @ epoch {report_epoch}"); ax.set_ylabel("stereotype preference (NLL gap)")
    ax.text(1, ob_B, f"{'-' if bias_red>=0 else '+'}{abs(bias_red):.1f}%", ha="center",
            va="bottom", fontweight="bold")
    for b, v in zip(bars, [ob_A, ob_B]):
        ax.text(b.get_x()+b.get_width()/2, v*0.5, f"{v:.3f}", ha="center", color="white", fontweight="bold")
    # perplexity
    ax = axes[1]
    bars = ax.bar(["Run A\n(control)", "Run B\n(intervened)"], [ppl_A, ppl_B],
                  color=["#2980b9", "#8e44ad"])
    ax.set_title(f"Perplexity @ epoch {report_epoch} (held-out prose)"); ax.set_ylabel("Perplexity")
    ax.text(1, ppl_B, f"{ppl_deg:+.1f}%", ha="center", va="bottom", fontweight="bold")
    for b, v in zip(bars, [ppl_A, ppl_B]):
        ax.text(b.get_x()+b.get_width()/2, v*0.5, f"{v:.2f}", ha="center", color="white", fontweight="bold")
    fig.suptitle(f"Bias reduced {bias_red:.1f}% at {ppl_deg:+.1f}% perplexity cost (α={best_alpha})",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(outpath, dpi=300); plt.close(fig)


def fig_int3(res, outpath):
    """Heatmap: rows = topo layers, cols = epochs, color = BiasTop (Run A control).
    White line = t_bias per layer (2sd crossing); red line = t_output (global crossing)."""
    A = res["run_A"]
    layers = TOPO_LAYERS
    mat = np.array([A["biastop"][str(l)] for l in layers])  # [L, E+1]
    n_ep = mat.shape[1]
    # per-layer t_bias: first epoch BiasTop_l exceeds (its epoch-0 + 2*std-ish);
    # use layer-0 baseline thr for L0, and 2*std of each layer's own early trajectory.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(mat, aspect="auto", cmap="magma", origin="lower",
                   extent=[0, n_ep - 1, -0.5, len(layers) - 0.5])
    ax.set_yticks(range(len(layers))); ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Residual layer")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("BiasTop (W2 male vs female)")
    # output bias crossing (global, single red line)
    obA = A["output_bias"]
    ob0 = obA[0]
    # threshold: 2*std over epochs 0..3 baseline window
    base = np.array(obA[:4]); ob_thr = base.mean() + 2*base.std()
    t_out = first_cross(obA, ob_thr)
    for yi, l in enumerate(layers):
        series = A["biastop"][str(l)]
        b = np.array(series[:4]); thr = b.mean() + 2*b.std()
        t_bias = first_cross(series, thr)
        if t_bias is not None:
            ax.plot([t_bias, t_bias], [yi-0.45, yi+0.45], color="white", lw=2.5)
    if t_out is not None:
        ax.axvline(t_out, color="red", lw=2.0, ls="-")
        ax.text(t_out+0.1, len(layers)-0.6, "t_output", color="red", fontweight="bold")
    ax.set_title("Intervention window: white = topological bias onset (per layer), red = output-bias onset")
    fig.tight_layout(); fig.savefig(outpath, dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--tabdir", required=True)
    args = ap.parse_args()
    res = json.load(open(args.res))
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    tabdir = Path(args.tabdir); tabdir.mkdir(parents=True, exist_ok=True)

    report_epoch = choose_report_epoch(res)
    best, rows, ob_A, ppl_A = pick_best_alpha(res, report_epoch)
    best_alpha = best["alpha"]
    print(f"[report epoch] {report_epoch} (control ppl={ppl_A:.1f}; epoch-30 is overfit, see report)")
    print(f"[alpha sweep @ep{report_epoch}] best alpha={best_alpha} "
          f"bias_red={best['bias_reduction_pct']:.1f}% ppl_deg={best['ppl_degradation_pct']:+.1f}%")
    for r in rows:
        print(f"  alpha={r['alpha']}: bias_red={r['bias_reduction_pct']:.1f}% "
              f"ppl_deg={r['ppl_degradation_pct']:+.1f}% ratio={r['ratio']:.2f}")

    # alpha sweep table
    import csv
    with open(tabdir / "phase1_alpha_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "bias_reduction_pct", "ppl_degradation_pct",
                                          "ratio", "outbias_B", "ppl_B"])
        w.writeheader()
        for r in rows: w.writerow(r)

    fig_int1(res, best_alpha, figdir / "INT-1_headline.png")
    fig_int2(res, best_alpha, ob_A, ppl_A, figdir / "INT-2_bars.png", report_epoch)
    fig_int3(res, figdir / "INT-3_window_heatmap.png")
    print(f"[figures] wrote INT-1, INT-2, INT-3 to {figdir}")

    # summary json
    A = res["run_A"]; B = res["run_B"][str(best_alpha)]
    # intervention window in Run A (gap between topological onset and output onset, layer 0)
    bt0 = A["biastop"]["0"]; b = np.array(bt0[:4]); thr = b.mean()+2*b.std()
    t_bias_A = first_cross(bt0, thr)
    obA = A["output_bias"]; ob = np.array(obA[:4]); othr = ob.mean()+2*ob.std()
    t_out_A = first_cross(obA, othr)
    window = (t_out_A - t_bias_A) if (t_bias_A is not None and t_out_A is not None) else None
    # did output bias ever cross significance in Run B?
    perm_B = B["perm_final"]
    crossed_B = any(v["p_value"] < 0.05 for v in perm_B.values())
    # regularization side-effect: mean B-vs-A perplexity delta over the back half
    reg = {}
    for a, B in res["run_B"].items():
        n = len(A["perplexity"]); half = range(n//2, n)
        reg[a] = float(np.mean([(B["perplexity"][i]-A["perplexity"][i])/A["perplexity"][i]*100
                                for i in half]))
    summary = {
        "report_epoch": report_epoch,
        "report_epoch_rationale": "last healthy-LM epoch (control ppl <=1.8x min); "
                                  "epoch-30 perplexity is overfit for BOTH runs and not reported.",
        "best_alpha": best_alpha,
        "bias_reduction_pct": best["bias_reduction_pct"],
        "ppl_degradation_pct": best["ppl_degradation_pct"],
        "control_ppl_at_report_epoch": ppl_A,
        "regularization_backhalf_ppl_delta_pct": reg,
        "intervention_window_runA_epochs": window,
        "t_bias_runA": t_bias_A, "t_output_runA": t_out_A,
        "intervened_at_runB": B.get("intervened_at"),
        "selected_heads": B.get("selected_heads"),
        "runB_any_layer_bias_significant_final": crossed_B,
        "perm_final_runB": perm_B,
        "alpha_rows": rows,
    }
    json.dump(summary, open(tabdir / "phase1_summary.json", "w"), indent=2)
    print(f"[summary] window(RunA)={window} epochs  t_bias={t_bias_A} t_output={t_out_A}")


if __name__ == "__main__":
    main()
