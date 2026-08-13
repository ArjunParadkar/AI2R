"""Phase 2 figures SCALE-1 (BiasTop vs output trajectory, small vs large) and
SCALE-2 (intervention window vs model size). Reads phase2 results JSON."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def norm01(a):
    a = np.array([x if x is not None else np.nan for x in a], float)
    mx = np.nanmax(a)
    return a / mx if mx and mx > 0 else a


def panel(ax, m, title):
    layers = m["topo_layers"]
    # use the strongest-rising layer for the representational trace
    best_l = max(layers, key=lambda l: (np.nanmax(m["biastop"][str(l)]) -
                                        m["biastop"][str(l)][0]))
    epochs = list(range(len(m["output_bias"])))
    ax.plot(epochs, norm01(m["biastop"][str(best_l)]), color="tab:blue", lw=2,
            label=f"BiasTop (L{best_l})")
    ax.plot(epochs, norm01(m["output_bias"]), color="red", ls="--", lw=2, label="output bias")
    tb, to = m.get("t_bias"), m.get("t_output")
    if tb is not None: ax.axvline(tb, color="tab:blue", ls=":", lw=1.5)
    if to is not None: ax.axvline(to, color="red", ls=":", lw=1.5)
    if tb is not None and to is not None:
        ax.axvspan(tb, to, color="gold", alpha=0.25)
        ax.text((tb+to)/2, 0.05, f"window={to-tb}ep", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_title(f"{title}\n({m['params_M']:.0f}M params)"); ax.set_xlabel("epoch")
    ax.set_ylabel("normalized"); ax.legend(fontsize=8); ax.grid(alpha=0.25)


def fig_scale1(res, outpath):
    names = list(res["models"].keys())
    small = res["models"][names[0]]; large = res["models"][names[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    panel(axes[0], small, "Smaller (from scratch)")
    panel(axes[1], large, "Larger (from scratch)")
    fig.suptitle("Output bias crystallizes, but representational BiasTop does NOT reliably "
                 "precede it\n(no measurable precedence window at any scale — negative result)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(outpath, dpi=300); plt.close(fig)


def fig_scale2(res, outpath, extra_points=None):
    pts = []
    for n, m in res["models"].items():
        if m.get("window") is not None:
            pts.append((m["params_M"], m["window"], n))
    if extra_points:
        pts.extend(extra_points)
    pts.sort()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if pts:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color="darkgreen", lw=2, ms=9)
        for x, y, n in pts:
            ax.annotate(f"{n}\n{x:.0f}M → {y}ep", (x, y), textcoords="offset points",
                        xytext=(8, 8), fontsize=9)
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No measurable precedence window at any scale\n"
                "(t_bias undefined — representational BiasTop never\ncrosses its 2σ "
                "baseline before output bias).\nNegative result: precedence is not a "
                "robust phenomenon.",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Model size (params)")
    ax.set_ylabel("Intervention window (epochs, t_output − t_bias)")
    ax.set_title("Precedence window vs model scale")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(outpath, dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--tinygpt_window", type=float, default=None,
                    help="optional existing TinyGPT (11.4M) window datapoint")
    args = ap.parse_args()
    res = json.load(open(args.res))
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    extra = [(11.4, args.tinygpt_window, "TinyGPT")] if args.tinygpt_window is not None else None
    fig_scale1(res, figdir / "SCALE-1_trajectories.png")
    fig_scale2(res, figdir / "SCALE-2_window_vs_size.png", extra_points=extra)
    print(f"[figures] SCALE-1, SCALE-2 -> {figdir}")
    print("windows:", {n: res["models"][n].get("window") for n in res["models"]})


if __name__ == "__main__":
    main()
