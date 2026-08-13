"""Phase 4 figure: the same WHITEBOX erasure removes SEPARABLE real-world bias (toxicity
identity bias) but NOT proxy-encoded bias (COMPAS) — the honest scope of the method."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def last_mean(hist, key, k=3):
    return float(np.mean([h[key] for h in hist[-k:]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compas", required=True)
    ap.add_argument("--tox", required=True)
    ap.add_argument("--figdir", required=True)
    args = ap.parse_args()
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    tox = json.load(open(args.tox)); comp = json.load(open(args.compas))

    # stabilize with last-k mean
    tA = last_mean(tox["runA"], "identity_toxic_rate"); tB = last_mean(tox["runB"], "identity_toxic_rate")
    taccA = last_mean(tox["runA"], "tox_acc"); taccB = last_mean(tox["runB"], "tox_acc")
    cA = abs(comp["final_A"]["demographic_parity_gap"]); cB = abs(comp["final_B"]["demographic_parity_gap"])
    caccA = comp["final_A"]["acc"]; caccB = comp["final_B"]["acc"]
    tred = (tA - tB)/tA*100 if tA else 0; cred = (cA - cB)/cA*100 if cA else 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    # Toxicity (positive)
    ax = axes[0]
    b = ax.bar(["control", "intervened"], [tA, tB], color=["#c0392b", "#27ae60"])
    ax.set_title(f"Toxicity model — identity bias\n(bias IS linguistically separable)")
    ax.set_ylabel("P(flag benign identity sentence as toxic)")
    ax.text(1, tB, f"-{tred:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax.text(0.5, max(tA,tB)*0.88, f"toxicity acc {taccA:.2f}→{taccB:.2f}", ha="center", fontsize=9)
    for bar,v in zip(b,[tA,tB]): ax.text(bar.get_x()+bar.get_width()/2, v/2, f"{v:.2f}", ha="center", color="white", fontweight="bold")
    # COMPAS (negative)
    ax = axes[1]
    b = ax.bar(["control", "intervened"], [cA, cB], color=["#c0392b", "#7f8c8d"])
    ax.set_title("COMPAS recidivism — racial gap\n(bias hidden in proxy features)")
    ax.set_ylabel("|demographic-parity gap|")
    ax.text(1, cB, f"-{cred:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax.text(0.5, max(cA,cB)*0.88, f"task acc {caccA:.2f}→{caccB:.2f}", ha="center", fontsize=9)
    for bar,v in zip(b,[cA,cB]): ax.text(bar.get_x()+bar.get_width()/2, v/2, f"{v:.2f}", ha="center", color="white", fontweight="bold")
    fig.suptitle("Same topological erasure: removes separable real-world bias, "
                 "measurably cannot remove proxy-encoded bias", fontweight="bold")
    fig.tight_layout(); fig.savefig(figdir / "HARM-1_realworld.png", dpi=300); plt.close(fig)
    print(f"[fig] HARM-1 written. toxicity bias {tA:.3f}->{tB:.3f} ({tred:+.0f}%); "
          f"COMPAS gap {cA:.3f}->{cB:.3f} ({cred:+.0f}%)")


if __name__ == "__main__":
    main()
