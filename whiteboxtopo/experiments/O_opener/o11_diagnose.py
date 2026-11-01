#!/usr/bin/env python3
"""O1.1 diagnosis — rebuild results/specimens.csv and results/O1_1_excluded.csv from every results/ckpt/<name>/final.pt
(SPEC_O O1.1 diagnostics + AMENDMENTS #2). Idempotent; run after training (or any time) — no specimen is retrained.

  python o11_diagnose.py            # all specimens with a final.pt
"""
from __future__ import annotations
import csv, glob, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import olib  # noqa: E402
from sweep_tosho import P, RES, CK, fourier_kappa, fourier_kappa_f, inclusion_rule, marginal_logits  # noqa: E402

FIELDS = ["name", "kind", "condition", "seed", "steps", "recipe", "train_acc", "test_acc", "kappa_emb", "n_eff_emb", "kappa_f", "n_eff_f",
          "n_freq_f", "K_f", "kappa_f_leak", "shortcut_share", "acc_marginal", "ang_err", "included", "reason", "wall"]


def n_eff_of(share):
    share = np.asarray(share, float); share = share / share.sum(); return float(np.exp(-(share * np.log(share + 1e-300)).sum()))


def emb_spectrum(E):
    F = np.fft.fft(E[:P], axis=0); pw = (np.abs(F) ** 2).sum(1); half = (P - 1) // 2
    folded = pw[1:half + 1] + pw[P - half:][::-1]; return folded / folded.sum()


def main():
    rows = []
    for final in sorted(glob.glob(os.path.join(CK, "*", "final.pt"))):
        name = os.path.basename(os.path.dirname(final))
        if name.startswith("grok_"):
            m, ck, data = olib.load_grok(name); cfg = ck["cfg"]
            L = olib.grok_logits(m, data)                                      # all p^2 pairs, leaked shortcut for planted
            kappa_f, K_f, share, n_eff = fourier_kappa_f(L); kappa_f_leak = shortcut_share = acc_marg = ""
            if data["planted"]:                                                # AMENDMENTS #3: marginalise over the shortcut token
                Lm = marginal_logits(m, data["a"], data["b"]); kappa_f_leak = f"{kappa_f:.4f}"; shortcut_share = f"{1 - kappa_f:.4f}"
                kappa_f, K_f, share, n_eff = fourier_kappa_f(Lm); acc_marg = f"{float((Lm.argmax(1) == data['c']).mean()):.4f}"
            E = m.wte.detach().cpu().numpy(); kappa_emb, _ = fourier_kappa(E); n_eff_emb = n_eff_of(emb_spectrum(E))
            tea = float(ck["test_acc"]); ok, reason = inclusion_rule(cfg["condition"], tea, kappa_f)
            if cfg.get("recipe", 1) < 2: ok, reason = False, "reference specimen (recipe 1, AMENDMENTS #4)" + (f"; would be: {reason}" if reason else "")
            rows.append(dict(name=name, kind="grok", condition=cfg["condition"], seed=cfg["seed"], steps=ck["epoch"], recipe=cfg.get("recipe", 1),
                             train_acc=f"{ck['train_acc']:.4f}", test_acc=f"{tea:.4f}", kappa_emb=f"{kappa_emb:.4f}", n_eff_emb=f"{n_eff_emb:.1f}",
                             kappa_f=f"{kappa_f:.4f}", n_eff_f=f"{n_eff:.1f}", n_freq_f=len(K_f), K_f=" ".join(map(str, K_f)),
                             kappa_f_leak=kappa_f_leak, shortcut_share=shortcut_share, acc_marginal=acc_marg, ang_err="",
                             included=int(ok), reason=reason, wall=f"{ck.get('wall', float('nan')):.0f}"))
        elif name.startswith("ring_"):
            ck = torch.load(final, map_location="cpu"); e = float(ck["ang_err"]); ok = e < 0.15
            rows.append(dict(name=name, kind="ring", condition="ring", seed=ck["cfg"]["seed"], steps=ck["step"], recipe=1, train_acc="", test_acc="",
                             kappa_emb="", n_eff_emb="", kappa_f="", n_eff_f="", n_freq_f="", K_f="", kappa_f_leak="", shortcut_share="", acc_marginal="", ang_err=f"{e:.4f}", included=int(ok),
                             reason="" if ok else f"angular error {e:.3f} >= 0.15", wall=f"{ck.get('wall', float('nan')):.0f}"))
    with open(os.path.join(RES, "specimens.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    with open(os.path.join(RES, "O1_1_excluded.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "reason"]); w.writeheader()
        for r in rows:
            if not r["included"]: w.writerow(dict(name=r["name"], reason=r["reason"]))
    for r in rows:
        print(f"{r['name']:20s} test {r['test_acc'] or '-':6s} kappa_emb {r['kappa_emb'] or '-':6s} kappa_f {r['kappa_f'] or '-':6s} "
              f"n_eff_f {r['n_eff_f'] or '-':5s} n_freq {r['n_freq_f'] or '-':>3} ang {r['ang_err'] or '-':6s} included={r['included']} {r['reason']}")
    n_in = sum(r["included"] for r in rows); print(f"{n_in}/{len(rows)} included")


if __name__ == "__main__":
    main()
