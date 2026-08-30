"""E3 runner: induction control FIRST, then IOI; sweeps k, k_e, embedding; nulls; figures.
Usage: ../.venv-tl/bin/python run_e3.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e3lib import cluster as cl  # noqa: E402
from e3lib import sheaf as sh  # noqa: E402
from e3lib.eap import EdgeGraph, eap_all, induction_metric, ioi_metric, prepare_model  # noqa: E402
from e3lib.pipeline import analyse_cover_multi, blocking_pairs, cover_from_embedding  # noqa: E402
from e3lib.prompts import make_induction_prompts, make_ioi_prompts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = "--smoke" in sys.argv
RES = os.path.join(HERE, "results_smoke" if SMOKE else "results")
CACHE = os.path.join(HERE, "cache", "smoke" if SMOKE else "full")
os.makedirs(RES, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
LOG = open(os.path.join(RES, "run.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def get_scores(model, task, n_per, graph, batch):
    f = os.path.join(CACHE, f"scores_{task}.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        return z["scores"], {6: z["emb6"], 11: z["emb11"]}, z["m_clean"], z["m_corr"], list(z["metas"])
    if task == "ioi":
        prompts = make_ioi_prompts(model, n_per_cell=n_per); metric = ioi_metric
    else:
        prompts = make_induction_prompts(model, n_per_len=n_per * 30 // 4 if n_per < 80 else 600); metric = induction_metric
    t = time.time()
    scores, embs, mc, mx = eap_all(model, prompts, metric, graph, batch, (6, 11), "cuda", log)
    log(f"[{task}] EAP for {len(prompts)} prompts in {time.time()-t:.0f}s; metric clean {mc.mean():.2f} corrupt {mx.mean():.2f}")
    metas = [p["meta"] for p in prompts]
    np.savez(f, scores=scores, emb6=embs[6], emb11=embs[11], m_clean=mc, m_corr=mx, metas=np.array(metas, dtype=object))
    return scores, embs, mc, mx, metas


def draw_nerve(cover, res, metas, key, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    edges, members = cover["edges"], cover["members"]
    ob = res["obstruction"]
    G = nx.Graph(); k = members.shape[1]
    G.add_nodes_from(range(k))
    for n, (i, j) in enumerate(edges):
        G.add_edge(i, j, w=1 - res["J"][n], ov=int(cover["overlap"][i, j]), m=ob["m_e"][n], J=res["J"][n])
    pos = nx.kamada_kawai_layout(G, weight=None) if G.number_of_edges() else nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=(8, 7))
    sizes = members.sum(0)
    labels = {}
    for i in range(k):
        idx = np.where(members[:, i])[0]
        vals = [str(metas[p][key]) for p in idx]
        u, c = np.unique(vals, return_counts=True); o = np.argsort(-c)
        labels[i] = f"{i}\n{u[o[0]]} {c[o[0]]/len(idx):.0%}" + (f"\n{u[o[1]]} {c[o[1]]/len(idx):.0%}" if len(o) > 1 else "")
    if G.number_of_edges():
        ws = np.array([G[a][b]["w"] for a, b in G.edges()])
        ovs = np.array([G[a][b]["ov"] for a, b in G.edges()])
        ms = np.array([G[a][b]["m"] for a, b in G.edges()])
        ec = nx.draw_networkx_edges(G, pos, ax=ax, width=1 + 4 * ovs / ovs.max(), edge_color=ws,
                                    edge_cmap=plt.cm.viridis, edge_vmin=0, edge_vmax=1, alpha=0.85)
        fr = [e for e, m in zip(G.edges(), ms) if m > 0]
        if fr:
            nx.draw_networkx_edges(G, pos, edgelist=fr, ax=ax, width=[2 + 8 * G[a][b]["m"] for a, b in fr],
                                   edge_color="none", style="dashed", alpha=0.9)
            nx.draw_networkx_edges(G, pos, edgelist=fr, ax=ax, width=[2 + 8 * G[a][b]["m"] for a, b in fr],
                                   edge_color="crimson", style="dashed", alpha=0.9)
        for (a, b), m, w in zip(G.edges(), ms, ws):
            x, y = (pos[a] + pos[b]) / 2
            ax.text(x, y, f"{1-w:.2f}" + (f"\nm={m:.2f}" if m > 0 else ""), fontsize=6, ha="center",
                    color="crimson" if m > 0 else "black", bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1)); sm.set_array([])
        fig.colorbar(sm, ax=ax, label="disagreement 1 − J (edge colour); width ∝ overlap size")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300 + 1500 * sizes / sizes.max(), node_color="#e8e8f0", edgecolors="k")
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6)
    ax.set_title(title + "\nred dashed = frustrated pairs (m_e > 0: no relabelling zeroes them); label = dominant " + key, fontsize=9)
    ax.axis("off"); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    t0 = time.time()
    from transformer_lens import HookedTransformer
    model = prepare_model(HookedTransformer.from_pretrained("gpt2", device="cuda"))
    graph = EdgeGraph()
    n_per = 4 if args.smoke else 80
    KS = [8, 10] if args.smoke else [8, 10, 12, 15]
    KES = [100, 200] if args.smoke else [50, 100, 200, 400, 800]
    NS, NB = (20, 20) if args.smoke else (200, 200)
    SUMMARY = dict(k=10, k_e=200, emb=11)
    summary_rows, curve_rows, pair_rows, cover_rows, bp_all = [], [], [], [], {}
    keep = {}
    for task in ["induction", "ioi"]:               # control FIRST (SPEC §7)
        scores, embs, mc, mx, metas = get_scores(model, task, n_per, graph, 8)
        key = "L" if task == "induction" else "order"
        keys = ["L"] if task == "induction" else ["order", "template"]
        log(f"[{task}] N={len(metas)} metric clean {mc.mean():.2f}±{mc.std():.2f} corrupt {mx.mean():.2f}; frac clean>corrupt {(mc>mx).mean():.3f}")
        for emb in [11, 6]:
            for k in KS:
                cover = cover_from_embedding(embs[emb], k, 0.5, 20, 0, 10)
                edges, tris = cover["edges"], cover["tris"]
                comp = cl.composition(cover["members"], metas, keys)
                for c in comp:
                    cover_rows.append(dict(task=task, emb=emb, k=k, theta=cover["theta"], **{kk: (json.dumps(v) if isinstance(v, dict) else v) for kk, v in c.items()}))
                hard = cl.hard_cover(cover["U"]); hard_edges, _, _ = sh.nerve(hard, 10)
                log(f"[{task} emb{emb} k={k}] theta={cover['theta']} sizes={cover['members'].sum(0).tolist()} "
                    f"multi-membership={int((cover['members'].sum(1)>1).sum())} nerve |E|={len(edges)} tris={len(tris)} "
                    f"cyclomatic={len(edges)-k+1 if edges else 0} hard-cover edges={len(hard_edges)}")
                t_an = time.time()
                multi = analyse_cover_multi(scores, cover["members"], edges, tris, KES, 0.5, None, NS, NB, 1)
                log(f"   analysis ({NS} shuffles + {NB} bootstraps x {len(KES)} k_e) in {time.time()-t_an:.0f}s")
                for k_e in KES:
                    res = multi[k_e]
                    ob = res["obstruction"]
                    is_sum = (k == SUMMARY["k"] and k_e == SUMMARY["k_e"])
                    summary_rows.append(dict(task=task, emb=emb, k=k, k_e=k_e, theta=cover["theta"], n_edges=len(edges), n_tris=len(tris),
                                             cyclomatic=ob["cyclomatic"], mean_J=float(np.mean(res["J"])) if len(edges) else np.nan,
                                             mean_J_raw=float(np.mean(res["J_raw"])) if len(edges) else np.nan,
                                             omega=ob["omega"], omega_rel=ob["omega_rel"], omega_per_edge=ob["omega_per_edge"],
                                             omega_ci_lo=res["boot_omega_ci"][0], omega_ci_hi=res["boot_omega_ci"][1],
                                             null_lo=res["null_band"][0], null_hi=res["null_band"][1], null_mean=float(np.mean(res["null_omega"])),
                                             p_null=res["p_null"], n_frustrated_edges=int((ob["m_e"] > 0).sum()),
                                             max_obs=int(ob["obs"].max()), hard_cover_edges=len(hard_edges), summary=is_sum))
                    for t_i, tau in enumerate(ob["taus"]):
                        curve_rows.append(dict(task=task, emb=emb, k=k, k_e=k_e, tau=tau, obs=int(ob["obs"][t_i]), h0=int(ob["h0"][t_i]),
                                               obs_ci_lo=res["boot_obs_ci"][0][t_i] if len(edges) else 0, obs_ci_hi=res["boot_obs_ci"][1][t_i] if len(edges) else 0,
                                               null_obs_hi=np.percentile(res["null_obs"][:, t_i], 97.5) if len(edges) else 0,
                                               tri_frustration=res["triangle_frustration"][t_i]))
                    for n, (i, j) in enumerate(edges):
                        pair_rows.append(dict(task=task, emb=emb, k=k, k_e=k_e, i=i, j=j, size_i=int(cover["members"][:, i].sum()),
                                              size_j=int(cover["members"][:, j].sum()), overlap=int(cover["overlap"][i, j]),
                                              J=res["J"][n], J_raw=res["J_raw"][n], J_ci_lo=res["boot_J_ci"][0][n], J_ci_hi=res["boot_J_ci"][1][n],
                                              J_bott=ob["J_bott"][n], m_e=ob["m_e"][n], m_ci_lo=res["boot_m_ci"][0][n], m_ci_hi=res["boot_m_ci"][1][n]))
                    log(f"   k_e={k_e}: Ω={ob['omega']:.3f} (boot CI {res['boot_omega_ci'][0]:.3f}-{res['boot_omega_ci'][1]:.3f}) "
                        f"Ω_rel={ob['omega_rel']:.3f} Ω/|E|={ob['omega_per_edge']:.3f} null band {res['null_band'][0]:.3f}-{res['null_band'][1]:.3f} p={res['p_null']:.3f} "
                        f"meanJ={np.mean(res['J']) if len(edges) else float('nan'):.3f} frustrated edges={(ob['m_e']>0).sum()} max obs={ob['obs'].max()}")
                    if is_sum or (k == SUMMARY["k"] and k_e in (100, 400)):
                        bp = blocking_pairs(res, edges, cover["members"], metas, keys, graph.edge_names, top=10)
                        bp_all[f"{task}_emb{emb}_k{k}_ke{k_e}"] = bp
                    if is_sum:
                        keep[(task, emb)] = (cover, res)
                        draw_nerve(cover, res, metas, key, os.path.join(RES, f"fig_nerve_{task}_emb{emb}_k{k}_ke{k_e}.png"),
                                   f"{task}: nerve of the k={k} cover (layer-{emb} embedding), k_e={k_e}")
        del scores
    pd.DataFrame(summary_rows).to_csv(os.path.join(RES, "summary.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(os.path.join(RES, "obstruction_curves.csv"), index=False)
    pd.DataFrame(pair_rows).to_csv(os.path.join(RES, "pairs.csv"), index=False)
    pd.DataFrame(cover_rows).to_csv(os.path.join(RES, "covers.csv"), index=False)
    with open(os.path.join(RES, "blocking_pairs.json"), "w") as f:
        json.dump(bp_all, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    # ---- figures
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cur = pd.DataFrame(curve_rows); summ = pd.DataFrame(summary_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for task, col in (("induction", "tab:blue"), ("ioi", "tab:red")):
        for emb, ls in ((11, "-"), (6, "--")):
            c = cur[(cur.task == task) & (cur.emb == emb) & (cur.k == SUMMARY["k"]) & (cur.k_e == SUMMARY["k_e"])]
            axes[0].plot(c.tau, c.obs, ls, color=col, label=f"{task} L{emb}")
            if emb == 11:
                axes[0].fill_between(c.tau, c.obs_ci_lo, c.obs_ci_hi, color=col, alpha=0.15)
                axes[0].plot(c.tau, c.null_obs_hi, ":", color=col, alpha=0.7, label=f"{task} shuffle-null 97.5%")
            axes[1].plot(c.tau, c.h0, ls, color=col, label=f"{task} L{emb}")
    axes[0].set(xlabel="τ (agreement threshold)", ylabel="obs(τ) = # independent frustrated cycles", title=f"Obstruction vs τ (k={SUMMARY['k']}, k_e={SUMMARY['k_e']})")
    axes[1].set(xlabel="τ", ylabel="H⁰(τ) = # consistent components", title="H⁰ vs τ")
    for ax in axes[:2]: ax.legend(fontsize=7)
    for task, col in (("induction", "tab:blue"), ("ioi", "tab:red")):
        s = summ[(summ.task == task) & (summ.emb == 11) & (summ.k == SUMMARY["k"])]
        axes[2].errorbar(s.k_e, s.omega_rel, fmt="o-", color=col, label=f"{task} Ω_rel")
        s2 = summ[(summ.task == task) & (summ.emb == 11) & (summ.k == SUMMARY["k"])]
        axes[2].plot(s2.k_e, s2.null_hi / np.maximum(1e-9, (1 - s2.mean_J) * s2.n_edges), ":", color=col, label=f"{task} null 97.5% (Ω_rel units)")
    axes[2].set(xscale="log", xlabel="k_e (circuit size)", ylabel="Ω_rel = unexplainable disagreement fraction", title="Integrated obstruction vs circuit size")
    axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_obstruction_vs_tau.png"), dpi=300); plt.close(fig)
    # null vs observed
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, task in zip(axes, ("induction", "ioi")):
        if (task, 11) in keep:
            _, res = keep[(task, 11)]
            ax.hist(res["null_omega"], bins=30, color="grey", alpha=0.7, label="prompt-shuffle null (200)")
            ax.axvline(res["obstruction"]["omega"], color="crimson", lw=2, label=f"observed Ω={res['obstruction']['omega']:.2f}")
            ax.axvspan(*res["boot_omega_ci"], color="crimson", alpha=0.15, label="bootstrap 95% CI")
            ax.set(title=f"{task}: Ω vs shuffle null (k=10, k_e=200, L11)", xlabel="Ω = Σ frustrated lifetimes"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_null_vs_observed.png"), dpi=300); plt.close(fig)
    # sweep heatmap: omega_rel over k x k_e for each task (emb 11), with null-exceedance marks
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for a, (task, emb) in enumerate([("induction", 11), ("ioi", 11), ("induction", 6), ("ioi", 6)]):
        s = summ[(summ.task == task) & (summ.emb == emb)].pivot(index="k", columns="k_e", values="omega_rel")
        p = summ[(summ.task == task) & (summ.emb == emb)].pivot(index="k", columns="k_e", values="p_null")
        im = axes[a].imshow(s.values, vmin=0, vmax=max(0.3, np.nanmax(summ.omega_rel)), cmap="magma", aspect="auto")
        axes[a].set_xticks(range(len(s.columns))); axes[a].set_xticklabels(s.columns); axes[a].set_yticks(range(len(s.index))); axes[a].set_yticklabels(s.index)
        for ii in range(s.shape[0]):
            for jj in range(s.shape[1]):
                axes[a].text(jj, ii, f"{s.values[ii,jj]:.2f}\n{'*' if p.values[ii,jj] < 0.025 else ''}", ha="center", va="center", fontsize=7, color="w")
        axes[a].set(title=f"{task} L{emb}: Ω_rel (* = above shuffle null)", xlabel="k_e", ylabel="k")
    fig.colorbar(im, ax=axes, label="Ω_rel"); fig.savefig(os.path.join(RES, "fig_sweep_omega_rel.png"), dpi=300); plt.close(fig)
    log(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
