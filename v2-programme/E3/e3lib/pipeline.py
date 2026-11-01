"""explanation_obstruction(model, prompts, task_fn, ...) — the future whitebox.gluing.

Framework: sheaf-of-explanations (arXiv 2603.17141) and the sheaf/holonomy atlas of
"A Gauge Theory of Superposition". Circuits: EAP (Syed, Rager & Conmy 2023; Hanna et
al. 2024) on the ACDC graph (Conmy et al. 2023). This module is only the measurement
instrument: cover → local circuits → restrictions → agreement → frustrated-cycle
obstruction on the nerve (SPEC §6).
"""
from __future__ import annotations

import numpy as np

from . import cluster as cl
from . import sheaf as sh
from .eap import EdgeGraph, eap_all, prepare_model  # noqa


def cover_from_embedding(emb, k, theta=0.5, n_pca=20, seed=0, n_min=10, adaptive=True, m=1.5,
                         method="knn2"):
    """Overlapping cover. method="knn2" (SPEC amendment 1): 2-nearest-centre cover on k-means++
    centres. method="fcm": the original §3 fuzzy c-means + relative rule (degenerate on these
    embeddings; kept for the record)."""
    X = cl.preprocess(emb, n_pca)
    if method == "knn2":
        members, d2, _ = cl.knn_centre_cover(X, k, 2, seed)
        U = -d2                                   # "membership" proxy: argmax = nearest centre
        edges, tris, ov = sh.nerve(members, n_min)
        return dict(U=U, members=members, edges=edges, tris=tris, overlap=ov, theta=None, fcm_obj=None)
    U, _, obj = cl.fcm(X, k, m=m, seed=seed)
    members = cl.soft_cover(U, theta)
    edges, tris, ov = sh.nerve(members, n_min)
    used_theta = theta
    if adaptive and _cyclomatic(k, edges) < 1 and theta > 0.33:
        members = cl.soft_cover(U, 0.33)
        edges, tris, ov = sh.nerve(members, n_min)
        used_theta = 0.33
    return dict(U=U, members=members, edges=edges, tris=tris, overlap=ov, theta=used_theta, fcm_obj=obj)


def _cyclomatic(n_v, edges):
    uf = sh._UF(n_v)
    for a, b in edges:
        uf.union(a, b)
    return len(edges) - n_v + len({uf.find(v) for v in range(n_v)})


def analyse_cover(scores, members, edges, tris, k_e=200, restrict_frac=0.5, taus=None,
                  n_shuffle=200, n_boot=200, seed=1, log=None):
    return analyse_cover_multi(scores, members, edges, tris, [k_e], restrict_frac, taus,
                               n_shuffle, n_boot, seed)[k_e]


def analyse_cover_multi(scores, members, edges, tris, k_es, restrict_frac=0.5, taus=None,
                        n_shuffle=200, n_boot=200, seed=1):
    """Everything after the model, for several circuit sizes k_e at once:
    agreement, obstruction, prompt-shuffle null (i), bootstrap CIs (iii).
    Bootstrap = row counts as weights (no copies); nerve fixed; memberships travel with rows."""
    n_v = members.shape[1]
    N = scores.shape[0]
    M, O = sh.cover_means(scores, members, edges)
    base = {}
    for k_e in k_es:
        J, Jraw, (secs, sets) = sh.agreements_from_means(M, O, edges, k_e, restrict_frac, True)
        ob = sh.obstruction(n_v, edges, J, taus)
        base[k_e] = dict(J=J, J_raw=Jraw, sections=secs, restricted_sets=sets, obstruction=ob, taus=ob["taus"],
                         triangle_frustration=sh.triangle_frustration(tris, edges, J, ob["taus"]),
                         null_omega=[], null_obs=[], boot_J=[], boot_m=[], boot_omega=[], boot_obs=[])
    taus = base[k_es[0]]["taus"]
    rng = np.random.default_rng(seed)
    for _ in range(n_shuffle):                      # (i) permute prompt rows of the membership matrix
        Mn, On = sh.cover_means(scores, members[rng.permutation(N)], edges)
        for k_e in k_es:
            Jn, _ = sh.agreements_from_means(Mn, On, edges, k_e, restrict_frac)
            obn = sh.obstruction(n_v, edges, Jn, taus)
            base[k_e]["null_omega"].append(obn["omega"]); base[k_e]["null_obs"].append(obn["obs"])
    for _ in range(n_boot):                         # (iii) bootstrap rows with replacement
        w = np.bincount(rng.integers(0, N, N), minlength=N)
        Mb, Ob = sh.cover_means(scores, members, edges, w)
        for k_e in k_es:
            Jb, _ = sh.agreements_from_means(Mb, Ob, edges, k_e, restrict_frac)
            obb = sh.obstruction(n_v, edges, Jb, taus)
            b = base[k_e]; b["boot_J"].append(Jb); b["boot_m"].append(obb["m_e"]); b["boot_omega"].append(obb["omega"]); b["boot_obs"].append(obb["obs"])
    q = lambda a, p: np.percentile(a, p, axis=0) if len(a) else np.nan
    out = {}
    for k_e in k_es:
        b = base[k_e]
        for key in ("null_omega", "null_obs", "boot_J", "boot_m", "boot_omega", "boot_obs"):
            b[key] = np.array(b[key])
        ob = b["obstruction"]
        b.update(null_band=(q(b["null_omega"], 2.5), q(b["null_omega"], 97.5)),
                 p_null=float((b["null_omega"] >= ob["omega"]).mean()) if n_shuffle else np.nan,
                 boot_J_ci=(q(b["boot_J"], 2.5), q(b["boot_J"], 97.5)), boot_m_ci=(q(b["boot_m"], 2.5), q(b["boot_m"], 97.5)),
                 boot_omega_ci=(q(b["boot_omega"], 2.5), q(b["boot_omega"], 97.5)),
                 boot_obs_ci=(q(b["boot_obs"], 2.5), q(b["boot_obs"], 97.5)))
        out[k_e] = b
    return out


def blocking_pairs(res, edges, members, metas=None, meta_keys=(), edge_names=None, top=10, n_diff=10):
    """Rank nerve edges by frustrated lifetime m_e, with composition and distinguishing model edges."""
    ob = res["obstruction"]
    order = np.lexsort((-(1 - res["J"]), -ob["m_e"]))
    rows = []
    for r in order[:top]:
        i, j = edges[r]
        ov = members[:, i] & members[:, j]
        row = dict(pair=(i, j), size_i=int(members[:, i].sum()), size_j=int(members[:, j].sum()),
                   overlap=int(ov.sum()), J=float(res["J"][r]), J_raw=float(res["J_raw"][r]),
                   J_bott=float(ob["J_bott"][r]), m_e=float(ob["m_e"][r]),
                   m_e_ci=(float(res["boot_m_ci"][0][r]), float(res["boot_m_ci"][1][r])),
                   J_ci=(float(res["boot_J_ci"][0][r]), float(res["boot_J_ci"][1][r])))
        if metas is not None:
            for key in meta_keys:
                for name, mask in (("i", members[:, i]), ("j", members[:, j]), ("ij", ov)):
                    vals = [metas[p][key] for p in np.where(mask)[0]]
                    u, c = np.unique(vals, return_counts=True)
                    o = np.argsort(-c)[:6]
                    row[f"{key}_{name}"] = {str(u[x]): int(c[x]) for x in o}
        if edge_names is not None:
            ri, rj = res["restricted_sets"][r]
            only_i = sorted(set(ri.tolist()) - set(rj.tolist()))[:n_diff]
            only_j = sorted(set(rj.tolist()) - set(ri.tolist()))[:n_diff]
            row["only_in_i"] = [edge_names[e] for e in only_i]
            row["only_in_j"] = [edge_names[e] for e in only_j]
        rows.append(row)
    return rows


def explanation_obstruction(model, prompts, task_fn, k=10, k_e=200, emb_layer=11, theta=0.5,
                            n_pca=20, n_min=10, restrict_frac=0.5, n_shuffle=200, n_boot=200,
                            batch_size=8, device="cuda", scores=None, embs=None, meta_keys=(),
                            log=print, seed=0):
    """Measure the obstruction to gluing per-cluster EAP circuits into a global explanation.

    model     : HookedTransformer (GPT-2-small layout: 12 layers x 12 heads).
    prompts   : list of dicts with `clean`/`corrupt` strings (or `tokens`/`tokens_corrupt`),
                `answer_tok`, optional `wrong_tok`, `meta`.
    task_fn   : batch -> (logits -> per-prompt metric tensor), e.g. e3lib.eap.ioi_metric.
    scores/embs: optional cached per-prompt EAP scores [N, 32491] and embeddings {layer: [N,768]}.

    Returns dict(obstruction=Ω, obstruction_rel, h0 (curve), per_tau (DataFrame-ready dict),
                 blocking_pairs, nerve=(edges, tris, overlap), cover, J, null_band, p_null, ...).
    Honest docstring: the framework is arXiv 2603.17141 / the gauge-superposition paper; the
    circuits are EAP; this function is the measurement instrument, nothing more.
    """
    graph = EdgeGraph()
    if scores is None:
        prepare_model(model)
        scores, embs, mc, mx = eap_all(model, prompts, task_fn, graph, batch_size, (emb_layer,), device, log)
    cover = cover_from_embedding(embs[emb_layer], k, theta, n_pca, seed, n_min)
    res = analyse_cover(scores, cover["members"], cover["edges"], cover["tris"], k_e, restrict_frac,
                        None, n_shuffle, n_boot, seed + 1)
    metas = [p.get("meta", {}) for p in prompts]
    bp = blocking_pairs(res, cover["edges"], cover["members"], metas, meta_keys, graph.edge_names)
    ob = res["obstruction"]
    return dict(obstruction=ob["omega"], obstruction_rel=ob["omega_rel"], obstruction_per_edge=ob["omega_per_edge"],
                h0=ob["h0"], per_tau=dict(tau=ob["taus"], obs=ob["obs"], h0=ob["h0"],
                                          triangle_frustration=res["triangle_frustration"]),
                blocking_pairs=bp, nerve=(cover["edges"], cover["tris"], cover["overlap"]), cover=cover,
                J=res["J"], J_raw=res["J_raw"], m_e=ob["m_e"], J_bott=ob["J_bott"],
                null_band=res["null_band"], p_null=res["p_null"], boot_omega_ci=res["boot_omega_ci"],
                cyclomatic=ob["cyclomatic"], scores=scores, embs=embs, graph=graph, analysis=res)
