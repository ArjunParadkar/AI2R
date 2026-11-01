"""Nerve, sections, restrictions, agreement and the Čech-style obstruction (SPEC §5–6).

Obstruction construction (SPEC §6.2): with agreement J_e on each nerve edge,
    obs(τ) = #{ e : J_e < τ ≤ J_bott(e) },  J_bott(e) = widest-path agreement around e,
    m_e = max(0, J_bott(e) − J_e),  Ω = Σ m_e = ∫ obs dτ,  Ω_rel = Ω / Σ (1 − J_e).
"""
from __future__ import annotations

import itertools

import numpy as np


# ---------------------------------------------------------------- cover / nerve
def nerve(members, n_min=10):
    """members: bool [N,k]. Returns (edges [(i,j)], triangles [(i,j,k)], overlap sizes dict)."""
    k = members.shape[1]
    M = members.astype(np.int32)
    ov = M.T @ M
    edges = [(i, j) for i in range(k) for j in range(i + 1, k) if ov[i, j] >= n_min]
    eset = set(edges)
    tris = []
    for i, j, l in itertools.combinations(range(k), 3):
        if (i, j) in eset and (i, l) in eset and (j, l) in eset:
            if int((M[:, i] & M[:, j] & M[:, l]).sum()) >= n_min:
                tris.append((i, j, l))
    return edges, tris, ov


# ---------------------------------------------------------------- sections
def section(scores, idx, k_e):
    """Top-k_e edges by |mean score over rows idx|. Returns sorted int array."""
    m = np.abs(scores[idx].mean(0))
    top = np.argpartition(-m, k_e - 1)[:k_e]
    return np.sort(top)


def restrict(scores, C, overlap_idx, k_r):
    """Re-score C on the overlap's rows, keep the k_r most important. Sorted int array."""
    m = np.abs(scores[overlap_idx][:, C].mean(0))
    top = np.argpartition(-m, k_r - 1)[:k_r] if k_r < len(C) else np.arange(len(C))
    return np.sort(C[top])


def jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def cover_means(scores, members, edges, weights=None):
    """Mean score vector per cluster and per overlap (optionally row-weighted, e.g. bootstrap
    counts). Returns (M [k, n_edges], O [|E|, n_edges])."""
    N, k = members.shape
    w = np.ones(N) if weights is None else np.asarray(weights, float)
    Mw = members * w[:, None]                                   # [N,k]
    M = (Mw.T @ scores) / np.maximum(Mw.sum(0)[:, None], 1e-12)
    if edges:
        Ow = np.stack([Mw[:, i] * members[:, j] for i, j in edges], 1)   # [N,|E|]
        O = (Ow.T @ scores) / np.maximum(Ow.sum(0)[:, None], 1e-12)
    else:
        O = np.zeros((0, scores.shape[1]))
    return M, O


def agreements_from_means(M, O, edges, k_e, restrict_frac=0.5, return_sets=False):
    """J (restricted, primary) and J_raw per nerve edge from precomputed means."""
    k_r = int(np.ceil(k_e * restrict_frac))
    aM, aO = np.abs(M), np.abs(O)
    secs = {}
    J, Jraw, sets = [], [], []
    for n, (i, j) in enumerate(edges):
        for c in (i, j):
            if c not in secs:
                secs[c] = np.sort(np.argpartition(-aM[c], k_e - 1)[:k_e])
        ri = secs[i][np.argpartition(-aO[n][secs[i]], k_r - 1)[:k_r]] if k_r < k_e else secs[i]
        rj = secs[j][np.argpartition(-aO[n][secs[j]], k_r - 1)[:k_r]] if k_r < k_e else secs[j]
        J.append(jaccard(ri, rj)); Jraw.append(jaccard(secs[i], secs[j]))
        if return_sets:
            sets.append((np.sort(ri), np.sort(rj)))
    out = (np.array(J), np.array(Jraw))
    return out + ((secs, sets),) if return_sets else out


def pair_agreements(scores, members, edges, k_e, restrict_frac=0.5, return_sets=False):
    """J (restricted, primary) and J_raw (identity restriction) per nerve edge."""
    k_r = int(np.ceil(k_e * restrict_frac))
    secs = {}
    J, Jraw, sets = [], [], []
    for (i, j) in edges:
        for c in (i, j):
            if c not in secs:
                secs[c] = section(scores, np.where(members[:, c])[0], k_e)
        ov = np.where(members[:, i] & members[:, j])[0]
        ri = restrict(scores, secs[i], ov, k_r)
        rj = restrict(scores, secs[j], ov, k_r)
        J.append(jaccard(ri, rj)); Jraw.append(jaccard(secs[i], secs[j]))
        if return_sets:
            sets.append((ri, rj))
    out = (np.array(J), np.array(Jraw))
    return out + ((secs, sets),) if return_sets else out


# ---------------------------------------------------------------- obstruction
class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[a] = b
        return a != b


def bottleneck(n_v, edges, J):
    """J_bott(e): widest-path agreement between e's endpoints in N \\ {e} (0 if disconnected).
    Kruskal in decreasing J on the other edges; the first J at which the endpoints join."""
    E = len(edges)
    order = np.argsort(-J, kind="stable")
    out = np.zeros(E)
    for e in range(E):
        uf = _UF(n_v)
        a, b = edges[e]
        val = 0.0
        for f in order:
            if f == e:
                continue
            uf.union(*edges[f])
            if uf.find(a) == uf.find(b):
                val = J[f]; break
        out[e] = val
    return out


def h0(n_v, edges, J, tau):
    uf = _UF(n_v)
    for e, (a, b) in enumerate(edges):
        if J[e] >= tau:
            uf.union(a, b)
    return len({uf.find(v) for v in range(n_v)})


def obstruction(n_v, edges, J, taus=None):
    """SPEC §6. Returns dict with per-edge J_bott, m_e and curves obs(τ), H⁰(τ)."""
    J = np.asarray(J, dtype=float)
    if taus is None:
        taus = np.round(np.arange(0, 1.0001, 0.02), 4)
    if len(edges) == 0:
        return dict(J_bott=np.zeros(0), m_e=np.zeros(0), taus=taus, obs=np.zeros(len(taus), int),
                    h0=np.array([n_v] * len(taus)), omega=0.0, omega_rel=0.0, omega_per_edge=0.0,
                    n_edges=0, cyclomatic=0)
    Jb = bottleneck(n_v, edges, J)
    m_e = np.maximum(0.0, Jb - J)
    obs = np.array([int(((J < t) & (t <= Jb)).sum()) for t in taus])
    H0 = np.array([h0(n_v, edges, J, t) for t in taus])
    disagreement = float((1 - J).sum())
    omega = float(m_e.sum())
    uf = _UF(n_v)
    for (a, b) in edges:
        uf.union(a, b)
    ncomp = len({uf.find(v) for v in range(n_v)})
    return dict(J_bott=Jb, m_e=m_e, taus=taus, obs=obs, h0=H0, omega=omega,
                omega_rel=omega / disagreement if disagreement > 0 else 0.0,
                omega_per_edge=omega / len(edges), n_edges=len(edges),
                cyclomatic=len(edges) - n_v + ncomp)


def triangle_frustration(tris, edges, J, taus):
    """Fraction of nerve triangles with exactly one inconsistent edge, per τ (SPEC §6.4)."""
    if not tris:
        return np.full(len(taus), np.nan)
    eid = {e: n for n, e in enumerate(edges)}
    out = []
    for t in taus:
        bad = 0
        for (i, j, l) in tris:
            n_inc = sum(J[eid[p]] < t for p in ((i, j), (i, l), (j, l)))
            bad += (n_inc == 1)
        out.append(bad / len(tris))
    return np.array(out)


# ---------------------------------------------------------------- brute force (tests)
def _gf2_rank(rows):
    rows = [int(r) for r in rows if r]
    rank = 0
    while rows:
        piv = max(rows)
        rows.remove(piv)
        hb = piv.bit_length() - 1
        rows = [r ^ piv if (r >> hb) & 1 else r for r in rows]
        rows = [r for r in rows if r]
        rank += 1
    return rank


def brute_force_obs(n_v, edges, J, tau):
    """Enumerate all simple cycles; frustrated = exactly one inconsistent edge;
    return dim( span(frustrated) + span(consistent) ) − dim span(consistent) over GF(2)."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n_v))
    eid = {}
    for n, (a, b) in enumerate(edges):
        G.add_edge(a, b); eid[frozenset((a, b))] = n
    frus, cons = [], []
    for cyc in nx.simple_cycles(G):
        vec, n_inc = 0, 0
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            n = eid[frozenset((a, b))]
            vec |= 1 << n
            n_inc += J[n] < tau
        if n_inc == 0:
            cons.append(vec)
        elif n_inc == 1:
            frus.append(vec)
    return _gf2_rank(frus + cons) - _gf2_rank(cons)
