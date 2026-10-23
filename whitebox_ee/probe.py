"""Probe set, nodes, and the patch-run machinery shared by every estimator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .systems import Patch, readout


@dataclass
class Member:
    """One realiser of a node: site, orthonormal basis, nearest-centroid value map."""
    site: str
    U: np.ndarray                  # (d, k)
    centroids: np.ndarray          # (n_values, k) in U-coordinates
    value_ids: np.ndarray          # value label of each centroid row
    source: str = ""
    match_U: Optional[np.ndarray] = None   # random-null members: norm-matched to this basis

    def values(self, states_at_site: np.ndarray) -> np.ndarray:
        z = states_at_site @ self.U
        d2 = ((z[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        return self.value_ids[d2.argmin(1)]

    def dist_to_fibre(self, states_at_site: np.ndarray) -> np.ndarray:
        z = states_at_site @ self.U
        d2 = ((z[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        return np.sqrt(d2.min(1))


@dataclass
class Node:
    name: str
    kind: str                                   # 'input' | 'concept' | 'output'
    members: List[Member] = field(default_factory=list)
    values: Optional[np.ndarray] = None         # on the probe set
    site_index: int = 0                         # causal layer of the node's site (probe.site_layer); inputs -1; outputs max+1
    meta: dict = field(default_factory=dict)

    @property
    def primary(self) -> Optional[Member]:
        return self.members[0] if self.members else None

    @property
    def patch_members(self) -> List[Member]:
        """Interventions use only the realisers at the node's own site; members at
        other sites are bookkeeping (phase1.md §2)."""
        if not self.members:
            return []
        s = self.members[0].site
        return [m for m in self.members if m.site == s][:1]


def member_from_labels(states: np.ndarray, labels: np.ndarray, site: str, ridge: float = 1e-3,
                       sv_tol: float = 1e-2, max_k: Optional[int] = None, source: str = "probe") -> Optional[Member]:
    """Realiser of a labelled variable: column space of the ridge decoder of its
    centred one-hot code from the centred states (a probe direction set)."""
    labels = np.asarray(labels)
    vals = np.unique(labels)
    if len(vals) < 2:
        return None
    H = states - states.mean(0)
    Y = (labels[:, None] == vals[None, :]).astype(float)
    Y = Y - Y.mean(0)
    n, d = H.shape
    tr = float((H * H).sum())
    if tr <= 1e-12:
        return None
    lam = ridge * tr / d
    if n < d:   # dual form: W = H^T (H H^T + lam I)^-1 Y
        W = H.T @ np.linalg.solve(H @ H.T + lam * np.eye(n), Y)
    else:
        W = np.linalg.solve(H.T @ H + lam * np.eye(d), H.T @ Y)
    Uw, S, _ = np.linalg.svd(W, full_matrices=False)
    if S[0] <= 1e-12:
        return None
    k = int((S > sv_tol * S[0]).sum())
    k = min(k, len(vals) - 1)
    if max_k:
        k = min(k, max_k)
    U = Uw[:, :k]
    z = states @ U
    cents = np.stack([z[labels == v].mean(0) for v in vals])
    return Member(site, U, cents, vals, source)


class Probe:
    """A probe set X (training-distribution inputs) with cached clean run."""

    def __init__(self, system, X, seed: int = 0):
        self.sys = system
        self.X = np.asarray(X) if not isinstance(X, list) else X
        self.N = len(X)
        logits, states = system.run(X)
        self.logits = logits
        self.states = states
        self.read = readout(logits)
        self.inputs = system.input_values(X)
        self.sites = list(system.sites)
        self._idx = {s: i for i, s in enumerate(self.sites)}
        self.rng = np.random.default_rng(seed)
        self.n_runs = 0

    def site_idx(self, site):
        return self._idx[site]

    def site_layer(self, site) -> int:
        """Causal order of a site: sites in the same layer (different positions) are
        computed in parallel, so only an earlier layer can be a parent."""
        f = getattr(self.sys, "site_layer", None)
        return int(f(site)) if f else self._idx[site]

    def prev_site(self, site):
        """The state the block writing `site` read from (same position, previous layer)."""
        f = getattr(self.sys, "prev_site", None)
        if f:
            return f(site)
        i = self._idx[site]
        return self.sites[i - 1] if i > 0 else None

    # ------------------------------------------------------------------
    def run_patched(self, rows: np.ndarray, srcs: Sequence[Tuple[Node, np.ndarray]] = (),
                    freeze: Sequence[Node] = (), input_edits: Optional[Dict[str, np.ndarray]] = None,
                    extra_patches: Sequence[Patch] = (), sites: Optional[Sequence[str]] = None):
        """Run X[rows] with each node in `srcs` patched from X[src_rows] (all its
        members), each node in `freeze` patched from X[rows] itself (held at its
        clean value), and optional input edits. Returns (logits, states)."""
        rows = np.asarray(rows)
        X = self._take(rows)
        patches = []
        for node, src_rows in srcs:
            if node.kind == "input" and not node.members:
                if input_edits is None:
                    input_edits = {}
                input_edits[node.name] = self.inputs[node.name][src_rows]
                continue
            for m in node.patch_members:
                patches.append(Patch(m.site, m.U, self.states[m.site][src_rows], m.match_U))
        for node in freeze:
            for m in node.patch_members:
                patches.append(Patch(m.site, m.U, self.states[m.site][rows]))
        patches += list(extra_patches)
        if input_edits:
            X = self.sys.edit_inputs(X, input_edits)
        self.n_runs += len(rows)
        # sites=(): only logits are needed; sites=None: all states; else the listed sites
        return self.sys.run(X, patches, sites=sites)

    def _take(self, rows):
        if isinstance(self.X, np.ndarray):
            return self.X[rows]
        return [self.X[i] for i in rows]

    def node_values_from_states(self, node: Node, states: Dict[str, np.ndarray], logits=None) -> np.ndarray:
        if node.kind == "output":
            return logits[node.meta["behaviour"]].argmax(-1)
        m = node.primary
        return m.values(states[m.site])

    def partners(self, values: np.ndarray, m: int, differ: bool = True, rows=None, rng=None):
        """For each row, m partner indices with a different (or equal) value."""
        rng = rng or self.rng
        rows = np.arange(self.N) if rows is None else np.asarray(rows)
        out = np.full((len(rows), m), -1, dtype=np.int64)
        by_val = {v: np.where(values == v)[0] for v in np.unique(values)}
        allidx = np.arange(self.N)
        for i, r in enumerate(rows):
            v = values[r]
            if differ:
                cand = None
                # rejection sampling is fast when the value is not dominant
                for _ in range(4):
                    t = rng.integers(0, self.N, 4 * m)
                    t = t[values[t] != v]
                    if len(t) >= m:
                        cand = t[:m]
                        break
                if cand is None:
                    pool = allidx[values != v]
                    if len(pool) == 0:
                        continue
                    cand = rng.choice(pool, m)
                out[i] = cand
            else:
                pool = by_val[v]
                pool = pool[pool != r] if len(pool) > 1 else pool
                out[i] = rng.choice(pool, m)
        return out
