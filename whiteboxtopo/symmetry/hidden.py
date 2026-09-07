"""Hidden-symmetry detector (handbook S1 ``symmetry.hidden``).

The *fiber dimension* of a network at its parameters is the corank of the
Jacobian of the realisation map theta -> (f_theta(x))_{x in inputs} (Grigsby,
Lindsey, Meyerhoff & Wu 2022: functional dimension). Every continuous symmetry
contributes one flat direction, so ``corank >= known_dim`` where ``known_dim``
counts the continuous generators of the known group G (for a ReLU MLP: one
scaling per hidden unit). The excess ``corank - known_dim`` is the number of
flat directions G does not explain. This module

1. computes the Jacobian (double precision, autograd), its rank and null space;
2. removes the span of the known generators (scalings) from the null space;
3. tries to *identify* what is left, in order: dead units (never active on the
   probe -> all incoming parameters and the outgoing weights are free),
   duplicate units (identical incoming (w, b) up to scale -> the two outgoing
   weight rows trade off), constant units (active on every probe input -> the
   ReLU is linear there, and the unit's incoming bias can move against the
   next layer's bias); anything else is reported as "unclassified".

Caveat stated up front: a flat *direction* is a first-order statement. It is
a genuine symmetry only if it integrates to a curve of function-preserving
parameters; dead/duplicate/constant units do, and are exact symmetries on the
probe set. Unclassified directions are reported, never assumed to be
symmetries.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import torch
from ..invariants.model import as_relu_mlp, ReluMLP


@dataclass
class HiddenSymmetryReport:
    n_params: int
    rank: int
    corank: int                     # fiber_dim
    known_dim: int                  # continuous generators of G accounted for
    excess: int                     # corank - known_dim (>= 0 up to numerical rank error)
    unexplained: int                # excess not identified by the heuristics
    identified: Dict[str, int] = field(default_factory=dict)   # {"dead": k, "duplicate": k, "constant": k}
    dead_units: List[tuple] = field(default_factory=list)
    duplicate_pairs: List[tuple] = field(default_factory=list)
    constant_units: List[tuple] = field(default_factory=list)
    singular_values: Optional[np.ndarray] = field(default=None, repr=False)
    null_directions: Optional[np.ndarray] = field(default=None, repr=False)   # (n_null, n_params), after removing G
    G_incomplete: bool = False

    def __str__(self):
        return (f"HiddenSymmetryReport(n_params={self.n_params}, rank={self.rank}, corank={self.corank}, "
                f"known_dim={self.known_dim}, excess={self.excess}, identified={self.identified}, "
                f"unexplained={self.unexplained}, G_incomplete={self.G_incomplete})")


def _param_layout(m: ReluMLP):
    """Offsets of W_l (row-major) and b_l inside the flat parameter vector [W0,...,Wd, b0,...,bd]."""
    off, cur = {}, 0
    for l, W in enumerate(m.Ws):
        off[("W", l)] = (cur, W.shape); cur += W.size
    for l, b in enumerate(m.bs):
        off[("b", l)] = (cur, b.shape); cur += b.size
    return off, cur


def scaling_generators(m: ReluMLP) -> np.ndarray:
    """Tangent vectors of the per-unit ReLU scalings at the identity: for hidden unit (l, j),
    d/da [a (w_in, b), w_out / a] at a=1 = (+w_in, +b, -w_out). Shape (sum widths, n_params)."""
    off, n = _param_layout(m)
    gens = []
    for l, w in enumerate(m.widths):
        for j in range(w):
            v = np.zeros(n)
            o, shp = off[("W", l)]; Wv = v[o:o + int(np.prod(shp))].reshape(shp); Wv[:, j] = m.Ws[l][:, j]
            o, shp = off[("b", l)]; v[o + j] = m.bs[l][j]
            o, shp = off[("W", l + 1)]; Wn = v[o:o + int(np.prod(shp))].reshape(shp); Wn[j, :] = -m.Ws[l + 1][j, :]
            gens.append(v)
    return np.stack(gens) if gens else np.zeros((0, n))


def realisation_jacobian(m: ReluMLP, X: np.ndarray) -> np.ndarray:
    """J[(x, o), theta] = d f_o(x) / d theta, double precision, shape (N*n_out, n_params)."""
    Xt = torch.tensor(np.asarray(X, dtype=np.float64))
    Ws = [torch.tensor(W, dtype=torch.float64, requires_grad=True) for W in m.Ws]
    bs = [torch.tensor(b, dtype=torch.float64, requires_grad=True) for b in m.bs]

    def f(*params):
        Wp, bp = params[:len(Ws)], params[len(Ws):]
        h = Xt
        for W, b in zip(Wp[:-1], bp[:-1]):
            h = torch.relu(h @ W + b)
        return (h @ Wp[-1] + bp[-1]).reshape(-1)

    J = torch.autograd.functional.jacobian(f, tuple(Ws + bs))
    return torch.cat([j.reshape(j.shape[0], -1) for j in J], dim=1).numpy()


def probe_inputs(m: ReluMLP, rng, n: Optional[int] = None, box=(-1.0, 2.0), X_base=None) -> np.ndarray:
    """Probe set: ``X_base`` (e.g. the truth table) plus uniform samples in ``box`` so that
    the Jacobian has more rows than parameters (m_factor 4 as in E1)."""
    n = n or 4 * sum(W.size + b.size for W, b in zip(m.Ws, m.bs))
    X = rng.uniform(box[0], box[1], size=(n, m.n_in))
    if X_base is not None:
        X = np.vstack([np.asarray(X_base, dtype=np.float64), X])
    return X


def hidden(model, inputs=None, rel_tol: float = 1e-8, seed: int = 0, keep_directions: bool = True,
           dup_tol: float = 1e-6) -> HiddenSymmetryReport:
    """Fiber dimension vs the known group, with identification of the excess (see module doc).

    ``inputs``: probe points; if None (or if fewer rows than parameters) random probes in
    [-1, 2]^n_in are appended so the rank is not limited by the probe count. ``rel_tol`` is the
    singular-value cut-off relative to the largest singular value."""
    m = as_relu_mlp(model)
    rng = np.random.default_rng(seed)
    n_params = sum(W.size + b.size for W, b in zip(m.Ws, m.bs))
    X = np.asarray(inputs, dtype=np.float64) if inputs is not None else None
    if X is None or X.shape[0] * m.n_out < 4 * n_params:
        X = probe_inputs(m, rng, X_base=X)
    J = realisation_jacobian(m, X)
    U, s, Vt = np.linalg.svd(J, full_matrices=True)
    rank = int((s > rel_tol * s[0]).sum()) if s.size and s[0] > 0 else 0
    corank = n_params - rank
    null = Vt[rank:]                                           # (corank, n_params) orthonormal
    # remove the span of the known scaling generators
    G = scaling_generators(m)
    known_dim = 0
    if G.shape[0]:
        # rank of the generators themselves (dead/duplicate units can make some coincide)
        Qg, Rg = np.linalg.qr(G.T, mode="reduced")
        gs = np.linalg.svd(G, compute_uv=False)
        known_dim = int((gs > 1e-10 * gs[0]).sum())
        # project null space onto the complement of span(G)
        Ug = np.linalg.svd(G.T, full_matrices=False)[0][:, :known_dim]
        resid = null - (null @ Ug) @ Ug.T
        rs = np.linalg.svd(resid, compute_uv=False) if resid.size else np.zeros(0)
        n_extra = int((rs > 1e-8).sum())
        extra_dirs = np.linalg.svd(resid, full_matrices=False)[2][:n_extra] if n_extra else np.zeros((0, n_params))
    else:
        n_extra, extra_dirs = corank, null
    excess = corank - known_dim
    # ---- identification heuristics on the probe set
    h = X; dead, const, dups = [], [], []
    for l, (W, b) in enumerate(zip(m.Ws[:-1], m.bs[:-1])):
        pre = h @ W + b; act = pre > 0
        for j in range(W.shape[1]):
            if not act[:, j].any():
                dead.append((l, j))
            elif act[:, j].all():
                const.append((l, j))
        # duplicates: incoming (w, b) proportional (positive factor)
        V = np.vstack([W, b[None, :]]); nrm = np.linalg.norm(V, axis=0); nrm[nrm == 0] = 1
        Vn = V / nrm
        for i in range(W.shape[1]):
            for j in range(i + 1, W.shape[1]):
                if (l, i) in dead or (l, j) in dead:
                    continue
                if np.linalg.norm(Vn[:, i] - Vn[:, j]) < dup_tol:
                    dups.append((l, i, j))
        h = np.maximum(pre, 0.0)
    # count flat directions each mechanism contributes beyond the scaling generator
    n_out_next = lambda l: m.Ws[l + 1].shape[1]
    dead_dirs = sum(m.Ws[l].shape[0] + 1 + n_out_next(l) - 1 for (l, j) in dead)       # incoming + bias + outgoing, minus the scaling already counted
    # a duplicated pair shares its activation pattern, so f depends only on w_out_i (w_i,b_i) + w_out_j (w_j,b_j):
    # fan_in+1 incoming + n_out outgoing trade directions, minus the one already inside span(scaling generators)
    dup_dirs = sum(m.Ws[l].shape[0] + 1 + n_out_next(l) - 1 for (l, i, j) in dups)
    const_dirs = len(const)                                                              # bias can move against next-layer bias (one direction each, first-order)
    identified = {"dead": int(dead_dirs), "duplicate": int(dup_dirs), "constant": int(const_dirs)}
    unexplained = max(int(excess) - sum(identified.values()), 0)
    return HiddenSymmetryReport(
        n_params=int(n_params), rank=rank, corank=int(corank), known_dim=int(known_dim), excess=int(excess),
        unexplained=unexplained, identified=identified, dead_units=dead, duplicate_pairs=dups, constant_units=const,
        singular_values=s if keep_directions else None,
        null_directions=extra_dirs if keep_directions else None,
        G_incomplete=bool(excess > 0))
