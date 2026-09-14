"""Certified extractors: path products (2.5), Fourier irreps (2.6), circular coordinates (2.7).

Three extraction primitives from FINAL_SPEC Section 4 (``invariants/``). Each
returns a quantity that is *constant on the gauge orbit* of the network, so it
is a statement about the function rather than about the parameterisation:

* :func:`path_product` -- ``f(x) = A(x) x + c(x)`` with
  ``A(x) = W_L D_{L-1}(x) ... D_1(x) W_1``. Hidden-unit permutations cancel
  (``P^T P = I``) and positive scalings telescope (``S^{-1} ... S``), so ``A``
  and ``c`` are *exactly* invariant under G = perms x positive scalings.
  Null validated against: the gauge orbit itself (``apply_gauge`` of a random
  group element must leave A, c unchanged to float precision --
  ``tests/test_extractors.py``).
  Failure boundary: ReLU (or any positively-homogeneous piecewise-linear)
  MLPs only. GELU/LayerNorm/attention have a different group (2.3) and the
  linearisation ``A(x)`` is then only local, not exact.

* :func:`fourier_irreps` -- DFT of a token embedding along the token axis.
  Column-space rotations of the embedding (a gauge freedom of the residual
  stream) act on the right and leave ``||E_hat_k||`` unchanged, so the
  frequency set K and the concentration kappa are invariant; the per-frequency
  cos/sin *blocks* are covariant (they rotate with the basis) and are returned
  flagged as such.
  Null validated against: a random orthogonal rotation of the embedding
  columns (K, kappa, power spectrum must be bit-for-bit stable) and a random
  Gaussian embedding (kappa ~ 2/(p-1) per frequency, no K survives tau).
  Failure boundary: cyclic group Z_p tasks with the token axis as the first
  ``p`` rows. Not a claim about *what the network does* with those frequencies.

* :func:`circular_coordinate` -- de Silva-Morozov-Vejdemo-Johansson circular
  coordinate of the most persistent H1 class of a point cloud.
  Null validated against: a Gaussian blob (no dominant bar; lifetime ratio
  ~ 1) and a noisy circle (coordinate must track the true angle).
  Failure boundary: one dominant H1 class. With two comparable bars the
  returned lifetime ratio approaches 1 and the coordinate is not identified;
  the ratio is returned precisely so the caller can refuse.

Per-unit scalings are *not* isometries of an activation cloud, so activity must
be normalised per unit before :func:`circular_coordinate`; that normalisation
is itself a gauge fixing and the caller must record it (2.7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .model import as_relu_mlp

__all__ = ["path_product", "path_products", "top_contributions", "TopContributions",
           "fourier_irreps", "IrrepResult", "circular_coordinate", "as_grok_model", "GrokModel"]


# ---------------------------------------------------------------- 2.5 path products
def path_products(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`path_product`: ``X`` (N, n_in) -> ``A`` (N, n_out, n_in), ``c`` (N, n_out).

    ``f(X[i]) == A[i] @ X[i] + c[i]`` exactly (to float rounding). Gauge-invariant
    under hidden-unit permutations and positive scalings (SPEC 2.5).
    """
    m = as_relu_mlp(model)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    masks, _, _ = m.forward(X)
    n = X.shape[0]
    # layer 1
    A = masks[0][:, :, None] * m.Ws[0].T[None, :, :]        # (N, w1, n_in)
    c = masks[0] * m.bs[0][None, :]                          # (N, w1)
    for l in range(1, m.depth):
        A = masks[l][:, :, None] * np.einsum("ij,njk->nik", m.Ws[l].T, A)
        c = masks[l] * (c @ m.Ws[l] + m.bs[l][None, :])
    A = np.einsum("ij,njk->nik", m.Ws[-1].T, A)
    c = c @ m.Ws[-1] + m.bs[-1][None, :]
    return A, c


def path_product(model, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact gauge-invariant attribution at a single input (SPEC 2.5).

    Returns ``(A, c)`` with ``A`` (n_out, n_in) and ``c`` (n_out,) such that
    ``f(x) = A @ x + c``. ``A[o, i]`` is the certified contribution of input
    coordinate ``i`` to output ``o`` per unit of ``x[i]`` -- the answer to P2
    for this decision, valid on the whole activation region containing ``x``.
    """
    A, c = path_products(model, np.asarray(x, dtype=np.float64)[None, :])
    return A[0], c[0]


@dataclass
class TopContributions:
    """Top-k certified input attributions of one decision (P2)."""
    indices: np.ndarray        # (n_out, k) input coordinates, |contribution| descending
    values: np.ndarray         # (n_out, k) the contributions A[o, i] * x[i]
    total: np.ndarray          # (n_out,) sum_i A[o, i] x[i]
    bias: np.ndarray           # (n_out,) c
    fraction: np.ndarray       # (n_out,) |top-k| / sum_i |A[o,i] x[i]|

    def to_dict(self) -> dict:
        return {"indices": self.indices.tolist(), "values": self.values.tolist(),
                "total": self.total.tolist(), "bias": self.bias.tolist(),
                "fraction": self.fraction.tolist()}


def top_contributions(A: np.ndarray, x: np.ndarray, k: int = 5, c: Optional[np.ndarray] = None) -> TopContributions:
    """The k largest-magnitude entries of the certified attribution ``A[o, :] * x`` per output."""
    A = np.asarray(A, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).ravel()
    if A.ndim != 2 or A.shape[1] != x.size:
        raise ValueError(f"A must be (n_out, n_in) matching x ({x.size},), got {A.shape}")
    k = int(min(k, x.size))
    contrib = A * x[None, :]
    order = np.argsort(-np.abs(contrib), axis=1)[:, :k]
    vals = np.take_along_axis(contrib, order, axis=1)
    denom = np.abs(contrib).sum(1)
    denom = np.where(denom <= 0, 1.0, denom)
    bias = np.zeros(A.shape[0]) if c is None else np.asarray(c, dtype=np.float64).ravel()
    return TopContributions(order, vals, contrib.sum(1), bias, np.abs(vals).sum(1) / denom)


# ---------------------------------------------------------------- 2.6 Fourier irreps
@dataclass
class IrrepResult:
    """Certified irrep inventory of a Z_p token embedding (SPEC 2.6).

    ``K``/``kappa``/``power`` are invariant under any orthogonal rotation of the
    embedding columns; ``weights`` (the cos/sin block coefficients per column)
    are *covariant* -- they rotate with the basis -- and are marked as such in
    ``covariant``. Downstream reconstruction may use only the invariant
    per-frequency ``amplitude``.
    """
    p: int
    tau: float
    K: Tuple[int, ...]
    kappa: float
    freqs: np.ndarray                       # (m,) k = 1 .. (p-1)//2
    power: np.ndarray                       # (m,) normalised folded power, sums to 1
    amplitude: Dict[int, float]             # k -> ||E_hat_k|| (folded), invariant
    weights: Dict[int, np.ndarray]          # k -> (2, d) [cos row; sin row] per column, covariant
    include_dc: bool = False
    covariant: Tuple[str, ...] = ("weights",)

    def omega(self, k: int) -> float:
        return 2.0 * np.pi * k / self.p

    def to_dict(self) -> dict:
        return {"p": self.p, "tau": self.tau, "K": list(self.K), "kappa": self.kappa,
                "freqs": self.freqs.tolist(), "power": self.power.tolist(),
                "amplitude": {int(k): float(v) for k, v in self.amplitude.items()},
                "include_dc": self.include_dc, "covariant": list(self.covariant)}


def fourier_irreps(embedding, p: int, tau: float = 0.05, include_dc: bool = False) -> IrrepResult:
    """Certified frequency inventory of an embedding over Z_p (SPEC 2.6).

    ``embedding`` is (>= p, d); only the first ``p`` rows (the Z_p tokens) are
    used, so a ``p+1``-row table with an '=' row may be passed unchanged.
    Power is folded over the conjugate pair (k, p-k), normalised by the total
    over *non-zero* frequencies, and ``K = {k : power_k > tau}``,
    ``kappa = sum_{k in K} power_k``.

    Deviation from the literal formula in 2.6: the DC term k=0 is excluded from
    the denominator by default (``include_dc=False``). k=0 is the trivial irrep
    -- a token-independent mean offset of the residual stream, not a rotation
    block -- and leaving it in lets a large constant offset mask K. Pass
    ``include_dc=True`` for the literal normalisation.

    Invariant under: right-multiplication of the embedding by any orthogonal
    matrix (checked in ``tests/test_extractors.py``), and under every hidden-unit
    gauge element (those do not touch the token axis).
    Failure boundary: assumes the token axis is rows ``0..p-1`` of a real matrix,
    and tau must stay above the flat-spectrum share 1/((p-1)//2), or the
    diagnostic cannot fail. At p = 113 there are 56 folded frequencies, so a flat
    (initialisation) spectrum gives every frequency about 0.018: at tau = 0.01 it
    would return K = all 56 and kappa = 1. Hence the default tau = 0.05, at which
    a flat spectrum returns K = {} and kappa = 0
    (experiments/O_opener/AMENDMENTS.md #0).
    """
    E = np.asarray(embedding, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(f"embedding must be 2-D (n_tokens, d), got shape {E.shape}")
    if E.shape[0] < p:
        raise ValueError(f"embedding has {E.shape[0]} rows, need at least p={p}")
    E = E[:p]
    d = E.shape[1]
    F = np.fft.fft(E, axis=0)                         # F[k, j] = sum_x E[x, j] e^{-2 pi i k x / p}
    pw = (np.abs(F) ** 2).sum(1)                      # (p,) power per raw frequency
    m = (p - 1) // 2
    freqs = np.arange(1, m + 1)
    folded = pw[1:m + 1] + pw[p - freqs]              # fold k with p-k (conjugates)
    total = folded.sum() + (pw[0] if include_dc else 0.0)
    if p % 2 == 0:                                    # Nyquist frequency, self-conjugate
        total += pw[p // 2]
    if total <= 0:
        raise ValueError("embedding has zero power; cannot extract irreps")
    power = folded / total
    K = tuple(int(k) for k in freqs[power > tau])
    kappa = float(power[power > tau].sum())
    amp = {int(k): float(np.sqrt(folded[k - 1])) for k in K}
    weights = {}
    for k in K:
        blk = np.empty((2, d))
        blk[0] = 2.0 / p * np.real(F[k])              # cos coefficients
        blk[1] = -2.0 / p * np.imag(F[k])             # sin coefficients
        weights[int(k)] = blk
    return IrrepResult(int(p), float(tau), K, kappa, freqs, power, amp, weights, bool(include_dc))


# ---------------------------------------------------------------- 2.7 circular coordinate
def _cocycle_backend():
    try:
        import ripser  # noqa: F401
        return "ripser"
    except Exception:
        pass
    try:
        import dreimac  # noqa: F401
        return "dreimac"
    except Exception:
        pass
    return None


def circular_coordinate(cloud: np.ndarray, maxdim: int = 1, prime: int = 47,
                        thresh: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Circular coordinate of the most persistent H1 class (de Silva-Morozov-VJ, SPEC 2.7).

    Persistent cohomology with Z_p coefficients gives a cocycle ``alpha`` for the
    dominant bar; it is lifted to a real 1-cochain on the 1-skeleton at the
    birth-death midpoint scale by least squares (``min ||d f - alpha||``), and
    ``theta = 2 pi (f mod 1)``.

    Returns ``(theta in [0, 2pi), lifetime_ratio)`` where the ratio is
    ``top bar lifetime / second bar lifetime`` (``inf`` if there is only one bar).
    A ratio near 1 means the class is *not* identified -- refuse the coordinate.

    Null validated against: a Gaussian blob (ratio ~ 1, coordinate meaningless)
    and a noisy circle (circular correlation with the true angle > 0.95).
    Failure boundary: needs a persistent-cohomology backend (``ripser`` with
    ``do_cocycles=True``, else ``dreimac``); raises NotImplementedError with the
    reason if neither is importable. Per-unit activation scaling changes the
    metric, so normalise the cloud first (that is a recorded gauge fixing).
    """
    X = np.asarray(cloud, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"cloud must be (n_points, dim), got {X.shape}")
    backend = _cocycle_backend()
    if backend is None:
        raise NotImplementedError(
            "circular_coordinate needs a persistent-cohomology backend with cocycles: "
            "neither 'ripser' (do_cocycles=True) nor 'dreimac' is importable in this "
            "environment, and this library never installs packages itself.")
    if backend == "ripser":
        from ripser import ripser as _ripser
        kw = dict(maxdim=maxdim, coeff=prime, do_cocycles=True)
        if thresh is not None:
            kw["thresh"] = float(thresh)
        res = _ripser(X, **kw)
        dgm = np.asarray(res["dgms"][1])
        cocycles = res["cocycles"][1]
    else:                                              # dreimac path: same data, its own PH call
        from dreimac import CircularCoords
        cc = CircularCoords(X, n_landmarks=min(len(X), 200), prime=prime)
        dgm = np.asarray(cc.dgms_[1])
        theta = np.asarray(cc.get_coordinates(), dtype=np.float64) % (2 * np.pi)
        return theta, _lifetime_ratio(dgm)
    if len(dgm) == 0:
        raise ValueError("no H1 classes in this cloud: there is no circular coordinate to extract")
    life = dgm[:, 1] - dgm[:, 0]
    top = int(np.argmax(life))
    birth, death = float(dgm[top, 0]), float(dgm[top, 1])
    scale = 0.5 * (birth + death)
    coc = np.asarray(cocycles[top])
    # lift Z_prime values to {-(p-1)/2 .. (p-1)/2}
    vals = coc[:, 2].astype(np.float64)
    vals = np.where(vals > prime / 2.0, vals - prime, vals)
    edge_val = {(int(min(i, j)), int(max(i, j))): (v if i < j else -v)
                for (i, j, v) in zip(coc[:, 0], coc[:, 1], vals)}
    theta = _lift_cocycle(X, edge_val, scale)
    return theta, _lifetime_ratio(dgm)


def _lifetime_ratio(dgm: np.ndarray) -> float:
    life = np.sort(np.asarray(dgm)[:, 1] - np.asarray(dgm)[:, 0])[::-1]
    if len(life) < 2 or life[1] <= 0:
        return float("inf")
    return float(life[0] / life[1])


def _lift_cocycle(X: np.ndarray, edge_val: dict, scale: float) -> np.ndarray:
    """Least-squares lift of an integer 1-cocycle on the 1-skeleton at ``scale`` to f: V -> R."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    from scipy.spatial.distance import squareform, pdist
    D = squareform(pdist(X))
    iu, ju = np.triu_indices(len(X), 1)
    keep = D[iu, ju] <= scale
    ei, ej = iu[keep], ju[keep]
    if len(ei) == 0:
        raise ValueError("empty 1-skeleton at the birth-death midpoint scale")
    b = np.array([edge_val.get((int(a), int(c)), 0.0) for a, c in zip(ei, ej)])
    rows = np.repeat(np.arange(len(ei)), 2)
    cols = np.empty(2 * len(ei), dtype=int); cols[0::2] = ei; cols[1::2] = ej
    data = np.empty(2 * len(ei)); data[0::2] = -1.0; data[1::2] = 1.0
    Dmat = coo_matrix((data, (rows, cols)), shape=(len(ei), len(X))).tocsr()
    f = lsqr(Dmat, b, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    return (2.0 * np.pi * (f % 1.0)) % (2.0 * np.pi)


# ---------------------------------------------------------------- duck-typed grok model
class GrokModel:
    """Duck-typed view of a grokked modular-addition model.

    Exposes ``.embed`` (p+1, d), ``.unembed`` (d, p) and ``forward(tokens) -> logits``
    for token arrays of shape (N, 3) = [a, b, '='] with logits at the last position
    over the ``p`` classes. Nothing here assumes a particular internal architecture;
    the extractors only ever read ``.embed`` / ``.unembed`` and call ``forward``.
    """

    def __init__(self, embed: np.ndarray, unembed: np.ndarray, forward):
        self.embed = np.asarray(embed, dtype=np.float64)
        self.unembed = np.asarray(unembed, dtype=np.float64)
        self._forward = forward
        self.p = self.unembed.shape[1]

    def forward(self, tokens) -> np.ndarray:
        return np.asarray(self._forward(tokens), dtype=np.float64)

    def __call__(self, tokens) -> np.ndarray:
        return self.forward(tokens)


def as_grok_model(obj) -> GrokModel:
    """Adapter: accept the duck-typed interface, or a torch module with ``W_E``/``W_U``.

    Failure boundary: raises TypeError if the object exposes neither
    ``.embed``/``.unembed`` nor ``W_E``/``W_U``. Torch is imported lazily and only
    when a torch module is actually passed.
    """
    if isinstance(obj, GrokModel):
        return obj
    if hasattr(obj, "embed") and hasattr(obj, "unembed"):
        return GrokModel(np.asarray(obj.embed), np.asarray(obj.unembed),
                         getattr(obj, "forward", obj))
    if hasattr(obj, "W_E") and hasattr(obj, "W_U"):
        def _np(t):
            return t.detach().cpu().double().numpy() if hasattr(t, "detach") else np.asarray(t)

        W_E, W_U = _np(obj.W_E), _np(obj.W_U)

        def fwd(tokens):
            import torch
            with torch.no_grad():
                out = obj(torch.as_tensor(np.asarray(tokens)))
            out = out.detach().cpu().double().numpy()
            return out[:, -1, :] if out.ndim == 3 else out

        return GrokModel(W_E, W_U, fwd)
    raise TypeError(f"cannot interpret {type(obj).__name__} as a grok model: expected "
                    "`.embed` (p+1, d) + `.unembed` (d, p) + forward(tokens), or a torch "
                    "module with attributes W_E, W_U and a forward")
