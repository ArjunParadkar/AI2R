"""P7, eigenvalue flow across training checkpoints (SPEC 2.10 / 6.8).

Per checkpoint cloud: the smallest normalized-Laplacian eigenvalues, the
spectral gap ``lam_2 - lam_1`` (indices 1-based as in SPEC 2.10: the trivial
eigenvalue is ``lam_1 = 0`` on a connected graph, so the gap is the first
non-trivial eigenvalue there), and the number of eigenvalues below a
threshold - an estimate of the cluster/component count.

Null it was validated against: a sequence of clouds going from one Gaussian
blob to two separated blobs, where the small-eigenvalue count must rise from
1 to 2 and the gap must collapse. A flow computed on clouds of *different*
sizes or with a different ``k`` per checkpoint is not comparable; the
eigenvalues of the normalized Laplacian depend on the graph, not only on the
geometry.

Failure boundary. ``crystallization_index`` is a descriptive changepoint on
the gap trace, not a test: it has no null distribution attached here, and if
the gap is flat or non-monotone the "first crossing" is whatever noise put
there first. SPEC 6.4/6.8 ask for the crystallization time to be compared
against a ground-truth event (ablation-defined onset, accuracy jump); this
function supplies only the left-hand side of that comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np

from .laplacian import laplacian, smallest_eigs

SMALL_EIG_THRESHOLD = 0.1


@dataclass
class EigenflowResult:
    """Laplacian spectrum per checkpoint (SPEC 6.8)."""
    eigenvalues: np.ndarray     # (n_checkpoints, n_eigs) ascending
    gap: np.ndarray             # (n_checkpoints,) lam_2 - lam_1
    n_small: np.ndarray         # (n_checkpoints,) count of eigenvalues < threshold
    threshold: float
    k: int
    n_eigs: int

    @property
    def n_checkpoints(self) -> int:
        return int(self.eigenvalues.shape[0])

    def crystallization_index(self, threshold: float = 0.5) -> int:
        """First checkpoint index whose spectral gap exceeds ``threshold`` x the final gap.

        With the default ``threshold=0.5`` this is "the first checkpoint where
        the gap has reached half its final value" (SPEC 6.8's "spectral gap
        opens"). ``threshold`` is a *fraction of the final gap*, not an
        absolute eigenvalue. Returns ``-1`` if the final gap is <= 0 or is
        never exceeded (e.g. the gap shrinks over training, which is what a
        cloud splitting into components does - use ``n_small`` there instead).
        """
        g = np.asarray(self.gap, dtype=np.float64)
        if g.size == 0 or not np.isfinite(g[-1]) or g[-1] <= 0:
            return -1
        hit = np.flatnonzero(g >= threshold * g[-1])
        return int(hit[0]) if hit.size else -1

    def to_dict(self) -> Dict[str, object]:
        return {"n_checkpoints": self.n_checkpoints, "n_eigs": int(self.n_eigs), "k": int(self.k),
                "threshold": float(self.threshold),
                "eigenvalues": np.asarray(self.eigenvalues).tolist(),
                "gap": np.asarray(self.gap).tolist(),
                "n_small": np.asarray(self.n_small).astype(int).tolist(),
                "crystallization_index": self.crystallization_index()}


def eigenflow(checkpoint_clouds: Sequence[np.ndarray], k: int = 15, n_eigs: int = 10,
              threshold: float = SMALL_EIG_THRESHOLD, eps: Optional[float] = None) -> EigenflowResult:
    """Laplacian eigenvalue flow over a sequence of activation clouds (SPEC 6.8).

    ``checkpoint_clouds`` is an ordered sequence of (n_points, n_dims) arrays -
    one per training checkpoint, same points in the same order if you want the
    trace to be about the representation rather than about the sample.
    Deterministic: no randomness anywhere in this path.
    """
    clouds = list(checkpoint_clouds)
    if not clouds:
        raise ValueError("need at least one checkpoint cloud")
    rows: List[np.ndarray] = []
    for A in clouds:
        L, _ = laplacian(np.asarray(A, dtype=np.float64), k=k, eps=eps)
        lam, _ = smallest_eigs(L, min(n_eigs, L.shape[0]))
        rows.append(np.sort(lam))
    m = min(len(r) for r in rows)
    E = np.stack([r[:m] for r in rows])
    gap = E[:, 1] - E[:, 0] if m >= 2 else np.zeros(E.shape[0])
    n_small = (E < threshold).sum(axis=1)
    return EigenflowResult(eigenvalues=E, gap=gap, n_small=n_small, threshold=float(threshold),
                           k=int(k), n_eigs=int(m))
