"""whitebox.memorization_signal — is this model memorising or generalising?

Built on the most robust WHITEBOX result (AUDIT.md bin 1): the topology of a
model's penultimate representation on a probe set collapses when the model
is fitting noise instead of structure (ResNet-18/CIFAR-10 ~1000x TC
separation; GPT-2 124M ~2900x). Phase 1 showed raw TC is dominated by
activation *spread*, so the signal uses scale-free features:

  tc_ratio   TC(H1) / mean TC(H1) of feature-shuffled copies of the same cloud
             (joint structure beyond the marginals; Phase 1 finding)
  pe1        persistent entropy of H1 (Rucco et al. 2016), scale-free
  tc_norm    TC(H1) / median pairwise distance^2, scale-free total persistence
  b1         beta_1 at the median bar midpoint (count; fixed probe size only)
  tc0_norm   TC(H0) / median pairwise distance^2  (cluster structure)

These are mapped to a calibrated probability by a logistic model fitted on
the Phase 2 battery. **The calibration is per architecture family** because
the battery showed the sign of every feature FLIPS between a CNN classifier
and a transformer LM (cross-family AUC = 0.0; benchmarks/PHASE2.md): ResNet
memorisers have a more entropic, less clustered penultimate cloud than
generalisers; GPT-2 memorisers have a less entropic, more collapsed one. So
``family`` is a required argument and the signal refuses to guess:

    "cnn_classifier"   penultimate pooled features of a conv classifier
                       (calibrated on ResNet-18/CIFAR-10, 21 models, 3 seeds)
    "lm_transformer"   mask-mean final hidden state of a causal LM
                       (calibrated on GPT-2 124M, 10 models + pretrained ref;
                       weaker ground truth: all fine-tunes overfit)

Calibrations live in ``calibration/memorization_<family>.json``.

Gauge (v2, benchmarks/PHASE6.md): the raw features are computed in the
network's native unit coordinates. E2 showed that TC of a raw ReLU activation
cloud is not invariant under the per-unit rescalings a ReLU layer admits;
``standardize=True`` z-scores each unit over the probe first (the E2 repair,
``whitebox.invariants.standardize_units``), which makes the features exactly
gauge-invariant. The Phase 6 benchmark measures what that costs the signal;
the default follows that result (see PHASE6.md and ``calib_path``).

Uncertainty: features are recomputed on ``n_boot`` probe subsamples of size
``subsample``; the reported interval is the subsample score spread
re-centred on the full-probe score (features are mildly size-dependent, the
shift is reported in PHASE2.md).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json, os
from typing import Dict, Optional, Sequence
import numpy as np

from ..core.reduce import fit_reducer
from ..core.persistence import compute_diagrams, pairwise_distances
from ..core.bootstrap import subsample_indices
from ..metrics.tc import topological_complexity, persistent_entropy
from ..metrics.betti import betti_at_scale, median_scale

CALIB_DIR = os.path.join(os.path.dirname(__file__), "calibration")
FAMILIES = ("cnn_classifier", "lm_transformer")


def calib_path(family: str, standardize: bool = False) -> str:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}; got {family!r}. The signal is calibrated per architecture family "
                         "because the feature sign flips between families (see benchmarks/PHASE2.md).")
    p = os.path.join(CALIB_DIR, f"memorization_{family}{'_std' if standardize else ''}.json")
    if standardize and not os.path.exists(p):
        raise ValueError(f"no calibration for family={family!r} with standardize=True (benchmarks/PHASE6.md: standardised "
                         "features were not calibrated for this family). Use standardize=False or fit your own calibration.")
    return p
FEATURE_NAMES = ["tc_ratio", "pe1", "tc_norm", "b1", "tc0_norm", "n_bars", "tc", "tc_shuffled", "med_dist"]


def memorization_features(acts: np.ndarray, n_components: int = 30, n_shuffle: int = 5, seed: int = 0,
                          thresh_percentile: float = 95.0, backend: str = "auto", distance_device: str = "cpu",
                          standardize: bool = False) -> Dict[str, float]:
    """Scale-free topological features of ONE activation cloud (n, d).
    The reducer is fitted on this cloud (no training history needed).
    ``standardize=True`` z-scores every unit over the probe before PCA (dead units
    dropped): the gauge-invariant variant (E2 repair, whitebox.invariants)."""
    acts = np.asarray(acts, dtype=np.float64)
    if standardize:
        from ..invariants.metrics import standardize_units
        acts = standardize_units(acts)
    r = fit_reducer(acts, n_components); Z = r.transform(acts)
    kw = dict(thresh_percentile=thresh_percentile, backend=backend, distance_device=distance_device)
    D = compute_diagrams(Z, None, **kw)
    Dm = pairwise_distances(Z, distance_device); med = float(np.median(Dm[np.triu_indices(len(Z), 1)]))
    tc = topological_complexity(D[1]); tc0 = topological_complexity(D[0])
    rng = np.random.default_rng(seed); shuf = []
    for _ in range(n_shuffle):
        Zs = np.stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])], 1)
        shuf.append(topological_complexity(compute_diagrams(Zs, None, **kw)[1]))
    tcs = float(np.mean(shuf))
    return dict(tc_ratio=tc / max(tcs, 1e-12), pe1=persistent_entropy(D[1]), tc_norm=tc / max(med, 1e-12) ** 2,
                b1=float(betti_at_scale(D[1], median_scale(D[1]))), tc0_norm=tc0 / max(med, 1e-12) ** 2,
                n_bars=float(len(D[1])), tc=tc, tc_shuffled=tcs, med_dist=med)


@dataclass
class Calibration:
    features: list
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    log_features: list
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, family: str = None, path: str = None, standardize: bool = False) -> "Calibration":
        path = path or calib_path(family, standardize)
        j = json.load(open(path))
        return cls(j["features"], np.array(j["mean"]), np.array(j["std"]), np.array(j["coef"]), float(j["intercept"]),
                   j.get("log_features", []), j.get("meta", {}))

    def save(self, path: str) -> None:
        json.dump(dict(features=self.features, mean=self.mean.tolist(), std=self.std.tolist(), coef=self.coef.tolist(),
                       intercept=self.intercept, log_features=self.log_features, meta=self.meta), open(path, "w"), indent=1)

    def transform(self, f: Dict[str, float]) -> np.ndarray:
        x = np.array([np.log(max(f[k], 1e-12)) if k in self.log_features else f[k] for k in self.features], dtype=float)
        return (x - self.mean) / self.std

    def score(self, f: Dict[str, float]) -> float:
        z = float(self.transform(f) @ self.coef + self.intercept)
        return float(1.0 / (1.0 + np.exp(-z)))


@dataclass
class MemorizationSignal:
    score: float          # P(memorising) under the battery calibration, in [0, 1]
    ci_low: float
    ci_high: float
    features: Dict[str, float]
    n_probe: int
    n_boot: int
    subsample: int
    calibration: dict

    @property
    def threshold(self) -> float:
        """Decision threshold = midpoint between the battery's highest generalising
        and lowest memorising score (0.5 if the classes overlapped)."""
        return float(self.calibration.get("threshold", 0.5))

    @property
    def verdict(self) -> str:
        t = self.threshold
        if self.ci_low > t: return "memorizing"
        if self.ci_high < t: return "generalizing"
        return "uncertain"

    @property
    def battery_position(self) -> str:
        """Where the score sits relative to the calibration battery."""
        b = self.calibration.get("battery_scores")
        if not b: return "n/a"
        g, m = b["generalizing"], b["memorizing"]
        if self.score <= max(g): return f"within the battery's generalising range [{min(g):.2f}, {max(g):.2f}]"
        if self.score >= min(m): return f"within the battery's memorising range [{min(m):.2f}, {max(m):.2f}]"
        return f"between the classes ({max(g):.2f} < score < {min(m):.2f})"

    def __repr__(self):
        return (f"MemorizationSignal(score={self.score:.3f} CI95=[{self.ci_low:.3f},{self.ci_high:.3f}] threshold={self.threshold:.2f} -> {self.verdict}; "
                f"{self.battery_position}; tc_ratio={self.features['tc_ratio']:.2f} pe1={self.features['pe1']:.2f} n={self.n_probe})")


def memorization_signal(model=None, probe=None, layer: Optional[str] = None, activations: Optional[np.ndarray] = None,
                        family: Optional[str] = None, n_boot: int = 20, subsample: Optional[int] = None, n_components: int = 30, seed: int = 0,
                        calibration: Optional[Calibration] = None, distance_device: str = "cpu", standardize: bool = False,
                        **kw) -> MemorizationSignal:
    """Calibrated memorisation score for a model on a probe set.

    Either pass ``model``, ``probe`` (iterable of batches) and ``layer`` (the
    penultimate-feature module name, e.g. ``"pool"`` / ``"avgpool"``; for HF
    models pass ``activations`` from ``whitebox.activations.hf_hidden_states``),
    or pass ``activations`` directly as an (n_probe, d) array.

    ``family`` selects the calibration ("cnn_classifier" or "lm_transformer");
    it is required unless a ``calibration`` object is passed.

    ``standardize`` (default False = the calibrated, coordinate-dependent v1
    signal): per-unit z-scoring of the probe cloud before the topology. This is
    the gauge-invariant form; benchmarks/PHASE6.md reports what it costs. It
    needs its own calibration file (``memorization_<family>_std.json``), which
    ships only for families where the benchmark kept the AUC.

    Returns a MemorizationSignal: ``score`` is P(memorising) under the Phase 2
    battery calibration for that family, with a 95 % interval from probe
    subsampling.
    """
    if activations is None:
        if model is None or probe is None or layer is None:
            raise ValueError("pass activations=..., or model=, probe=, layer=")
        from ..activations import ActivationExtractor
        ex = ActivationExtractor(model, [layer]); activations = ex(probe)[layer]; ex.remove()
    A = np.asarray(activations, dtype=np.float64); n = len(A)
    cal = calibration or Calibration.load(family, standardize=standardize)
    fkw = dict(n_components=n_components, seed=seed, distance_device=distance_device, standardize=standardize, **kw)
    f = memorization_features(A, **fkw); point = cal.score(f)
    m = subsample or max(100, n // 2)
    if n_boot > 0 and n > 20:
        idx = subsample_indices(n, m, n_boot, seed)
        s = np.array([cal.score(memorization_features(A[i], **fkw)) for i in idx])
        lo, hi = np.percentile(s, [2.5, 97.5]); c = float(np.median(s))
        ci = (float(np.clip(point + (lo - c), 0, 1)), float(np.clip(point + (hi - c), 0, 1)))
    else:
        ci = (point, point)
    return MemorizationSignal(point, ci[0], ci[1], f, n, n_boot, m, cal.meta)
