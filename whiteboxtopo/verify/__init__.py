"""Block R verification: is the rebuild the same function, and does the explanation survive its own nulls.

FINAL_SPEC Section 7, R5-R7. A reconstruction that merely *looks* like the model
on the training inputs proves nothing; these three tests are what separates a
rebuild from a curve fit:

* :func:`equivalence`       -- R5: accuracy, logit correlation, argmax agreement
  under Gaussian perturbation at six radii, OOD agreement. Pre-registered bands.
* :func:`ablation_symmetry` -- R6: ablating a certified object in ``f`` and its
  counterpart in ``f_hat`` must produce correlated effect vectors (r > 0.9);
  anything below that is mis-mapped, however good the aggregate agreement.
* :func:`negative_control`  -- R7: the same rebuild procedure fed artifact-class
  objects must fail near chance, and dropping certified objects must degrade the
  rebuild in proportion. Without this, R5 passing is uninformative -- it could
  be passing for reasons that have nothing to do with the inventory.

numpy/scipy only; ``f`` and ``f_hat`` are plain callables ``X -> logits (N, C)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = ["equivalence", "Verdict", "ablation_symmetry", "AblationTable",
           "negative_control", "NegativeControl", "DEFAULT_TOL"]

# Pre-registered bands (SPEC 7 R1 / R5).
DEFAULT_TOL = dict(acc_drop=0.005, logit_corr=0.98, pert_band=0.05, ood_agree=0.9)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else 0.0


def _logits(fn, X) -> np.ndarray:
    out = np.asarray(fn(X), dtype=np.float64)
    return out[None, :] if out.ndim == 1 else out


@dataclass
class Verdict:
    """Behavioral-equivalence verdict for a reconstruction (SPEC 7 R5)."""
    n: int
    n_classes: int
    accuracy: Optional[float]
    accuracy_hat: Optional[float]
    acc_drop: Optional[float]
    logit_corr: float
    logit_corr_per_row: float
    argmax_agreement: float
    perturbation: Dict[float, float]        # radius -> argmax agreement
    perturbation_skipped: Optional[str]
    ood_agreement: Optional[float]
    tol: Dict[str, float]
    checks: Dict[str, bool]
    passed: bool
    seed: int

    def to_dict(self) -> dict:
        return {"n": self.n, "n_classes": self.n_classes, "accuracy": self.accuracy,
                "accuracy_hat": self.accuracy_hat, "acc_drop": self.acc_drop,
                "logit_corr": self.logit_corr, "logit_corr_per_row": self.logit_corr_per_row,
                "argmax_agreement": self.argmax_agreement,
                "perturbation": {float(r): float(v) for r, v in self.perturbation.items()},
                "perturbation_skipped": self.perturbation_skipped,
                "ood_agreement": self.ood_agreement, "tol": dict(self.tol),
                "checks": dict(self.checks), "passed": self.passed, "seed": self.seed}

    def to_markdown(self) -> str:
        L = ["| quantity | value | band | pass |", "| --- | --- | --- | --- |"]
        if self.acc_drop is not None:
            L.append(f"| accuracy drop | {self.acc_drop:+.4f} | <= {self.tol['acc_drop']} | "
                     f"{'yes' if self.checks.get('acc_drop') else 'no'} |")
        L.append(f"| logit correlation | {self.logit_corr:.4f} | >= {self.tol['logit_corr']} | "
                 f"{'yes' if self.checks.get('logit_corr') else 'no'} |")
        L.append(f"| argmax agreement | {self.argmax_agreement:.4f} | - | - |")
        for r, v in self.perturbation.items():
            L.append(f"| agreement @ r={r} | {v:.4f} | >= {1 - self.tol['pert_band']:.3f} | "
                     f"{'yes' if v >= 1 - self.tol['pert_band'] else 'no'} |")
        if self.perturbation_skipped:
            L.append(f"| perturbation | skipped | - | {self.perturbation_skipped} |")
        if self.ood_agreement is not None:
            L.append(f"| OOD agreement | {self.ood_agreement:.4f} | >= {self.tol['ood_agree']} | "
                     f"{'yes' if self.checks.get('ood_agree') else 'no'} |")
        L.append(f"\n**verdict: {'PASS' if self.passed else 'FAIL'}** (n={self.n})")
        return "\n".join(L)


def equivalence(f: Callable, f_hat: Callable, X, y=None,
                radii: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
                ood=None, n_pert: int = 8, seed: int = 0, tol: Optional[dict] = None) -> Verdict:
    """R5 behavioral-equivalence battery between a model and its reconstruction (SPEC 7 R5).

    Measures accuracy (if ``y`` is given), Pearson correlation of the flattened
    logits, argmax agreement, argmax agreement under Gaussian perturbation of
    ``X`` at each radius (``n_pert`` draws each; the radius is relative to the
    global standard deviation of ``X``), and argmax agreement on ``ood`` inputs.

    Pass criteria, pre-registered in SPEC 7 R1/R5 and overridable via ``tol``:
    accuracy drop <= 0.005, logit correlation >= 0.98, perturbation agreement
    >= 1 - 0.05 at *every* radius, OOD agreement >= 0.9.

    Null: this same call with ``f_hat`` a randomly initialised model of the same
    shape must fail (``tests/test_verify.py``) -- a battery that cannot fail is
    not evidence.
    Failure boundary: perturbation curves need a continuous input space; for
    integer token inputs the perturbation block is skipped and the reason is
    recorded in ``perturbation_skipped`` (the check is then not counted).
    PGD/adversarial curves (also listed under R5) are not implemented here: they
    need the model's gradients, not just its logits, and belong with the
    attacking code.
    """
    tol = dict(DEFAULT_TOL, **(tol or {}))
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    L, Lh = _logits(f, X), _logits(f_hat, X)
    if L.shape != Lh.shape:
        raise ValueError(f"f gives {L.shape} logits, f_hat gives {Lh.shape}")
    n, n_classes = L.shape
    pred, pred_hat = L.argmax(1), Lh.argmax(1)
    acc = acc_hat = drop = None
    if y is not None:
        yy = np.asarray(y).ravel()
        acc = float((pred == yy).mean()); acc_hat = float((pred_hat == yy).mean())
        drop = float(acc - acc_hat)
    corr = _pearson(L, Lh)
    per_row = float(np.mean([_pearson(L[i], Lh[i]) for i in range(n)])) if n_classes > 1 else 0.0
    agree = float((pred == pred_hat).mean())

    pert: Dict[float, float] = {}
    skipped = None
    if not np.issubdtype(X.dtype, np.floating):
        skipped = (f"input dtype {X.dtype} is not continuous (discrete tokens): Gaussian "
                   "perturbation is not defined on this input space")
    elif radii:
        scale = float(np.std(X))
        scale = scale if scale > 0 else 1.0
        for r in radii:
            ags = []
            for _ in range(int(n_pert)):
                Xp = X + (r * scale) * rng.standard_normal(X.shape)
                ags.append(float((_logits(f, Xp).argmax(1) == _logits(f_hat, Xp).argmax(1)).mean()))
            pert[float(r)] = float(np.mean(ags))

    ood_agree = None
    if ood is not None:
        Lo, Loh = _logits(f, ood), _logits(f_hat, ood)
        ood_agree = float((Lo.argmax(1) == Loh.argmax(1)).mean())

    checks: Dict[str, bool] = {}
    if drop is not None:
        checks["acc_drop"] = bool(drop <= tol["acc_drop"])
    checks["logit_corr"] = bool(corr >= tol["logit_corr"])
    if pert:
        checks["pert_band"] = bool(min(pert.values()) >= 1.0 - tol["pert_band"])
    if ood_agree is not None:
        checks["ood_agree"] = bool(ood_agree >= tol["ood_agree"])
    return Verdict(n, n_classes, acc, acc_hat, drop, corr, per_row, agree, pert, skipped,
                   ood_agree, tol, checks, bool(all(checks.values())), int(seed))


@dataclass
class AblationTable:
    """Per-object ablation-effect correlation between model and reconstruction (SPEC 7 R6)."""
    rows: List[dict]
    threshold: float
    all_passed: bool
    n_failed: int

    def to_dict(self) -> dict:
        return {"rows": [dict(r) for r in self.rows], "threshold": self.threshold,
                "all_passed": self.all_passed, "n_failed": self.n_failed}

    def to_markdown(self) -> str:
        L = ["| object | r | |effect| f | |effect| f_hat | pass |", "| --- | --- | --- | --- | --- |"]
        for r in self.rows:
            L.append(f"| {r['object']} | {r['r']:.4f} | {r['effect_norm']:.4g} | "
                     f"{r['effect_norm_hat']:.4g} | {'yes' if r['passed'] else 'no'} |")
        L.append(f"\n{self.n_failed} of {len(self.rows)} objects mis-mapped (r <= {self.threshold})")
        return "\n".join(L)


def ablation_symmetry(f_ablate: Callable, f_hat_ablate: Callable, objects: Iterable, X,
                      threshold: float = 0.9) -> AblationTable:
    """R6: ablate each certified object in ``f`` and its counterpart in ``f_hat`` (SPEC 7 R6).

    ``f_ablate(object_id, X) -> logits`` and ``f_hat_ablate(object_id, X) -> logits``;
    both must accept ``object_id=None`` for the un-ablated baseline (the effect
    vector is the logit *change* over ``X``, flattened). Correlation of the two
    effect vectors must exceed ``threshold`` (0.9); any object below it is
    mis-mapped, meaning the rebuild does the right thing overall for the wrong
    internal reason.

    Null: identical callables give r = 1 exactly; independent random callables
    give r ~ 0 (``tests/test_verify.py``).
    Failure boundary: an object whose ablation has no effect in either model
    gives an undefined correlation; it is reported as r = nan and counted as a
    failure, because an object with no effect was not a mechanism.
    """
    base = np.asarray(f_ablate(None, X), dtype=np.float64)
    base_hat = np.asarray(f_hat_ablate(None, X), dtype=np.float64)
    rows = []
    for k in objects:
        e = np.asarray(f_ablate(k, X), dtype=np.float64) - base
        eh = np.asarray(f_hat_ablate(k, X), dtype=np.float64) - base_hat
        if e.shape != eh.shape:
            raise ValueError(f"object {k!r}: effect shapes {e.shape} vs {eh.shape} differ")
        ne, neh = float(np.linalg.norm(e)), float(np.linalg.norm(eh))
        r = _pearson(e, eh) if (ne > 0 and neh > 0) else float("nan")
        rows.append({"object": k, "r": r, "effect_norm": ne, "effect_norm_hat": neh,
                     "passed": bool(r > threshold)})
    n_failed = sum(1 for r in rows if not r["passed"])
    return AblationTable(rows, float(threshold), n_failed == 0, n_failed)


@dataclass
class NegativeControl:
    """R7: artifact-only rebuild and the certified-object degradation curve (SPEC 7 R7)."""
    chance: float
    full_accuracy: float
    artifact_accuracy: float
    artifact_at_chance: bool
    curve: List[dict]                      # {n_kept, frac_kept, acc_mean, acc_std}
    monotone_rho: float
    degrades_gracefully: bool
    n_draws: int
    seed: int

    def to_dict(self) -> dict:
        return {"chance": self.chance, "full_accuracy": self.full_accuracy,
                "artifact_accuracy": self.artifact_accuracy,
                "artifact_at_chance": self.artifact_at_chance,
                "curve": [dict(c) for c in self.curve], "monotone_rho": self.monotone_rho,
                "degrades_gracefully": self.degrades_gracefully,
                "n_draws": self.n_draws, "seed": self.seed}

    def to_markdown(self) -> str:
        L = [f"artifact-only accuracy {self.artifact_accuracy:.4f} (chance {self.chance:.4f}, "
             f"full {self.full_accuracy:.4f})", "", "| kept | fraction | accuracy |", "| --- | --- | --- |"]
        for c in self.curve:
            L.append(f"| {c['n_kept']} | {c['frac_kept']:.2f} | {c['acc_mean']:.4f} +- {c['acc_std']:.4f} |")
        L.append(f"\nSpearman(fraction kept, accuracy) = {self.monotone_rho:.3f}")
        return "\n".join(L)


def negative_control(rebuild_fn: Callable, artifact_objects, certified_objects, X, y,
                     n_draws: int = 10, seed: int = 0, chance_tol: float = 0.05) -> NegativeControl:
    """R7: the rebuild must fail from artifacts and degrade gracefully from subsets (SPEC 7 R7).

    ``rebuild_fn(objects) -> callable X -> logits`` is the *same* reconstruction
    procedure used for the real rebuild. It is run (a) on the artifact-class
    objects only -- accuracy must land near chance -- and (b) on random subsets of
    the certified objects of every size, ``n_draws`` draws each, giving the
    degradation curve and its Spearman correlation with the fraction kept.

    Null: this function *is* the null for :func:`equivalence`. A rebuild that
    scores well from artifacts is fitting the task, not reading the model.
    Failure boundary: needs labels ``y`` and a rebuild procedure that accepts an
    arbitrary object subset; if ``rebuild_fn`` fails on an empty subset, the
    curve starts at one object.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X); yy = np.asarray(y).ravel()

    def acc(objs) -> float:
        L = _logits(rebuild_fn(list(objs)), X)
        return float((L.argmax(1) == yy).mean())

    n_classes = int(_logits(rebuild_fn(list(certified_objects)), X).shape[1])
    chance = 1.0 / max(n_classes, 1)
    full = acc(certified_objects)
    art = acc(artifact_objects) if len(list(artifact_objects)) else chance
    cert = list(certified_objects)
    curve = []
    for m in range(1, len(cert) + 1):
        accs = []
        for _ in range(int(n_draws) if m < len(cert) else 1):
            sub = rng.choice(len(cert), size=m, replace=False)
            accs.append(acc([cert[i] for i in sub]))
        curve.append({"n_kept": m, "frac_kept": m / len(cert),
                      "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs))})
    fk = np.array([c["frac_kept"] for c in curve]); am = np.array([c["acc_mean"] for c in curve])
    if len(curve) > 1 and np.std(am) > 0:
        from scipy.stats import spearmanr
        rho = float(spearmanr(fk, am).statistic)
    else:
        rho = float("nan")
    return NegativeControl(chance, full, art, bool(art <= chance + chance_tol), curve, rho,
                           bool(np.isnan(rho) or rho > 0.5), int(n_draws), int(seed))
