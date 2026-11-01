"""residual_structure -- the whiteboxtopo.darkmatter public API.

Ported from whitebox-v2 E6 (2026-08-30, GPT-2 small / gpt2-small-res-jb SAELens
residual-stream SAEs, blocks.{2,6,10}.hook_resid_pre, 65k OpenWebText tokens/layer;
see E6/SPEC.md and E6/FINDINGS.md in the experiment tree). Two independent
instruments read the SAE reconstruction error E = x - SAE(x) ("dark matter",
Engels, Smith & Tegmark 2024, arXiv:2410.14670):

  Test 1 (spectral): a Marchenko-Pastur bulk + Tracy-Widom edge null on E's
    sample correlation matrix (``whiteboxtopo.darkmatter.rmt``), with the
    feature-shuffle kill (marginals preserved, cross-dimension correlation
    destroyed -- must give <= 1 outlier or the test is void on that data) and
    a without-replacement subsample CI (E6 SPEC Amendment 2: the naive
    with-replacement block bootstrap duplicates rows and re-introduces
    cross-column dependence into a shuffled matrix -- the same trap as E1's
    Amendment 5).

  Test 2 (topological): the v1 TC(H1)-with-feature-shuffle-null instrument
    (``whiteboxtopo.darkmatter.topo``, reusing ``whiteboxtopo.signals.memorization``
    unchanged) on a random subsample of the error cloud, with a bootstrap CI
    over independent subsamples and a covariance-matched Gaussian control
    (N(mean(E), Cov(E)) -- "covariance but no extra topology").

THE VERDICT IS NOT A CLAIM OF "STRUCTURED DARK MATTER". Read the docstring of
``residual_structure`` and of ``DarkMatterReport.verdict`` before using either
field for anything beyond what it literally measures: on the only battery run
so far (E6, GPT-2 small, 3 layers), Test 1 passed and Test 2 failed at every
layer -- every layer's verdict was "spectrally_structured" and none was
"topologically_structured" or "both". A tc_ratio < 1 means the error cloud is
LESS topologically complex than its own feature-shuffled null, not "no
structure" -- it is a specific, and specifically negative, finding (E6
FINDINGS.md: 0.11-0.46 at the three layers tested).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

from .rmt import mp_outliers, feature_shuffle, gaussian_matched_cov, subsample_outliers, MPResult
from .topo import tc_ratio_bootstrap

# Pre-registered thresholds (E6 SPEC, H-E6). Not re-fit here; they are the numbers the
# one existing battery was scored against, kept as module constants so a caller can see
# exactly what "structured" would have meant.
SPECTRAL_N_OUTLIERS_MIN = 10
TOPOLOGY_TC_RATIO_MIN = 1.2
TOPOLOGY_CI_LOWER_MIN = 1.0


@dataclass
class DarkMatterReport:
    """Both instrument readings for one error cloud. Nothing here is a claim that the
    dark matter IS structured -- see the module docstring and ``verdict``."""
    n: int
    p: int
    gamma: float
    form: str
    q: float

    # --- Test 1: spectral (Marchenko-Pastur / Tracy-Widom) ---
    mp_edge: float                      # bulk edge lambda+
    tw_threshold: float                 # MP-outlier cutoff at quantile q
    n_outliers: int
    structured_fraction: float          # sum(outlier eigenvalues) / sum(all eigenvalues)
    excess_fraction: float              # honest lower bound: bulk share removed
    eigenvalues: np.ndarray
    n_outliers_ci: Tuple[float, float]  # without-replacement subsample CI (Amendment 2)
    shuffle_n_outliers: int             # max over shuffle seeds (kill iii)
    shuffle_outliers_by_seed: Tuple[int, ...]
    spectral_test_valid: bool           # shuffle gives <= 1 outlier; if False Test 1 is VOID
    spectrally_structured: bool         # n_outliers>10, CI lower>10, AND spectral_test_valid

    # --- Test 2: topology (TC(H1) vs feature-shuffle null) ---
    topology_run: bool
    tc_ratio: Optional[float] = None
    tc_ratio_ci: Optional[Tuple[float, float]] = None
    tc_ratio_boot_mean: Optional[float] = None
    b1: Optional[float] = None
    pe1: Optional[float] = None
    tc: Optional[float] = None
    tc_shuffled: Optional[float] = None
    n_points: Optional[int] = None
    gaussian_control_tc_ratio: Optional[float] = None   # covariance-matched Gaussian's own ratio
    topologically_structured: bool = False

    verdict: str = "neither"
    meta: Dict = field(default_factory=dict)

    def __repr__(self):
        t2 = f"tc_ratio={self.tc_ratio:.3g} CI={self.tc_ratio_ci}" if self.topology_run else "topology not run"
        return (f"DarkMatterReport(n_outliers={self.n_outliers}/{self.p} (CI {self.n_outliers_ci}, "
               f"valid={self.spectral_test_valid}), {t2}, verdict={self.verdict!r})")


def _verdict(spectral: bool, topo_run: bool, topological: bool) -> str:
    if spectral and topo_run and topological:
        return "both"
    if spectral:
        return "spectrally_structured"
    if topo_run and topological:
        return "topologically_structured"
    return "neither"


def residual_structure(errors: Optional[np.ndarray] = None, *, acts: Optional[np.ndarray] = None,
                       recon: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None,
                       form: str = "correlation", q: float = 0.99,
                       topology: bool = True, gaussian_control: bool = True,
                       n_points: int = 2000, n_boot: int = 20, n_components: int = 30,
                       n_subsample: int = 20, subsample_frac: float = 0.8,
                       n_shuffle_seeds: int = 5, seed: int = 0) -> DarkMatterReport:
    """Structured-fraction of a residual/error matrix under an MP null, plus the v1
    topology-with-shuffle-null ratio, combined into a documented verdict.

    Parameters
    ----------
    errors : (n, d) array, or None
        The SAE (or any) reconstruction error e = x - recon(x), tokens/samples x dims.
        If None, pass ``acts`` and ``recon`` instead (both (n, d)) and ``errors = acts - recon``
        is computed for you -- this is the "(acts, recon) pairs" calling convention.
    groups : (n,) array, optional
        Sequence/document id per row, for the without-replacement subsample CI and the
        (unused-by-default) block bootstrap. Default: every row is its own group (i.i.d.
        subsampling of rows). Pass this if your rows are not exchangeable (e.g. tokens
        within a sequence), matching E6's per-sequence resampling.
    form : "correlation" (primary; heterogeneous-diagonal-safe) or "covariance" (secondary,
        over-counts -- E6 kill (i): a matched-diagonal Gaussian gives 0 outliers in
        "correlation" form but produces spurious outliers in "covariance" form).
    topology : run Test 2 (the more expensive persistent-homology instrument). Set False
        to skip it (e.g. very large p where PCA->30 is still fine, but you only want Test 1).
    gaussian_control : also compute the covariance-matched Gaussian control's tc_ratio and
        require E's ratio to exceed it before calling the topology reading "structured"
        (E6 SPEC's interpretive commitment: matching the covariance-only control means
        "topologically Gaussian-like", not "structured computation").
    n_points, n_boot, n_components : Test 2 subsample size / bootstrap draws / PCA dims
        (E6 SPEC: 2000 / 20 / 30; smaller values are fine for quick checks but widen the CI).
    n_subsample, subsample_frac : Test 1's without-replacement subsample CI (E6 SPEC
        Amendment 2: 20 draws of 80% of the groups).
    n_shuffle_seeds : independent feature-shuffle seeds for the kill-(iii) validity check
        (E6 used 20 on the real battery; 5 is the library default for speed, still enough
        to catch a shuffle that is not actually decorrelated).

    Returns
    -------
    DarkMatterReport -- see its docstring and the module docstring for what ``verdict``
    does and does not mean. THIS FUNCTION DOES NOT CLAIM "DARK MATTER IS STRUCTURED";
    it reports two independent, falsifiable readings and their nulls.
    """
    if errors is None:
        if acts is None or recon is None:
            raise ValueError("pass errors=E, or acts= and recon= (errors = acts - recon is computed for you)")
        errors = np.asarray(acts, dtype=np.float64) - np.asarray(recon, dtype=np.float64)
    E = np.asarray(errors, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(f"errors must be (n, d); got shape {E.shape}")
    n, p = E.shape

    r = mp_outliers(E, form=form, q=q)
    shuffle_counts = tuple(mp_outliers(feature_shuffle(E, seed=seed + 1000 + i), form=form, q=q).n_outliers
                           for i in range(n_shuffle_seeds))
    shuffle_valid = max(shuffle_counts) <= 1
    ci = subsample_outliers(E, groups=groups, n_draws=n_subsample, frac=subsample_frac, seed=seed + 1, form=form, q=q)
    spectrally_structured = bool(r.n_outliers > SPECTRAL_N_OUTLIERS_MIN
                                 and ci["n_outliers_ci"][0] > SPECTRAL_N_OUTLIERS_MIN and shuffle_valid)

    topo_run = bool(topology)
    tc_kw: Dict = {}
    gauss_ratio = None
    topologically_structured = False
    if topo_run:
        t = tc_ratio_bootstrap(E, n_points=n_points, n_boot=n_boot, n_components=n_components, seed=seed)
        tc_kw = dict(tc_ratio=t["tc_ratio"], tc_ratio_ci=t["tc_ratio_ci"], tc_ratio_boot_mean=t["tc_ratio_boot_mean"],
                    b1=t["b1"], pe1=t["pe1"], tc=t["tc"], tc_shuffled=t["tc_shuffled"], n_points=t["n_points"])
        if gaussian_control:
            Gc = gaussian_matched_cov(E, seed=seed + 2000, n=n)
            tg = tc_ratio_bootstrap(Gc, n_points=n_points, n_boot=0, n_components=n_components, seed=seed + 2000)
            gauss_ratio = tg["tc_ratio"]
        topologically_structured = bool(
            t["tc_ratio"] > TOPOLOGY_TC_RATIO_MIN and t["tc_ratio_ci"][0] > TOPOLOGY_CI_LOWER_MIN
            and (gauss_ratio is None or t["tc_ratio"] > gauss_ratio))

    verdict = _verdict(spectrally_structured, topo_run, topologically_structured)

    return DarkMatterReport(
        n=n, p=p, gamma=r.gamma, form=form, q=q,
        mp_edge=r.mp_edge, tw_threshold=r.tw_threshold, n_outliers=r.n_outliers,
        structured_fraction=r.structured_fraction, excess_fraction=r.excess_fraction, eigenvalues=r.eigenvalues,
        n_outliers_ci=ci["n_outliers_ci"], shuffle_n_outliers=max(shuffle_counts),
        shuffle_outliers_by_seed=shuffle_counts, spectral_test_valid=shuffle_valid,
        spectrally_structured=spectrally_structured,
        topology_run=topo_run, gaussian_control_tc_ratio=gauss_ratio,
        topologically_structured=topologically_structured, verdict=verdict,
        meta=dict(n_shuffle_seeds=n_shuffle_seeds, subsample_ci=ci), **tc_kw)
