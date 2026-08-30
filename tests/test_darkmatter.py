"""whitebox.darkmatter -- planted controls (E6 SPEC's own instrument-controls section)
plus the E6 battery headline reproduced from the committed results CSVs.
"""
import csv
import os
import ast

import numpy as np
import pytest

from whitebox.darkmatter import (residual_structure, DarkMatterReport, mp_outliers, feature_shuffle,
                                 gaussian_matched_cov, subsample_outliers, SPECTRAL_N_OUTLIERS_MIN,
                                 TOPOLOGY_TC_RATIO_MIN)
from conftest import circle

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks", "results", "darkmatter")


# ---------------------------------------------------------------------------
# (a) pure i.i.d. Gaussian: n_outliers ~ 0, tc_ratio ~ 1 -- verdict "neither"
# ---------------------------------------------------------------------------
def test_gaussian_neither():
    rng = np.random.default_rng(0)
    E = rng.standard_normal((1200, 120))
    rep = residual_structure(E, n_points=500, n_boot=5, n_components=12, n_subsample=5, n_shuffle_seeds=3)
    assert rep.n_outliers <= 1
    assert rep.spectral_test_valid
    assert not rep.spectrally_structured
    assert 0.75 < rep.tc_ratio < 1.3          # shuffle null has spread at n=500
    assert not rep.topologically_structured
    assert rep.verdict == "neither"
    # SPEC control 4: residual_structure returns the fields the Stage-2 API requires
    for field in ("structured_fraction", "n_outliers", "mp_edge", "tc_ratio", "tc_ratio_ci"):
        assert hasattr(rep, field)
    assert isinstance(rep, DarkMatterReport)


# ---------------------------------------------------------------------------
# (b) planted rank-k spike (well above the BBP threshold) + noise: n_outliers >= k,
# shuffle validity holds (the null still works on this data)
# ---------------------------------------------------------------------------
def test_planted_spike_detected():
    rng = np.random.default_rng(1)
    n, p, k = 2500, 180, 5
    gamma = p / n
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    spike = 6 * lam_plus                      # BBP: well above threshold for detectability
    U, _ = np.linalg.qr(rng.standard_normal((p, k)))
    Z = rng.standard_normal((n, k)) * np.sqrt(spike)
    E = Z @ U.T + rng.standard_normal((n, p))
    rep = residual_structure(E, topology=False, n_subsample=5, n_shuffle_seeds=5)
    assert rep.n_outliers >= k
    assert rep.spectral_test_valid                  # kill (iii): the null itself still works
    assert rep.shuffle_n_outliers <= 1


# ---------------------------------------------------------------------------
# (c) genuine H1 structure: a noisy circle embedded in d dims -> tc_ratio clears
# its own shuffle null AND the covariance-matched Gaussian control
# ---------------------------------------------------------------------------
def test_noisy_circle_topologically_structured():
    n, d = 1200, 40
    c = circle(n, r=3.0, noise=0.15, seed=2)
    rng = np.random.default_rng(2)
    pad = 0.3 * rng.standard_normal((n, d - 2))
    E = np.concatenate([c, pad], axis=1) + 0.15 * rng.standard_normal((n, d))
    rep = residual_structure(E, n_points=500, n_boot=5, n_components=10, n_subsample=5, n_shuffle_seeds=3)
    assert rep.tc_ratio > TOPOLOGY_TC_RATIO_MIN
    assert rep.tc_ratio_ci[0] > 1.0
    assert rep.gaussian_control_tc_ratio is not None and rep.tc_ratio > rep.gaussian_control_tc_ratio
    assert rep.topologically_structured
    assert rep.verdict in ("topologically_structured", "both")


# ---------------------------------------------------------------------------
# (d) the E6 headline reproduces from the committed E6 results CSVs
# (benchmarks/results/darkmatter/, copied verbatim from whitebox-v2/E6/results/)
# ---------------------------------------------------------------------------
def test_e6_headline_reproduces_from_csv():
    path = os.path.join(RESULTS_DIR, "e6_verdict.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert {r["layer"] for r in rows} == {"2", "6", "10"}
    for r in rows:
        n_out = int(r["n_out_E"])
        ci = ast.literal_eval(r["n_out_E_ci_noreplace"])
        tc_ratio = float(r["tc_ratio_E_point"])
        tc_ratio_ci = r["tc_ratio_E_ci"]
        # Test 1 passes cleanly at every layer (E6 FINDINGS.md: n_out 179-198 of 768)
        assert r["test1_valid"] == "True"
        assert r["test1_pass"] == "True"
        assert n_out > SPECTRAL_N_OUTLIERS_MIN and ci[0] > SPECTRAL_N_OUTLIERS_MIN
        # Test 2 fails at every layer -- tc_ratio(E) is BELOW 1.0 (simpler than its shuffle
        # null), not merely below the 1.2 H-E6 bar
        assert tc_ratio < 1.0
        assert r["test2_pass_point"] == "False"
        # H-E6 (both must hold) fails at every layer
        assert r["H_E6_point"] == "False"
    # the topology table's gauss-cov control calibrates to ~1, as the SPEC expected,
    # and is well above E's ratio at every layer
    with open(os.path.join(RESULTS_DIR, "e6_test2_topology.csv")) as f:
        topo_rows = list(csv.DictReader(f))
    by_layer = {(r["layer"], r["cloud"]): float(r["tc_ratio"]) for r in topo_rows}
    for layer in ("2", "6", "10"):
        e_ratio, gauss_ratio = by_layer[(layer, "E")], by_layer[(layer, "gauss_cov(E)")]
        assert e_ratio < gauss_ratio
        assert 0.8 < gauss_ratio < 1.2
    # shuffle-seed table: max outliers over 20 seeds x 3 layers is exactly 0 (kill iii holds)
    with open(os.path.join(RESULTS_DIR, "e6_test1_shuffle_seeds.csv")) as f:
        shuf_rows = list(csv.DictReader(f))
    n_out_col = "n_outliers" if "n_outliers" in shuf_rows[0] else [k for k in shuf_rows[0] if "outlier" in k][0]
    assert max(int(r[n_out_col]) for r in shuf_rows) == 0


# ---------------------------------------------------------------------------
# (e) shuffle validity check fails (reports invalid) when the shuffle itself has outliers:
# heavy-tailed (Cauchy, dof=1) marginals with aspect ratio near 1 break the Gaussian-
# derived Tracy-Widom threshold even after the column-shuffle preserves those marginals --
# exactly the failure mode kill (iii) exists to catch.
# ---------------------------------------------------------------------------
def test_shuffle_validity_catches_non_gaussian_marginals():
    rng = np.random.default_rng(1)
    E = rng.standard_t(1.0, size=(250, 240))
    rep = residual_structure(E, topology=False, n_subsample=5, n_shuffle_seeds=5, seed=0)
    assert not rep.spectral_test_valid
    assert rep.shuffle_n_outliers > 1
    assert not rep.spectrally_structured    # the void-test guard: never claim structure on invalid data

    # sanity: the low-level function used directly agrees
    counts = [mp_outliers(feature_shuffle(E, seed=1000 + i)).n_outliers for i in range(5)]
    assert max(counts) > 1
