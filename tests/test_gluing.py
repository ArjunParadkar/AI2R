"""whitebox.gluing -- planted controls (mirrors E3 SPEC's own instrument tests:
homogeneous cover -> 0, a two-mechanism bridging split -> nonzero Omega with the
blocking pair recovered, shuffle-null sanity, brute-force cycle-space validation)
plus the E3 battery headline reproduced from the committed results CSVs.
"""
import csv
import os

import numpy as np
import pytest

from whitebox.gluing import (
    explanation_obstruction, GluingReport, CAVEAT,
    nerve, obstruction, bottleneck, h0, brute_force_obs,
    cover_from_embedding,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks", "results", "gluing")


# ---------------------------------------------------------------------------
# helper: a 4-cluster cyclic cover with two planted "mechanisms" (circuits A, B
# sharing 10 of 20 edges) and one deliberately asymmetric bridging overlap.
# Clusters 0,1 -> circuit A; clusters 2,3 -> circuit B; cluster order around the
# nerve cycle is 0-1-2-3-0. Edge (1,2) is a "clean" A/B crossing (pure-A-type
# overlap rows) -> low, tied-looking agreement. Edge (0,3) is a mixed-row
# crossing (half A-type, half B-type rows in the overlap) -> the shared index
# range 10-19 gets doubled signal from both halves and both restrictions latch
# onto it, so J(0,3) comes out HIGH despite crossing the same mechanism
# boundary. The asymmetry is what makes edge (1,2) genuinely non-2-colourable
# (frustrated): its endpoints are still connected via 1-0-3-2, all high-J.
# ---------------------------------------------------------------------------
def _two_mechanism_cover(seed=0, core=100, ov=20, n_edges=40, noise=0.3, base=10.0):
    rng = np.random.default_rng(seed)
    circuitA = np.arange(0, 20)
    circuitB = np.arange(10, 30)   # overlaps circuit A on indices 10-19
    N = core * 4 + ov * 4
    scores = rng.normal(0, noise, size=(N, n_edges))

    idx = 0
    blocks = {}
    for name, n in (("core0", core), ("core1", core), ("core2", core), ("core3", core),
                    ("ov01", ov), ("ov12", ov), ("ov23", ov), ("ov30", ov)):
        blocks[name] = np.arange(idx, idx + n)
        idx += n

    def set_circuit(rows, circuit):
        sig = base - 0.01 * np.arange(len(circuit))   # monotonic decreasing -> deterministic top-k
        scores[np.ix_(rows, circuit)] += sig[None, :]

    set_circuit(blocks["core0"], circuitA)
    set_circuit(blocks["core1"], circuitA)
    set_circuit(blocks["core2"], circuitB)
    set_circuit(blocks["core3"], circuitB)
    set_circuit(blocks["ov01"], circuitA)
    set_circuit(blocks["ov12"], circuitA)                    # pure-A crossing overlap
    half = ov // 2
    set_circuit(blocks["ov30"][:half], circuitA)              # mixed crossing overlap
    set_circuit(blocks["ov30"][half:], circuitB)
    set_circuit(blocks["ov23"], circuitB)

    members = np.zeros((N, 4), bool)
    members[blocks["core0"], 0] = True
    members[blocks["core1"], 1] = True
    members[blocks["core2"], 2] = True
    members[blocks["core3"], 3] = True
    members[blocks["ov01"], 0] = True; members[blocks["ov01"], 1] = True
    members[blocks["ov12"], 1] = True; members[blocks["ov12"], 2] = True
    members[blocks["ov23"], 2] = True; members[blocks["ov23"], 3] = True
    members[blocks["ov30"], 3] = True; members[blocks["ov30"], 0] = True
    return scores, members


# ---------------------------------------------------------------------------
# (a) homogeneous explanations: every example shares one circuit -> Omega ~ 0
# and inside its own shuffle null band, regardless of an arbitrary cover.
# ---------------------------------------------------------------------------
def test_homogeneous_cover_omega_near_zero_inside_null():
    rng = np.random.default_rng(0)
    n, n_edges = 400, 50
    scores = rng.normal(0, 0.1, size=(n, n_edges))
    scores[:, :10] += 10.0                       # one shared circuit, every example
    emb = rng.normal(size=(n, 5))                # arbitrary embedding -- must not matter
    rep = explanation_obstruction(scores, embedding=emb, k=6, k_e=10, n_min=10,
                                  n_shuffle=100, n_boot=50, seed=0)
    assert isinstance(rep, GluingReport)
    assert rep.omega == pytest.approx(0.0, abs=1e-9)
    assert rep.null_lo <= rep.omega <= rep.null_hi + 1e-9
    assert not rep.significant_vs_null
    assert rep.mean_J == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# (b) two planted mechanisms split across cover clusters, one asymmetric
# bridging overlap -> Omega > null band, the planted blocking pair (1,2)
# recovered as the top (and only) frustrated edge.
# ---------------------------------------------------------------------------
def test_two_mechanism_split_recovers_blocking_pair():
    scores, members = _two_mechanism_cover(seed=0)
    rep = explanation_obstruction(scores, cover=members, k_e=20, n_min=10,
                                  n_shuffle=200, n_boot=100, seed=1)
    assert rep.cyclomatic == 1                    # a single 4-cycle nerve
    assert len(rep.edges) == 4
    assert rep.omega > rep.null_hi                 # clears its own null band
    assert rep.significant_vs_null
    assert rep.p_null <= 0.025
    assert rep.n_frustrated_edges == 1
    top = rep.blocking_pairs[0]
    assert top["pair"] == (1, 2)                   # the planted crossing edge
    assert top["m_e"] > 0.5
    assert top["m_e_ci"][0] > 0.0                   # bootstrap CI excludes 0
    # the other three edges (0,1), (2,3) [within-mechanism] and (0,3) [the
    # asymmetric mixed crossing] are not frustrated
    non_top = [bp for bp in rep.blocking_pairs if bp["pair"] != (1, 2)]
    assert all(bp["m_e"] == pytest.approx(0.0, abs=1e-9) for bp in non_top)


def test_two_mechanism_split_robust_across_seeds():
    for seed in (1, 2, 3, 42):
        scores, members = _two_mechanism_cover(seed=seed)
        rep = explanation_obstruction(scores, cover=members, k_e=20, n_min=10,
                                      n_shuffle=50, n_boot=0, seed=1)
        assert rep.omega > 0.5
        assert rep.blocking_pairs[0]["pair"] == (1, 2)


# ---------------------------------------------------------------------------
# (c) shuffle null sanity: permuting which examples belong to which cluster
# (destroying the correlation between membership and the planted mechanism,
# keeping cluster sizes and the nerve fixed) collapses the planted signal.
# ---------------------------------------------------------------------------
def test_shuffle_destroys_planted_signal():
    scores, members = _two_mechanism_cover(seed=0)
    rep = explanation_obstruction(scores, cover=members, k_e=20, n_min=10,
                                  n_shuffle=0, n_boot=0, seed=1)
    rng = np.random.default_rng(9)
    perm = rng.permutation(members.shape[0])
    rep_shuffled = explanation_obstruction(scores, cover=members[perm], k_e=20, n_min=10,
                                           n_shuffle=0, n_boot=0, seed=1)
    assert rep.omega > 0.5
    assert rep_shuffled.omega < 0.05 * rep.omega   # signal collapses under the shuffle

    # the report's own internal null (permuting rows the same way, many times)
    # brackets the planted Omega above the null band it produces
    rep_with_null = explanation_obstruction(scores, cover=members, k_e=20, n_min=10,
                                            n_shuffle=200, n_boot=0, seed=1)
    assert rep_with_null.null_hi < rep_with_null.omega


# ---------------------------------------------------------------------------
# (d) cyclomatic number / obstruction correct on a known nerve (a 5-cycle),
# validated both against a hand count and a brute-force cycle enumeration
# (independent of the closed-form bottleneck construction), plus a sweep of
# random tiny nerves.
# ---------------------------------------------------------------------------
def test_cyclomatic_number_on_cycle_graph():
    n_v = 5
    members = np.zeros((100, n_v), bool)
    for i in range(n_v):
        members[i * 20:(i + 1) * 20, i] = True
        members[i * 20:(i + 1) * 20, (i + 1) % n_v] = True
    edges, tris, ov = nerve(members, n_min=10)
    assert len(edges) == 5                          # a simple 5-cycle
    assert tris == []
    J = np.full(5, 0.9)
    J[0] = 0.1                                       # one low edge on the cycle
    res = obstruction(n_v, edges, J)
    assert res["cyclomatic"] == 1                    # |E| - |V| + 1 component = 5-5+1
    assert res["omega"] == pytest.approx(0.8, abs=1e-9)   # J_bott(0)=0.9, m_0=0.8, rest 0
    for tau, expect in ((0.05, 0), (0.15, 1), (0.5, 1), (0.95, 0)):
        idx = int(np.argmin(np.abs(res["taus"] - tau)))
        assert res["obs"][idx] == expect
        assert brute_force_obs(n_v, edges, J, tau) == expect


def test_obstruction_matches_brute_force_on_random_small_nerves():
    rng = np.random.default_rng(0)
    checked = 0
    for trial in range(30):
        n_v = rng.integers(3, 7)
        members = rng.random((60, n_v)) < 0.35
        edges, tris, ov = nerve(members, n_min=3)
        if len(edges) < 3:
            continue
        J = rng.random(len(edges))
        res = obstruction(int(n_v), edges, J)
        for tau in (0.2, 0.5, 0.8):
            idx = int(np.argmin(np.abs(res["taus"] - tau)))
            assert res["obs"][idx] == brute_force_obs(int(n_v), edges, J, tau)
        checked += 1
    assert checked >= 10       # enough random graphs actually had a nontrivial nerve


# ---------------------------------------------------------------------------
# embedding-driven cover: cover_from_embedding (method="knn2", the default)
# produces the every-example-in-exactly-two-clusters structure, and
# explanation_obstruction accepts an embedding directly (not just a cover=).
# ---------------------------------------------------------------------------
def test_cover_from_embedding_knn2_two_clusters_each():
    rng = np.random.default_rng(0)
    n, k = 300, 6
    emb = rng.normal(size=(n, 16))
    cov = cover_from_embedding(emb, k=k, n_pca=10, seed=0, n_min=5)
    assert cov["members"].sum(1).max() == 2
    assert cov["members"].sum(1).min() == 2       # exactly 2 clusters per example
    assert cov["tris"] == []                       # triple overlaps empty by construction


def test_explanation_obstruction_requires_exactly_one_of_cover_or_embedding():
    scores = np.zeros((10, 5))
    with pytest.raises(ValueError):
        explanation_obstruction(scores)
    with pytest.raises(ValueError):
        explanation_obstruction(scores, cover=np.zeros((10, 3), bool), embedding=np.zeros((10, 2)))


# ---------------------------------------------------------------------------
# the report never claims "distinct mechanisms" -- no such field exists, and
# the caveat spells out the E3 control failure.
# ---------------------------------------------------------------------------
def test_report_has_no_mechanism_verdict_field_and_carries_the_caveat():
    field_names = {f for f in GluingReport.__dataclass_fields__}
    for banned in ("verdict", "mechanisms", "distinct_mechanisms", "localisation", "localization"):
        assert banned not in field_names
    assert "control" in CAVEAT.lower() or "null" in CAVEAT.lower()
    assert "1.517" in CAVEAT and "0.564" in CAVEAT and "0.991" in CAVEAT
    assert "second-embedding" in CAVEAT or "second embedding" in CAVEAT


# ---------------------------------------------------------------------------
# (e) the E3 headline numbers reproduce from the committed results CSVs
# (benchmarks/results/gluing/, copied verbatim from whitebox-v2/E3/results/).
# Summary config: k=10, k_e=200, embedding=layer 11. Induction (the control)
# FAILS its own null-band criterion -- the battery's central caution, recorded
# here so nobody re-derives it. IOI clears its own null (p=0.005) but with a
# smaller raw Omega than the control in every grid cell (not re-checked here
# beyond the two summary rows -- see E3/FINDINGS.md for the full 40-cell sweep).
# ---------------------------------------------------------------------------
def test_e3_headline_reproduces_from_csv():
    with open(os.path.join(RESULTS_DIR, "summary.csv")) as f:
        rows = list(csv.DictReader(f))
    summary = {(r["task"], r["emb"], r["k"], r["k_e"]): r for r in rows if r["summary"] == "True"}
    induction = summary[("induction", "11", "10", "200")]
    ioi = summary[("ioi", "11", "10", "200")]

    # H-E3-control: FAILS. Omega is outside its own null band, not inside it.
    omega_ind = float(induction["omega"])
    null_lo_ind, null_hi_ind = float(induction["null_lo"]), float(induction["null_hi"])
    assert omega_ind == pytest.approx(1.517265144364561, abs=1e-6)
    assert null_lo_ind == pytest.approx(0.5644051946379892, abs=1e-6)
    assert null_hi_ind == pytest.approx(0.9909147972145916, abs=1e-6)
    assert omega_ind > null_hi_ind                  # the control failure
    assert float(induction["p_null"]) == 0.0
    assert float(induction["omega"]) / float(induction["n_edges"]) <= 0.05   # H-E3-control (a) still holds

    # H-E3-IOI: literal criteria met (p_null clears 0.025) but with SMALLER raw
    # Omega than the (failed) control -- the pre-registered direction is wrong.
    omega_ioi = float(ioi["omega"])
    assert omega_ioi == pytest.approx(0.22358426318822355, abs=1e-6)
    assert float(ioi["p_null"]) == pytest.approx(0.005, abs=1e-6)
    assert float(ioi["p_null"]) <= 0.025
    assert omega_ioi < omega_ind                     # the 40/40 inversion, at the summary cell

    with open(os.path.join(RESULTS_DIR, "pairs.csv")) as f:
        pair_rows = list(csv.DictReader(f))
    ioi_pairs = [r for r in pair_rows if r["task"] == "ioi" and r["emb"] == "11"
                and r["k"] == "10" and r["k_e"] == "200"]
    top = max(ioi_pairs, key=lambda r: float(r["m_e"]))
    assert (top["i"], top["j"]) == ("4", "5")
    assert float(top["m_e"]) == pytest.approx(0.0899, abs=1e-3)
    assert float(top["m_ci_lo"]) > 0.0                # bootstrap CI excludes 0
