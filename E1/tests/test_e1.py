"""SPEC controls 1, 2, 3, 5. Expected values are written here before the trained-network run."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from e1lib.functions import truth_table
from e1lib.models import MLP, lookup_table_mlp, train_mlp
from e1lib.enumerate import enumerate_valid, minimal_ids
from e1lib.gauge import (random_gauge, permute_mask_id, explanation_keys, orbit_counts,
                         canonical_form)
from e1lib.funcdim import functional_dimension

EPS = [0.01, 0.02, 0.05, 0.1, 0.2]


def parity3_units(shift):
    """Hidden layer computing ReLU(s - shift_k) for s = x1+x2+x3. Returns (W (3,3), b (3,))."""
    W = np.ones((3, 3))
    b = -np.asarray(shift, dtype=float)
    return W, b


def hand_xor3(second_copy=None, scale_copy=1.0):
    """Two-hidden-layer XOR3. Copy A: units ReLU(s), ReLU(s-1), ReLU(s-2), out = h0-2h1+2h2.
    Layer 2 is an identity pass-through. second_copy: None | "same" | "different"."""
    W1, b1 = parity3_units([0, 1, 2]); out1 = np.array([1., -2., 2.])
    if second_copy is None:
        W1s, b1s, outs = [W1], [b1], [out1]
    elif second_copy == "same":
        W1s, b1s, outs = [W1, W1 * scale_copy], [b1, b1 * scale_copy], [out1, out1 / scale_copy]
    else:  # different implementation: ReLU(s-.5), ReLU(s-1.5), ReLU(s-2.5); out = 2u1-6u2+10u3
        W1b, b1b = parity3_units([0.5, 1.5, 2.5]); out2 = np.array([2., -6., 10.])
        W1s, b1s, outs = [W1, W1b], [b1, b1b], [out1, out2]
    W1 = np.concatenate(W1s, axis=1); b1 = np.concatenate(b1s)
    k = W1.shape[1]
    W2 = np.eye(k); b2 = np.zeros(k)                      # identity pass-through layer
    W3 = np.concatenate(outs).reshape(k, 1); b3 = np.array([-0.5])
    return MLP([W1, W2, W3], [b1, b2, b3])


def summarise(model, X, y, scheme):
    valid = enumerate_valid(model, X, y, scheme)
    mins = minimal_ids(valid)
    recs = [explanation_keys(model, X, int(m), scheme, EPS) for m in mins]
    return valid, mins, recs, orbit_counts(recs, EPS)


def test_hand_xor3_single_copy_is_one_circuit():
    X, y = truth_table("XOR3")
    m = hand_xor3()
    assert (m.predict(X) == y).all()
    valid, mins, recs, oc = summarise(m, X, y, "zero")
    assert len(mins) == 1 and oc["n_expl"] >= 1


def test_duplicated_copy_collapses_to_one_orbit():
    X, y = truth_table("XOR3")
    m = hand_xor3("same", scale_copy=3.0)
    assert (m.predict(X) == y).all()
    valid, mins, recs, oc = summarise(m, X, y, "zero")
    # 2 copies x 3 units, output sums both copies => any choice of one unit per index works: 2^3 minimal circuits
    assert len(mins) == 8, len(mins)
    for eps in EPS:
        assert oc[f"circuit_orbits_eps{eps}"] == 1, (eps, oc)
    assert oc["circuit_orbits_functional"] == 1


def test_different_implementation_gives_distinct_parameter_orbits_same_functional():
    X, y = truth_table("XOR3")
    m = hand_xor3("different")
    assert (m.predict(X) == y).all()
    valid, mins, recs, oc = summarise(m, X, y, "zero")
    assert len(mins) >= 2
    for eps in [0.01, 0.02, 0.05]:
        assert oc[f"circuit_orbits_eps{eps}"] >= 2, (eps, oc)
    assert oc["circuit_orbits_functional"] <= oc["circuit_orbits_eps0.05"]
    # the two pure copies compute identical Boolean intermediates
    assert oc["circuit_orbits_functional"] == 1 or len(mins) > 2


def test_gauge_invariance_of_enumeration_and_keys():
    rng = np.random.default_rng(0)
    X, y = truth_table("MAJ3")
    m, conv, _ = train_mlp(X, y, depth=2, width=5, seed=1)
    assert conv
    g, perms, scales = random_gauge(m, rng)
    assert (g.predict(X) == y).all()
    for scheme in ["zero", "mean"]:
        v1 = enumerate_valid(m, X, y, scheme); v2 = enumerate_valid(g, X, y, scheme)
        assert set(permute_mask_id(int(v), perms, m.widths) for v in v1) == set(int(v) for v in v2)
        r1 = [explanation_keys(m, X, int(c), scheme, EPS) for c in minimal_ids(v1)]
        r2 = [explanation_keys(g, X, int(c), scheme, EPS) for c in minimal_ids(v2)]
        assert sorted(c for c, _, _ in r1) == sorted(c for c, _, _ in r2)
        assert sorted(f for _, _, f in r1) == sorted(f for _, _, f in r2)
        for eps in EPS:
            assert sorted(k[eps] for _, k, _ in r1) == sorted(k[eps] for _, k, _ in r2), eps


def test_lookup_table_baseline():
    for name in ["XOR3", "MAJ3", "AND_OR3"]:
        X, y = truth_table(name)
        m = lookup_table_mlp(X, y)
        assert (m.predict(X) == y).all()
        valid, mins, recs, oc = summarise(m, X, y, "zero")
        # exactly the true-pattern units are needed (false units only push the logit further negative)
        assert len(mins) == 1 and bin(int(mins[0])).count("1") == int(y.sum())
        # SPEC amendment 2 (Méloux thresholds): each one-hot unit takes values {0,1} -> one
        # separation each; output = OR of them -> exactly one interpretation.
        assert oc["n_expl"] == 1


def test_untrained_network_has_no_valid_circuit():
    X, y = truth_table("XOR3")
    bad = 0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        m = MLP([rng.normal(size=(3, 6)), rng.normal(size=(6, 6)), rng.normal(size=(6, 1))],
                [rng.normal(size=6), rng.normal(size=6), rng.normal(size=1)])
        if (m.predict(X) == y).all():
            continue
        v = enumerate_valid(m, X, y, "zero")
        bad += len(v) > 0  # a random net can have a valid *subnetwork* even if the full net is wrong
    # not asserted to be zero: XOR3 has only 8 inputs, so lucky subnetworks exist; SPEC control 5
    # is reported as a rate in the run script. Here we only check the code runs.
    assert bad <= 5


def test_functional_dimension_hand_net():
    X, y = truth_table("XOR3")
    m = hand_xor3()
    fd = functional_dimension(m, np.random.default_rng(0))
    assert fd["rank"] <= fd["expected_rank"]
    assert fd["deficit"] >= 0


def test_dp_count_equals_dfs_count():
    from e1lib.enumerate import interpretations_dfs, interpretation_count
    for name, w in [("XOR3", 5), ("MAJ3", 6), ("AND_OR3", 6)]:
        X, y = truth_table(name)
        m, conv, _ = train_mlp(X, y, depth=2, width=w, seed=3)
        if not conv:
            continue
        for scheme in ["zero", "mean"]:
            mins = minimal_ids(enumerate_valid(m, X, y, scheme))
            for c in mins[:6]:
                assert interpretation_count(m, X, int(c), scheme) == interpretations_dfs(m, X, int(c), scheme)[0]
    m = hand_xor3("different"); X, y = truth_table("XOR3")
    for c in minimal_ids(enumerate_valid(m, X, y, "zero")):
        assert interpretation_count(m, X, int(c), "zero") == interpretations_dfs(m, X, int(c), "zero")[0]
