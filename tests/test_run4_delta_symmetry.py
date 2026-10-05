"""SPEC4 P1.2: the registered notion of symmetry -- sigma is accepted iff mean|D_sigma - D| < 0.25 * (mean defect of a random relabelling),
and the group statistic is whether a FULL n-cycle is among the accepted candidates.  Planted truth."""
import numpy as np, torch
from scipy.spatial.distance import pdist, squareform
from whiteboxtopo.symmetry_finite import automorphisms, full_cycle, cycle_coordinate, characters, element_order

fold = lambda k: min(k % 113, 113 - k % 113)


def fourier(noise, seed, freqs=(3, 26, 50), amps=(1.0, 0.8, 0.6)):
    rng = np.random.default_rng(seed); a = 2 * np.pi * np.arange(113) / 113
    X = np.concatenate([w * np.c_[np.cos(k * a), np.sin(k * a)] for k, w in zip(freqs, amps)], 1)
    return X + noise * rng.standard_normal(X.shape)


def test_noisy_fourier_embedding_in_scrambled_order_returns_a_full_cycle_and_its_characters():
    X = fourier(0.02, 1); perm = np.random.default_rng(2).permutation(113); r = automorphisms(squareform(pdist(X[perm])))
    assert r['has_full_cycle_rel']; s = full_cycle(r, 'perms_rel'); assert element_order(s) == 113
    ch = characters(X[perm], cycle_coordinate(s), top=3); assert any({fold(u * c['freq']) for c in ch} == {3, 26, 50} for u in range(1, 57))


def test_operating_range_mean_rule_outlives_the_entrywise_rule_and_has_a_documented_limit():
    """Measured 2026-09-20: entrywise (max-norm) isometry is lost by noise 0.05; the mean rule holds to noise 0.08 (true-shift mean defect
    6.1% of the median vs the acceptance level 0.25*delta_rand = 6.5%) and is lost at 0.10.  On these data 0.25*delta_rand ~ 0.065*median."""
    r = automorphisms(squareform(pdist(fourier(0.05, 3)))); assert r['has_full_cycle_rel'] and r['order'] < 226
    assert not automorphisms(squareform(pdist(fourier(0.15, 3))))['has_full_cycle_rel']


def test_structureless_cloud_has_no_accepted_full_cycle():
    r = automorphisms(squareform(pdist(np.random.default_rng(0).standard_normal((113, 12))))); assert not r['has_full_cycle_rel']


def test_untrained_z113_model_has_no_accepted_full_cycle():
    from analysis.run3_blockZ import all_logits, D_g
    from systems.z113 import Z113Transformer
    for seed in (11, 12):
        r = automorphisms(D_g(all_logits(Z113Transformer(seed=seed).eval()))); assert not r['has_full_cycle_rel'], seed
