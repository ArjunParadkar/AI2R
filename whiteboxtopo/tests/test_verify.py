"""whiteboxtopo.verify -- R5 equivalence battery, R6 ablation symmetry, R7 negative control."""
import numpy as np
import pytest

from whiteboxtopo.invariants.extractors import fourier_irreps
from whiteboxtopo.reconstruct import from_irreps
from whiteboxtopo.verify import equivalence, ablation_symmetry, negative_control, DEFAULT_TOL
from test_extractors import P, K_TRUE, synthetic_grok, random_mlp


# ------------------------------------------------------------------ R5
def test_equivalence_self_passes():
    m = random_mlp(seed=20)
    X = np.random.default_rng(21).standard_normal((120, 6))
    y = m.logits(X).argmax(1)
    v = equivalence(m.logits, m.logits, X, y=y, n_pert=3)
    assert v.passed and v.acc_drop == 0.0 and v.logit_corr > 0.999
    assert v.argmax_agreement == 1.0
    assert set(v.perturbation) == set(DEFAULT_TOL and (0.01, 0.02, 0.05, 0.1, 0.2, 0.5))
    assert min(v.perturbation.values()) == 1.0
    assert "PASS" in v.to_markdown() and v.to_dict()["passed"] is True


def test_equivalence_random_model_fails():
    """Null: a battery that cannot fail is not evidence."""
    m, other = random_mlp(seed=20), random_mlp(seed=99)
    X = np.random.default_rng(22).standard_normal((120, 6))
    y = m.logits(X).argmax(1)
    v = equivalence(m.logits, other.logits, X, y=y, n_pert=3)
    assert not v.passed
    assert v.logit_corr < 0.98 and not v.checks["logit_corr"]
    assert not v.checks["pert_band"]
    assert "FAIL" in v.to_markdown()


def test_equivalence_ood_and_perturbation_bands():
    m = random_mlp(seed=23)
    X = np.random.default_rng(24).standard_normal((80, 6))
    ood = 6.0 * np.random.default_rng(25).standard_normal((40, 6))
    v = equivalence(m.logits, m.logits, X, ood=ood, n_pert=2)
    assert v.ood_agreement == 1.0 and v.checks["ood_agree"] and v.passed
    assert v.accuracy is None and v.acc_drop is None and "acc_drop" not in v.checks


def test_equivalence_skips_perturbation_on_token_inputs():
    """Failure boundary: Gaussian perturbation is undefined on discrete tokens."""
    mdl = synthetic_grok()
    inv = fourier_irreps(mdl.unembed.T, P)                  # certified readout inventory
    rec = from_irreps(inv.K, inv.amplitude, P)
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    toks = np.stack([a.ravel(), b.ravel(), np.full(P * P, P)], 1)
    v = equivalence(mdl.forward, rec.logits_from_tokens, toks, y=(a.ravel() + b.ravel()) % P)
    assert v.perturbation == {} and "discrete tokens" in v.perturbation_skipped
    assert v.accuracy == 1.0 and v.accuracy_hat == 1.0 and v.passed


# ------------------------------------------------------------------ R6
def _ablator(model, X):
    """(object_id, X) -> logits, object_id=None being the un-ablated baseline."""
    from whiteboxtopo.reconstruct import from_circuits

    def f(obj, Xq):
        masks = [np.ones(w, dtype=bool) for w in model.widths]
        if obj is not None:
            masks[obj[0]][obj[1]] = False
        return from_circuits(model, masks, X)(Xq)
    return f


def test_ablation_symmetry_identical_callables():
    m = random_mlp(seed=30)
    X = np.random.default_rng(31).standard_normal((60, 6))
    f = _ablator(m, X)
    objs = [(0, 0), (0, 5), (1, 2), (1, 9)]
    tab = ablation_symmetry(f, f, objs, X)
    assert tab.all_passed and tab.n_failed == 0
    assert all(abs(r["r"] - 1.0) < 1e-12 for r in tab.rows)
    assert "yes" in tab.to_markdown() and len(tab.to_dict()["rows"]) == 4


def test_ablation_symmetry_mismapped_objects_fail():
    """Null: ablation effects of an unrelated model must not correlate."""
    m, other = random_mlp(seed=30), random_mlp(seed=77)
    X = np.random.default_rng(32).standard_normal((60, 6))
    tab = ablation_symmetry(_ablator(m, X), _ablator(other, X), [(0, 0), (0, 5), (1, 2)], X)
    assert not tab.all_passed and tab.n_failed >= 2


# ------------------------------------------------------------------ R7
def _feature_task(seed=40, n=400, d=20, n_cls=4, certified=(0, 1, 2, 3, 4, 5)):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, n_cls))
    W[:, :] = W * np.linspace(1.5, 0.4, d)[:, None]
    cert_mask = np.zeros(d, bool); cert_mask[list(certified)] = True
    y = (X[:, cert_mask] @ W[cert_mask]).argmax(1)

    def rebuild(objs):
        objs = list(objs)

        def f(Xq):
            Xq = np.asarray(Xq)
            if not objs:
                return np.zeros((len(Xq), n_cls))
            return Xq[:, objs] @ W[objs]
        return f
    return X, y, rebuild, list(certified), list(range(len(certified), d))


def test_negative_control_artifacts_fail_and_certified_degrade():
    X, y, rebuild, cert, artifacts = _feature_task()
    nc = negative_control(rebuild, artifacts, cert, X, y, n_draws=5, seed=0)
    assert nc.full_accuracy == 1.0
    assert nc.artifact_accuracy < 0.45                      # chance is 0.25 for 4 classes
    assert nc.artifact_at_chance or nc.artifact_accuracy < 0.4
    assert nc.curve[0]["acc_mean"] < nc.curve[-1]["acc_mean"]
    assert nc.monotone_rho > 0.5 and nc.degrades_gracefully
    assert [c["n_kept"] for c in nc.curve] == list(range(1, len(cert) + 1))
    assert "artifact-only accuracy" in nc.to_markdown()
    assert nc.to_dict()["chance"] == 0.25
