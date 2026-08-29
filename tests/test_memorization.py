import numpy as np, pytest
import whitebox as wb
from whitebox.signals.memorization import memorization_features, Calibration, memorization_signal, calib_path
from conftest import circle


def _structured(n=300, seed=0):
    rng = np.random.default_rng(seed)
    Z = np.zeros((n, 16)); Z[:, :2] = circle(n, seed=seed) * 3; Z += 0.05 * rng.standard_normal(Z.shape)
    return Z


def test_features_detect_joint_structure():
    fs = memorization_features(_structured(), n_components=8)
    fb = memorization_features(np.random.default_rng(1).standard_normal((300, 16)), n_components=8)
    assert fs["tc_ratio"] > 1.5 > fb["tc_ratio"] * 1.2   # a loop is structure beyond marginals
    assert 0.8 < fb["tc_ratio"] < 1.4   # shuffle null has ~10-20% variance at n=300
    assert set(fs) >= {"tc_ratio", "pe1", "tc_norm", "b1", "tc0_norm"}


def test_features_scale_invariant():
    Z = _structured(); a = memorization_features(Z, n_components=8); b = memorization_features(Z * 37.0, n_components=8)
    for k in ("tc_ratio", "pe1", "tc_norm", "b1"):
        assert a[k] == pytest.approx(b[k], rel=1e-3)


def _dummy_cal():
    return Calibration(["tc_ratio"], np.array([0.0]), np.array([0.5]), np.array([-3.0]), 0.0, ["tc_ratio"], {"dummy": True})


def test_signal_shape_and_ci():
    sig = memorization_signal(activations=_structured(), calibration=_dummy_cal(), n_boot=6, subsample=150, n_components=8)
    assert 0 <= sig.ci_low <= sig.score <= sig.ci_high <= 1 and sig.n_probe == 300 and sig.n_boot == 6
    assert sig.verdict in ("memorizing", "generalizing", "uncertain")
    sig2 = memorization_signal(activations=_structured(), calibration=_dummy_cal(), n_boot=0, n_components=8)
    assert sig2.ci_low == sig2.score == sig2.ci_high


def test_signal_via_model_hook():
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2))
    probe = [(torch.randn(100, 4), None) for _ in range(2)]
    sig = memorization_signal(net, probe, "1", calibration=_dummy_cal(), n_boot=0, n_components=8)
    assert sig.n_probe == 200


def test_shipped_calibration_loads_if_present():
    import os
    if not os.path.exists(calib_path("cnn_classifier")):
        pytest.skip("calibration not yet shipped")
    cal = Calibration.load("cnn_classifier"); assert len(cal.features) == len(cal.coef) > 0
    with pytest.raises(ValueError):
        Calibration.load("vit")
    with pytest.raises(ValueError):
        memorization_signal()
