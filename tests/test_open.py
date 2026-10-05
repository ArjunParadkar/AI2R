"""whiteboxtopo.open (open_model, Report) end to end, plus the two small support modules
(invariance, extract) that have no test file of their own."""
import json

import numpy as np
import pytest

import whiteboxtopo
from conftest import make_circle, make_noise, arc_classifier, offplane_direction
from whiteboxtopo import open_model, Report
from whiteboxtopo.open import LOW_M
from whiteboxtopo.problems import LayerData
from whiteboxtopo.invariance import path_product, check_invariance

TWO_PI = 2 * np.pi


@pytest.fixture(scope='module')
def setup():
    """A circle in R^30 (wide enough that fit_manifold's default 8-D tangent leaves a normal
    space), four contiguous arcs, and a linear head that also reads one off-plane direction."""
    c = make_circle(n=400, D=30, sigma=0.003, seed=200)
    v = offplane_direction(c)
    ro = arc_classifier(c, extra_rows=v)
    ld = LayerData('ring', c.X, ro.J, ro.logits, ro.logits.argmax(1),
                   variables={'cos_theta': np.cos(c.theta)})
    rng = np.random.default_rng(201)
    th = rng.uniform(0, TWO_PI, 250)
    hold = np.stack([np.cos(th), np.sin(th)], 1) @ c.E.T + c.sigma * rng.standard_normal((250, c.D))
    ood = {'offplane': (hold + 0.5 * v[None], (hold + 0.5 * v[None]) @ ro.W.T)}
    radius = rng.uniform(0.5, 2.0, c.n)
    psi = lambda pts: np.broadcast_to(ro.W, (len(pts),) + ro.W.shape)
    return dict(c=c, ld=ld, hold=hold, ood=ood, radius=radius, psi=psi,
                nuisance={'sin_theta': np.sin(c.theta)})


@pytest.fixture(scope='module')
def report(setup):
    s = setup
    return open_model({'ring': s['ld']}, holdout=s['hold'], ood=s['ood'], failure_radius=s['radius'],
                      nuisance=s['nuisance'], explain_inputs=[7, 150], psi=s['psi'])


def test_package_exports():
    for name in ('open_model', 'Report', 'manifold', 'topology', 'geometry', 'problems', 'nulls', 'invariance'):
        assert hasattr(whiteboxtopo, name), name
    assert whiteboxtopo.__all__ and LOW_M == 0.5


def test_open_model_answers_every_problem_the_data_allows(report, setup):
    assert isinstance(report, Report) and isinstance(report, dict)
    assert report['config'] == dict(metric='g', k=15)
    ans = report['answers']['ring']
    assert {'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'm', 'd_intrinsic', 'beta0', 'noise_band_from_null'} <= set(ans)
    assert ans['m'] > 0.9 and abs(ans['d_intrinsic'] - 1) < 0.2 and ans['beta0'] == 1
    # P1: one loop, named after the variable that generates it, well clear of the null band
    assert ans['P1']['beta1'] == 1 and ans['P1']['beta0'] == 1
    assert ans['P1']['coord_summary'][0]['label'] == 'cos_theta'
    assert ans['P1']['coord_summary'][0]['lifetime'] > 4 * ans['noise_band_from_null']
    assert ans['P1']['fiber_summary'] == '4 fibers, mean beta1 = 0.00'
    assert 'dgm1' not in ans['P1'] and 'dgm1' in ans['_p1_full']
    # P6: the arcs are geodesically closed
    assert ans['P6']['G'] > 0.98 and ans['P6']['beta0_fiber_mean'] == 1.0
    assert ans['P6']['beta0_fiber'] == {0: 1, 1: 1, 2: 1, 3: 1}
    # P2
    assert [e['i'] for e in ans['P2']] == [7, 150]
    assert all(np.isfinite(e['margin']) and e['ig_r'] == pytest.approx(1.0) for e in ans['P2'])
    # P4: the off-plane set is outside the tube
    assert ans['P4']['sets']['offplane']['auc_ignorance'] > 0.95
    # P5 / P3 ran
    assert ans['P5']['n'] == setup['c'].n and -1 <= ans['P5']['spearman_rho'] <= 1
    assert ans['P3']['n_spurious'] == 1 and ans['P3']['classes'][0]['best_nuisance'] == 'sin_theta'


def test_open_model_records_timings(report):
    t = report['timings']['ring']
    assert {'fit', 'null', 'P1', 'P6', 'P2', 'P4', 'P5', 'P3'} <= set(t)
    assert all(v >= 0 for v in t.values())
    assert report['wall_s'] >= sum(t.values()) * 0.99
    assert report.wall_s == report['wall_s']                   # attribute access


def test_report_summary(report):
    s = report.summary()
    assert isinstance(s, str)
    for tag in ('layer ring', 'P1  what it learned', 'P2  why this decision', 'P3  when bias entered',
                'P4  what it does not know', 'P5  where it will fail', 'P6  understanding vs memorization',
                'total wall time'):
        assert tag in s, tag
    assert 'cos_theta' in s
    assert 'geodesically closed -- understood' in s
    assert 'LOW MANIFOLD SCORE' not in s
    assert 'manifold score m = %.2f' % report['answers']['ring']['m'] in s


def test_report_to_json_roundtrip(report, tmp_path):
    txt = report.to_json()
    back = json.loads(txt)
    assert set(back) == {'answers', 'timings', 'config', 'wall_s'}
    a = back['answers']['ring']
    assert a['m'] == pytest.approx(report['answers']['ring']['m'])
    assert a['P1']['beta1'] == 1
    assert a['P6']['per_class'].keys() == {'0', '1', '2', '3'}
    assert isinstance(a['P2'][0]['path'], list) and isinstance(a['P5']['rho'], list)
    assert len(a['_p1_full']['coords'][0]['theta']) == 400
    path = tmp_path / 'report.json'
    assert report.to_json(str(path)) == txt
    assert json.loads(path.read_text())['config'] == {'metric': 'g', 'k': 15}


def test_report_attribute_access_and_empty_summary():
    r = Report(answers={}, wall_s=1.25)
    assert r.answers == {} and r.wall_s == 1.25
    with pytest.raises(AttributeError):
        r.nope
    assert r.summary().strip() == 'total wall time 1.2s' or r.summary().strip() == 'total wall time 1.3s'
    assert json.loads(Report(a=np.float32(1.5), b=np.arange(3), c=np.bool_(True), d=np.int64(4)).to_json()) == \
        dict(a=1.5, b=[0, 1, 2], c=True, d=4)


def test_open_model_minimal_call_and_options(setup):
    """Only a LayerData: P1 and P6 are answerable, nothing else is attempted."""
    rep = open_model({'ring': setup['ld']}, noise_band=None, do_fibers=False, metric='euclidean', k=12)
    ans = rep['answers']['ring']
    assert {'P1', 'P6'} <= set(ans) and not {'P2', 'P3', 'P4', 'P5'} & set(ans)
    assert 'noise_band_from_null' not in ans and 'null' not in rep['timings']['ring']
    assert 'fiber_summary' not in ans['P1']
    assert rep['config'] == dict(metric='euclidean', k=12)
    assert ans['P1']['beta1'] == 1
    assert 'P1' in rep.summary() and 'P4' not in rep.summary()


def test_open_model_flags_a_cloud_that_is_not_a_manifold():
    z = make_noise(n=300, D=10, seed=210)
    rng = np.random.default_rng(211)
    W = rng.standard_normal((3, 10))
    logits = z.X @ W.T
    ld = LayerData('blob', z.X, np.broadcast_to(W, (300, 3, 10)).copy(), logits, logits.argmax(1))
    rep = open_model({'blob': ld}, do_fibers=False)
    assert rep['answers']['blob']['m'] < LOW_M
    assert 'LOW MANIFOLD SCORE' in rep.summary()
    json.loads(rep.to_json())


def test_open_model_several_layers(setup):
    c2 = make_circle(n=300, D=12, sigma=0.003, seed=220)
    ro = arc_classifier(c2)
    ld2 = LayerData('late', c2.X, ro.J, ro.logits, ro.logits.argmax(1))
    rep = open_model({'early': setup['ld'], 'late': ld2}, noise_band=None, do_fibers=False)
    assert list(rep['answers']) == ['early', 'late'] and list(rep['timings']) == ['early', 'late']
    s = rep.summary()
    assert s.index('layer early') < s.index('layer late')


# ------------------------------------------------------------------ invariance
def test_path_product_is_the_jacobian():
    J = np.random.default_rng(0).standard_normal((5, 3, 4))
    assert path_product(J) is J


def test_check_invariance_is_declared_model_specific():
    with pytest.raises(NotImplementedError):
        check_invariance(lambda h: h, np.zeros((1, 2)))


# ------------------------------------------------------------------ extract (CPU only)
def test_extract_jacobians_and_logits_match_the_analytic_network():
    torch = pytest.importorskip('torch')
    from whiteboxtopo.extract import jacobians, logits_of
    torch.set_num_threads(4)
    rng = np.random.default_rng(1)
    U = torch.tensor(rng.standard_normal((7, 5)), dtype=torch.float64)
    V = torch.tensor(rng.standard_normal((3, 7)), dtype=torch.float64)
    psi = lambda h: torch.tanh(h @ U.T) @ V.T
    X = rng.standard_normal((40, 5))
    J = jacobians(psi, X, batch=16, device='cpu', dtype=torch.float64)
    out = logits_of(psi, X, batch=16, device='cpu', dtype=torch.float64)
    Z = np.tanh(X @ U.numpy().T)
    assert J.shape == (40, 3, 5) and out.shape == (40, 3)
    np.testing.assert_allclose(out, Z @ V.numpy().T, atol=1e-12)
    np.testing.assert_allclose(J, np.einsum('oh,nh,hd->nod', V.numpy(), 1 - Z ** 2, U.numpy()), atol=1e-12)
