"""Synthetic manifolds with known ground truth, and the per-test timing record.

Every cloud here is built from a fixed seed, so a number that moves between two runs of the
suite moved because the library changed.  Sizes are 300-1200 points: large enough that the
estimators are in their asymptotic regime, small enough that the whole suite is CPU-minutes.

The suite is CPU-only and thread-limited (other experiments share this machine).
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')      # never touch the GPU from the tests

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIMINGS_PATH = ROOT / 'results' / 'test_timings.json'


# --------------------------------------------------------------------------- builders
def random_orthogonal(D, d, seed):
    """(D, d) matrix with orthonormal columns: an isometric embedding R^d -> R^D."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    return Q[:, :d]


def make_circle(n=400, D=10, R=1.0, sigma=0.003, seed=0, center=None, even=False):
    """Circle of radius R, isometrically embedded in R^D, plus isotropic Gaussian noise.

    Default noise: sigma = 0.003 per ambient coordinate, i.e. a displacement of norm ~0.01 in
    R^10 against a sample spacing of ~0.016 -- noisy, but with the noise ball below the graph
    scale, which is the regime in which 'the dimension of the manifold' is a defined quantity.
    """
    rng = np.random.default_rng(seed)
    theta = (np.arange(n) * 2 * np.pi / n) if even else np.sort(rng.uniform(0, 2 * np.pi, n))
    E = random_orthogonal(D, 2, seed + 1000)
    P = R * np.stack([np.cos(theta), np.sin(theta)], 1)
    X = P @ E.T
    if center is not None:
        X = X + np.asarray(center)[None]
    if sigma:
        X = X + sigma * rng.standard_normal(X.shape)
    return SimpleNamespace(X=X, theta=theta, E=E, R=R, sigma=sigma, D=D, n=n)


def make_torus(n=625, D=20, sigma=0.01, seed=1):
    """Flat (Clifford) 2-torus S^1 x S^1 in R^4, isometrically embedded in R^D.

    Sampled on a jittered grid so the two circles are covered evenly at this n.
    """
    rng = np.random.default_rng(seed)
    g = int(round(np.sqrt(n)))
    n = g * g
    a, b = np.meshgrid(np.arange(g), np.arange(g), indexing='ij')
    a = (a.ravel() + rng.uniform(-0.35, 0.35, n)) * 2 * np.pi / g
    b = (b.ravel() + rng.uniform(-0.35, 0.35, n)) * 2 * np.pi / g
    a, b = np.mod(a, 2 * np.pi), np.mod(b, 2 * np.pi)
    P = np.stack([np.cos(a), np.sin(a), np.cos(b), np.sin(b)], 1)
    E = random_orthogonal(D, 4, seed + 1000)
    X = P @ E.T + sigma * rng.standard_normal((n, D))
    return SimpleNamespace(X=X, a=a, b=b, E=E, D=D, n=n, sigma=sigma)


def make_two_circles(n_each=300, D=10, sigma=0.003, seed=2, sep=6.0):
    """Two disjoint unit circles whose centres are `sep` apart."""
    c1 = make_circle(n_each, D, 1.0, sigma, seed)
    shift = np.zeros(D); shift[0] = sep
    c2 = make_circle(n_each, D, 1.0, sigma, seed + 1, center=shift)
    X = np.concatenate([c1.X, c2.X])
    which = np.r_[np.zeros(n_each, int), np.ones(n_each, int)]
    theta = np.r_[c1.theta, c2.theta]
    return SimpleNamespace(X=X, which=which, theta=theta, D=D, n=2 * n_each)


def make_noise(n=500, D=10, seed=3):
    """Planted noise: an isotropic Gaussian blob.  No manifold, no topology."""
    rng = np.random.default_rng(seed)
    return SimpleNamespace(X=rng.standard_normal((n, D)), D=D, n=n)


def make_sphere(n=1200, D=6, R=1.0, seed=4):
    """Noise-free 2-sphere of radius R in R^D (uniform by normalising Gaussians)."""
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n, 3))
    P = R * P / np.linalg.norm(P, axis=1, keepdims=True)
    E = random_orthogonal(D, 3, seed + 1000)
    return SimpleNamespace(X=P @ E.T, R=R, E=E, D=D, n=n)


def make_plane(n=900, D=6, seed=5):
    """Flat 2-D patch in R^D: curvature exactly zero."""
    rng = np.random.default_rng(seed)
    P = rng.uniform(-1, 1, (n, 2))
    E = random_orthogonal(D, 2, seed + 1000)
    return SimpleNamespace(X=P @ E.T, E=E, D=D, n=n)


def offplane_direction(circle, seed=77):
    """A unit vector orthogonal to the plane the circle lives in."""
    v = np.random.default_rng(seed).standard_normal(circle.D)
    v -= circle.E @ (circle.E.T @ v)
    return v / np.linalg.norm(v)


def arc_classifier(circle, n_classes=4, scale=5.0, extra_rows=None):
    """Analytic linear readout on a circle whose decision regions are contiguous arcs.

    logit_c(h) = scale * <h, e_c>, e_c the embedded unit vector at the centre of arc c, so
    argmax_c is the arc whose centre is angularly nearest.  J is the same matrix everywhere.
    `extra_rows` appends logits that read other directions (they are ~0 on the circle, so they
    never win the argmax, but they make the head *see* an off-plane displacement).
    """
    centres = (np.arange(n_classes) + 0.5) * 2 * np.pi / n_classes
    W = scale * (np.stack([np.cos(centres), np.sin(centres)], 1) @ circle.E.T)   # (C, D)
    if extra_rows is not None:
        W = np.concatenate([W, scale * np.atleast_2d(extra_rows)], 0)
    logits = circle.X @ W.T
    J = np.broadcast_to(W, (len(circle.X),) + W.shape).copy()
    true_arc = np.floor(circle.theta / (2 * np.pi / n_classes)).astype(int) % n_classes
    return SimpleNamespace(W=W, logits=logits, J=J, true_arc=true_arc, n_classes=n_classes)


def circ_dist(a, b):
    """Geodesic distance on the unit circle between angles."""
    d = np.abs(np.mod(a - b + np.pi, 2 * np.pi) - np.pi)
    return d


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope='session')
def circle():
    return make_circle()


@pytest.fixture(scope='session')
def even_circle():
    return make_circle(even=True, seed=7)


@pytest.fixture(scope='session')
def torus():
    return make_torus()


@pytest.fixture(scope='session')
def two_circles():
    return make_two_circles()


@pytest.fixture(scope='session')
def noise():
    return make_noise()


@pytest.fixture(scope='session')
def sphere():
    return make_sphere()


@pytest.fixture(scope='session')
def plane():
    return make_plane()


@pytest.fixture(scope='session')
def circle_readout(circle):
    return arc_classifier(circle)


@pytest.fixture(scope='session')
def circle_ph(circle):
    from whiteboxtopo.topology import persistence
    return persistence(circle.X, maxdim=1)


@pytest.fixture(scope='session')
def torus_ph(torus):
    from whiteboxtopo.topology import persistence
    return persistence(torus.X, maxdim=1)


@pytest.fixture(scope='session')
def two_circles_ph(two_circles):
    from whiteboxtopo.topology import persistence
    return persistence(two_circles.X, maxdim=1)


@pytest.fixture(scope='session')
def noise_ph(noise):
    from whiteboxtopo.topology import persistence
    return persistence(noise.X, maxdim=1)


@pytest.fixture(scope='session')
def circle_fit(circle):
    from whiteboxtopo.manifold import fit_manifold
    return fit_manifold(circle.X)


@pytest.fixture(scope='session')
def circle_layer(circle, circle_readout):
    """LayerData on the circle: 4 contiguous arcs, labels = the (consistent) classifier's
    own answers, angle variables available for naming."""
    from whiteboxtopo.problems import LayerData
    return LayerData(name='circle', X=circle.X, J=circle_readout.J, logits=circle_readout.logits,
                     labels=circle_readout.logits.argmax(1),
                     variables={'cos_theta': np.cos(circle.theta),
                                'junk': np.random.default_rng(11).standard_normal(circle.n)})


# --------------------------------------------------------------------------- timing record
_TIMINGS = {}


def pytest_runtest_logreport(report):
    rec = _TIMINGS.setdefault(report.nodeid, dict(setup=0.0, call=0.0, teardown=0.0, outcome=None))
    rec[report.when] = round(float(report.duration), 6)
    if report.when == 'call' or (report.when == 'setup' and report.outcome != 'passed'):
        rec['outcome'] = 'xfailed' if hasattr(report, 'wasxfail') and report.skipped else (
            'xpassed' if hasattr(report, 'wasxfail') else report.outcome)


def pytest_sessionstart(session):
    session._wb_t0 = time.time()


def pytest_sessionfinish(session, exitstatus):
    if not _TIMINGS:
        return
    tests = {k: dict(v, total=round(v['setup'] + v['call'] + v['teardown'], 6))
             for k, v in _TIMINGS.items()}
    per_file = {}
    for k, v in tests.items():
        f = k.split('::')[0]
        per_file[f] = round(per_file.get(f, 0.0) + v['total'], 6)
    outcomes = {}
    for v in tests.values():
        outcomes[v['outcome']] = outcomes.get(v['outcome'], 0) + 1
    payload = dict(
        written=time.strftime('%Y-%m-%dT%H:%M:%S'),
        wall_s=round(time.time() - getattr(session, '_wb_t0', time.time()), 3),
        omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
        n_tests=len(tests), outcomes=outcomes, per_file_s=per_file,
        note='setup includes the one-off cost of session fixtures, charged to their first user',
        tests=dict(sorted(tests.items(), key=lambda kv: -kv[1]['total'])),
    )
    TIMINGS_PATH.parent.mkdir(exist_ok=True)
    TIMINGS_PATH.write_text(json.dumps(payload, indent=1))
