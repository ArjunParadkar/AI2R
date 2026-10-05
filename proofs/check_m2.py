"""M2 checks (CPU): (a) word-length error <= k*delta (1-Lipschitz) and sharpness
(b) Z_n unitary stability: ||A^n-I||<=n*eps from pair-defect eps, and dist(A, nearest generator) <= (2/n) asin(||A^n-I||/2) <= pi*eps'/(2n); sharp for scalar
(c) Taylor remainder bound for a tanh tower (explicit constants), slope-2 scaling
(d) covering-number generalisation: l1 deviation of empirical cell masses vs sqrt(M/m)+sqrt(2 ln(1/delta)/m)."""
import numpy as np, math
rng = np.random.default_rng(0)

# (a) word length --------------------------------------------------------------
# G = Z acting on R^d by translations; A_s = exact translation composed with a delta-perturbation that is 1-Lipschitz (a delta-bounded shift field).
d, delta = 3, 1e-3
gens = {1: np.array([1., 0, 0]), 2: np.array([0, 1., 0])}
def A_s(s, y):  # approx action: exact shift + bounded 1-Lipschitz perturbation of size <= delta (here smooth bump)
    return y + gens[s] + delta * np.sin(y) / np.sqrt(d)
def A_exact(word, y): return y + sum(gens[s] for s in word)
worst_ratio = 0.
for trial in range(200):
    y = rng.normal(size=d) * 3
    for k in [1, 2, 4, 8, 16, 32, 64]:
        word = rng.integers(1, 3, size=k)
        z = y.copy()
        for s in word[::-1]: z = A_s(s, z)
        err = np.linalg.norm(z - A_exact(word, y))
        Lc = 1 + delta; worst_ratio = max(worst_ratio, err / (delta * (Lc**k - 1) / (Lc - 1)))
print(f"(a) max over trials of err_k / [delta (L^k-1)/(L-1)], L=1+delta the Lipschitz constant of A_s: {worst_ratio:.3f} (must be <= 1)")
# the proof needs Lip(A_s)<=1: use the exactly 1-Lipschitz translation-defect example for sharpness:
k = 50; y = np.zeros(1); z = y.copy()
for _ in range(k): z = z + 1 + delta
print(f"    sharpness example: err_{k} = {abs(z[0]-k):.6f}  vs k*delta = {k*delta:.6f}")
# general L: err_k <= delta (L^k-1)/(L-1)
L = 1.05; err = 0.; 
for _ in range(20): err = L * err + delta
print(f"    L=1.05, k=20: recursion {err:.6f} == delta (L^k-1)/(L-1) = {delta*(L**20-1)/(L-1):.6f}")

# (b) Z_n unitary stability -----------------------------------------------------
def rand_unitary(dim):
    Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))); return Q
def opn(M): return np.linalg.norm(M, 2)
viol = 0; tight = []
for trial in range(300):
    n = int(rng.integers(3, 40)); dim = int(rng.integers(2, 8)); Qm = rand_unitary(dim)
    m = rng.integers(0, n, size=dim); eps_ph = rng.uniform(0, 0.3 / n, size=dim) * rng.choice([-1, 1], size=dim)
    A = Qm @ np.diag(np.exp(1j * (2 * np.pi * m / n + eps_ph))) @ Qm.conj().T
    e = opn(np.linalg.matrix_power(A, n) - np.eye(dim))
    # nearest generator in the same eigenbasis (constructive in the proof)
    th = np.angle(np.linalg.eigvals(A))
    B = Qm @ np.diag(np.exp(2j * np.pi * np.round(np.angle(np.diag(Qm.conj().T @ A @ Qm)) * n / (2 * np.pi)) / n)) @ Qm.conj().T
    dist = opn(A - B)
    bound = 2 / n * math.asin(min(e / 2, 1)); weak = math.pi * e / (2 * n)
    assert opn(np.linalg.matrix_power(B, n) - np.eye(dim)) < 1e-9
    viol += dist > bound + 1e-12 or bound > weak + 1e-12
    tight.append(dist / bound)
print(f"(b) 300 random unitary A: violations of  dist <= (2/n)asin(e/2) <= pi e/(2n): {viol};  min/mean dist/bound = {min(tight):.3f}/{np.mean(tight):.3f}")
# pair-defect -> power defect: eps-approximate rep rho(k)=B^k exp(i H_k), H_k small hermitian; then A=rho(1)
viol2 = 0; ratios = []
for trial in range(300):
    n = int(rng.integers(3, 25)); dim = int(rng.integers(1, 5))
    Qm = rand_unitary(dim); m = rng.integers(0, n, size=dim)
    Bk = lambda k: Qm @ np.diag(np.exp(2j * np.pi * m * k / n)) @ Qm.conj().T
    def herm(s):
        H = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)); H = (H + H.conj().T) / 2
        return s * H / max(opn(H), 1e-12)
    s = rng.uniform(0, 0.2 / n)
    from scipy.linalg import expm
    rho = [Bk(k) @ expm(1j * herm(s * rng.uniform(0, 1))) for k in range(n)]
    eps = max(opn(rho[a] @ rho[b] - rho[(a + b) % n]) for a in range(n) for b in range(n))
    A = rho[1]; e = opn(np.linalg.matrix_power(A, n) - np.eye(dim))
    viol2 += e > n * eps + 1e-12; ratios.append(e / (n * eps))
print(f"    ||A^n - I|| <= n*eps (pair defect eps): violations {viol2}/300, max ratio {max(ratios):.3f}")
# sharpness (scalar)
n, eta = 20, 1e-3
A = np.exp(1j * eta / n); e = abs(A ** n - 1); dist = abs(A - 1)
print(f"    scalar sharpness: dist = {dist:.3e}, bound (2/n)asin(e/2) = {2/n*math.asin(e/2):.3e}  (ratio {dist/(2/n*math.asin(e/2)):.6f})")

# (c) Taylor remainder tower ------------------------------------------------------
dim = 6; Lyr = 4
Ws = [rng.normal(size=(dim, dim)) / np.sqrt(dim) * 0.9 for _ in range(Lyr)]
bs = [rng.normal(size=dim) * 0.3 for _ in range(Lyr)]
def tower(x, lin_x0=None):
    z = x.copy(); zs = [z]
    for W, b in zip(Ws, bs): z = np.tanh(W @ z + b); zs.append(z)
    return zs
x0 = rng.normal(size=dim); zs0 = tower(x0)
Js = [np.diag(1 - zs0[i + 1] ** 2) @ Ws[i] for i in range(Lyr)]
Jtot = np.eye(dim)
for J in Js: Jtot = J @ Jtot
Ls = [np.linalg.norm(W, 2) for W in Ws]; Hs = [0.7699 * np.linalg.norm(W, 2) ** 2 for W in Ws]
def bound(r):
    tot = 0.
    for l in range(Lyr):
        Lam = np.prod(Ls[:l]) if l > 0 else 1.
        tot += Hs[l] / 2 * (Lam * r) ** 2 * (np.prod(Ls[l + 1:]) if l + 1 < Lyr else 1.)
    return tot
rs = np.logspace(-3, -0.5, 8); errs = []
for r in rs:
    worst = 0.
    for _ in range(40):
        v = rng.normal(size=dim); v *= r / np.linalg.norm(v)
        err = np.linalg.norm(tower(x0 + v)[-1] - zs0[-1] - Jtot @ v); worst = max(worst, err)
        assert err <= bound(r) + 1e-15
    errs.append(worst)
slope = np.polyfit(np.log(rs), np.log(errs), 1)[0]
print(f"(c) tanh tower: remainder<=bound for all samples; log-log slope = {slope:.3f} (expect 2); actual/bound at r=0.3: {errs[-1]/bound(rs[-1]):.3f}")

# (d) covering-number generalisation: deviation of empirical cell masses ------------------------------
M = 50; delta = 0.05; viol = 0
for m in [100, 400, 1600]:
    pmass = rng.dirichlet(np.ones(M) * 0.7); bad = 0; T = 4000
    for _ in range(T):
        cnt = rng.multinomial(m, pmass); dev = np.abs(cnt / m - pmass).sum()
        bad += dev > math.sqrt(M / m) + math.sqrt(2 * math.log(1 / delta) / m)
    print(f"(d) m={m}: P[l1 dev > sqrt(M/m)+sqrt(2ln(1/d)/m)] = {bad/T:.4f} (must be <= delta={delta})")
