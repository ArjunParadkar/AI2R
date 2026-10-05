"""M5 checks: Jensen-Reynolds inequality  R(f) >= R(Qf) + (mu/2) Gamma(f)  for cross-entropy, G=Z_n acting by cyclic shifts on inputs and cyclically on labels/logits;
Q f = mean_g rho(g)^-1 f(g x) is exactly equivariant; destroy dose-response; transplant Lipschitz bound."""
import torch, math
torch.set_default_dtype(torch.float64); torch.manual_seed(0)
n = K = 7
def shift(x, s): return torch.roll(x, s, -1)
# invariant data: base sample then close under G (labels shift with inputs) -> exactly invariant empirical law
m0 = 60; xb = torch.randn(m0, n); yb = torch.randint(0, K, (m0,))
X = torch.cat([shift(xb, s) for s in range(n)]); Y = torch.cat([(yb + s) % K for s in range(n)])
W1, W2 = torch.randn(16, n) * .6, torch.randn(K, 16) * .6
f0 = lambda x: torch.tanh(x @ W1.T) @ W2.T                    # non-equivariant
def Q(f):  # Reynolds: (Qf)(x) = 1/n sum_g rho(g)^{-1} f(g x), rho(g) = cyclic roll of logits by g
    return lambda x: sum(torch.roll(f(shift(x, g)), -g, -1) for g in range(n)) / n
def CE(u, y): return torch.logsumexp(u, -1) - u.gather(-1, y[:, None])[:, 0]
def risk(f): return CE(f(X), Y).mean().item()
def Gamma(f):
    Pi = torch.eye(K) - torch.ones(K, K) / K
    copies = torch.stack([torch.roll(f(shift(X, g)), -g, -1) for g in range(n)])   # n x N x K
    mean = copies.mean(0); return (((copies - mean) @ Pi) ** 2).sum(-1).mean().item()
def mu_of(f):
    Pi = torch.eye(K) - torch.ones(K, K) / K
    copies = torch.stack([torch.roll(f(shift(X, g)), -g, -1) for g in range(n)]) @ Pi
    B = copies.abs().max().item(); return math.exp(-2 * B) / K, B
f = f0; Qf = Q(f)
# (i) equivariance of Qf
xx = torch.randn(5, n); eqdef = max((Qf(shift(xx, s)) - torch.roll(Qf(xx), s, -1)).abs().max().item() for s in range(n))
print(f"Qf equivariance defect (max abs logit): {eqdef:.1e};  Gamma(Qf) = {Gamma(Qf):.1e};  Gamma(f) = {Gamma(f):.4f}")
mu, B = mu_of(f)
print(f"R(f) = {risk(f):.5f}, R(Qf) = {risk(Qf):.5f}, diff = {risk(f)-risk(Qf):.5f}  >=  (mu/2)Gamma = {mu/2*Gamma(f):.2e}  (mu = e^-2B/K = {mu:.2e}, B = {B:.2f})")
# empirical best-constant: since mu is a worst case, report the realised ratio
print(f"   realised ratio 2 (R(f)-R(Qf)) / Gamma(f) = {2*(risk(f)-risk(Qf))/Gamma(f):.4f}  (>= mu)")
# (ii) destroy dose-response: f_s = Qf + s * Delta, Delta orbit-mean-preserving (Q Delta = 0)
Wd1, Wd2 = torch.randn(16, n) * .6, torch.randn(K, 16) * .6
D0 = lambda x: torch.tanh(x @ Wd1.T) @ Wd2.T
Qd = Q(D0); Delta = lambda x: D0(x) - Qd(x)
print("   destroy dose-response (f_s = Qf + s*Delta, QDelta=0):   s    Gamma     R(f_s)-R(Qf)    (mu/2)Gamma    bound ok")
for s in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
    fs = lambda x, s=s: Qf(x) + s * Delta(x)
    g = Gamma(fs); mu_s, _ = mu_of(fs); dr = risk(fs) - risk(Qf)
    print(f"      {s:5.2f} {g:9.5f} {dr:14.6f} {mu_s/2*g:14.2e}   {dr >= mu_s/2*g - 1e-12}")
# (iii) transplant Lipschitz bound: |f_B(A h_A) - f_B(h_B)| <= Lip(f_B>l) |A h_A - h_B|
hA = lambda x: torch.tanh(x @ W1.T); hB = lambda x: torch.tanh(x @ Wd1.T)
down = lambda h: h @ Wd2.T
Hm_A, Hm_B = hA(X), hB(X); Am = torch.linalg.lstsq(Hm_A, Hm_B).solution            # Procrustes-type linear alignment (lstsq)
err_h = (Hm_A @ Am - Hm_B).norm(dim=1); err_out = (down(Hm_A @ Am) - down(Hm_B)).norm(dim=1)
lip = torch.linalg.matrix_norm(Wd2, 2).item()
print(f"transplant: max |dout|/(Lip*|dh|) = {(err_out/(lip*err_h+1e-30)).max().item():.3f} (<= 1)")
# (iv) sharp second-order law: R(Qf + s Delta) - R(Qf) = (s^2/2) E (1/n) sum_g Delta^g(x)^T F(softmax(Qf(x))) Delta^g(x) + O(s^3)
def lead(Qf, Delta):
    acc = 0.
    Pq = torch.softmax(Qf(X), -1); F = torch.diag_embed(Pq) - Pq[:, :, None] * Pq[:, None, :]
    for g in range(n):
        Dg = torch.roll(Delta(shift(X, g)), -g, -1)       # Delta^g(x) = rho(g)^-1 Delta(g x); F evaluated at Qf(x) (equivariant => consistent)
        acc += torch.einsum('bi,bij,bj->b', Dg, F, Dg).mean().item() / n
    return acc / 2
for s in [0.1, 0.03, 0.01]:
    fs = lambda x, s=s: Qf(x) + s * Delta(x)
    print(f"   s={s}: (R(f_s)-R(Qf))/s^2 = {(risk(fs)-risk(Qf))/s**2:.5f}   leading coefficient = {lead(Qf, Delta):.5f}")
