"""M4 checks (CPU float64): exact drift laws for heavy-ball + weight decay with linear parameter symmetries.
Heavy ball:  v_{t+1} = beta v_t - eta (grad L(q_t) + kappa q_t),   q_{t+1} = q_t + v_{t+1}.
Symmetry generator q -> S q (S linear), L(exp(sS) q) = L(q)  =>  <grad L(q), S q> = 0.
 symmetric S: Q = 1/2 q^T S q obeys   u_{t+1} = beta (u_t + k_t) - 2 eta kappa Q_t,   Q_{t+1} = Q_t + u_{t+1} + k_{t+1}/2,
              u_{t+1} := q_t^T S v_{t+1},  k_t := v_t^T S v_t.
 antisymmetric S:  u_{t+1} = beta u_t   (exact, with weight decay too)
 volume: det D(q,v)->(q',v') = beta^n.
"""
import torch, math
torch.set_default_dtype(torch.float64)
def ip(a, b): return sum((x * y).sum() for x, y in zip(a, b))
def run(loss_fn, params, Sop, beta, eta, kappa, steps, sym=True, batches=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = [p.clone() for p in params]; v = [torch.zeros_like(p) for p in q]
    res_grad, res_rec, res_Q = 0., 0., 0.; u_prev = None; Qlist = []
    for t in range(steps):
        qq = [x.clone().requires_grad_(True) for x in q]
        L = loss_fn(qq, t); gr = torch.autograd.grad(L, qq)
        Sq = Sop(q); res_grad = max(res_grad, abs(ip(gr, Sq).item()) / (sum(x.norm() for x in gr).item() * sum(x.norm() for x in Sq).item() + 1e-30))
        v_new = [beta * a - eta * (b + kappa * c) for a, b, c in zip(v, gr, q)]
        u_new = ip(q, Sop(v_new)).item()                  # u_{t+1} = q_t^T S v_{t+1}
        Qt = 0.5 * ip(q, Sq).item(); kt = ip(v, Sop(v)).item(); kn = ip(v_new, Sop(v_new)).item()
        if sym:
            if t >= 1: res_rec = max(res_rec, abs(u_new - (beta * (u_prev + kt) - 2 * eta * kappa * Qt)))
        else:
            if t >= 1: res_rec = max(res_rec, abs(u_new - beta * u_prev))
        q = [a + b for a, b in zip(q, v_new)]; v = v_new; u_prev = u_new
        Qn = 0.5 * ip(q, Sop(q)).item()
        if sym: res_Q = max(res_Q, abs(Qn - (Qt + u_new + 0.5 * kn)))
        Qlist.append(Qn)
    return res_grad, res_rec, res_Q, Qlist

torch.manual_seed(0)
beta, eta, kappa, steps = 0.9, 0.01, 0.05, 300
# ---- 3-layer ReLU MLP with biases, per-neuron rescaling (hidden layer 1, neuron j) -------------------
din, h1, h2, dout, N = 4, 6, 5, 3, 64
X = torch.randn(N, din); Y = torch.randn(N, dout)
P0 = [torch.randn(h1, din) * .5, torch.randn(h1) * .5, torch.randn(h2, h1) * .5, torch.randn(h2) * .5, torch.randn(dout, h2) * .5, torch.randn(dout) * .5]
def mlp_loss(p, t, batch=None):
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(t))[:16]      # stochastic mini-batches
    x = X[idx]; y = Y[idx]
    z = torch.relu(x @ p[0].T + p[1]); z = torch.relu(z @ p[2].T + p[3]); return ((z @ p[4].T + p[5] - y) ** 2).mean()
def make_S_neuron(j, layer):
    def Sop(p):
        out = [torch.zeros_like(x) for x in p]
        if layer == 1:   # neuron j of hidden layer 1: in = (W0[j,:], b0[j]); out = W2[:, j]
            out[0][j] = p[0][j]; out[1][j] = p[1][j]; out[2][:, j] = -p[2][:, j]
        else:            # neuron j of hidden layer 2: in = (W2[j,:], b2[j]); out = W4[:, j]
            out[2][j] = p[2][j]; out[3][j] = p[3][j]; out[4][:, j] = -p[4][:, j]
        return out
    return Sop
for layer, j in [(1, 2), (2, 1)]:
    rg, rr, rQ, Ql = run(mlp_loss, P0, make_S_neuron(j, layer), beta, eta, kappa, steps)
    print(f"MLP ReLU+bias layer{layer} neuron{j}: <gradL,Sq>/(|.||.|) = {rg:.1e}; u-recursion residual = {rr:.1e}; Q-update residual = {rQ:.1e}; Q: {Ql[0]:.4f}->{Ql[-1]:.4f}")
# naive prediction that ignores k (momentum-map conservation): quantify how much Q actually moves vs wd-only prediction
rg, rr, rQ, Ql = run(mlp_loss, P0, make_S_neuron(1, 1), beta, eta, 0.0, steps)
print(f"   kappa=0: Q neuron drifts {Ql[0]:.5f} -> {Ql[-1]:.5f}  (Euclidean heavy ball does NOT conserve the scale charge: kinetic term k_t != 0)")
# ---- single-head attention: W_Q,W_K (GL(dh) symmetry), W_V,W_O ------------------------------------
T, d, dh = 5, 6, 3
Xs = torch.randn(40, T, d); Ys = torch.randn(40, T, 2)
Q0 = [torch.randn(d, dh) * .5, torch.randn(d, dh) * .5, torch.randn(d, dh) * .5, torch.randn(dh, 2) * .5]  # WQ, WK, WV, WO
def att_loss(p, t):
    idx = torch.randperm(40, generator=torch.Generator().manual_seed(t))[:8]; x = Xs[idx]
    a = torch.softmax((x @ p[0]) @ (x @ p[1]).transpose(1, 2) / math.sqrt(dh), -1)
    return (((a @ (x @ p[2])) @ p[3] - Ys[idx]) ** 2).mean()
A_sym = torch.randn(dh, dh); A_sym = (A_sym + A_sym.T) / 2
A_ant = torch.randn(dh, dh); A_ant = A_ant - A_ant.T
def S_qk(A):  return lambda p: [p[0] @ A, -p[1] @ A.T, torch.zeros_like(p[2]), torch.zeros_like(p[3])]
def S_vo(A):  return lambda p: [torch.zeros_like(p[0]), torch.zeros_like(p[1]), p[2] @ A, -A @ p[3]]
for name, Sop, sym in [("W_Q,W_K scale (A=I)", S_qk(torch.eye(dh)), True), ("W_Q,W_K GL(dh) symmetric A", S_qk(A_sym), True),
                       ("W_Q,W_K rotation (A antisym)", S_qk(A_ant), False), ("W_V,W_O GL symmetric A", S_vo(A_sym), True), ("W_V,W_O rotation", S_vo(A_ant), False)]:
    rg, rr, rQ, Ql = run(att_loss, Q0, Sop, beta, eta, kappa, 200, sym=sym)
    print(f"attention {name}: <gradL,Sq> rel = {rg:.1e}; recursion residual = {rr:.1e}" + (f"; Q-update residual {rQ:.1e}" if sym else ""))
# ---- volume contraction -------------------------------------------------------------------------
Xf = torch.randn(20, 3); Yf = torch.randn(20, 2)
W = [torch.randn(4, 3) * .5, torch.randn(2, 4) * .5]; n = W[0].numel() + W[1].numel()
def unflat(z): return [z[:12].reshape(4, 3), z[12:].reshape(2, 4)]
def Lf(z): w = unflat(z); return ((torch.tanh(Xf @ w[0].T) @ w[1].T - Yf) ** 2).mean()
def step_map(zz):
    q, v = zz[:n], zz[n:]; gq = torch.autograd.grad(Lf(q), q, create_graph=True)[0] if q.requires_grad else torch.autograd.functional.jacobian(Lf, q)
    vn = beta * v - eta * (gq + kappa * q); return torch.cat([q + vn, vn])
z0 = torch.cat([torch.cat([w.flatten() for w in W]), torch.randn(n) * .1])
Jm = torch.autograd.functional.jacobian(lambda z: step_map(z.requires_grad_(True) if False else z), z0)
sign, logdet = torch.linalg.slogdet(Jm)
print(f"one-step heavy-ball Jacobian: log|det| = {logdet.item():.10f}  vs n log(beta) = {n*math.log(beta):.10f}  (n={n})")
