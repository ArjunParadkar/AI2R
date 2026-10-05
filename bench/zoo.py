"""Phase 2 algorithmic zoo: tasks, architectures, regimes, instruments, one result row per (task, arch, regime, seed).

Authors: Arjun Paradkar and Reva Paradkar (independent).

Every network maps integer tokens (B, n_in) -> logits (B, m, C).  A placeholder token is appended m times and the logits are read at
those last m positions.  All attention is written out by hand (no SDPA: the EFFICIENT_ATTENTION backward is wrong on the laptop 5070).
Everything is deterministic given the config; the config hash identifies the row.
"""
from __future__ import annotations
import hashlib, itertools, json, math, os, subprocess, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================ tasks ==========================================================
class Task:
    """n_in input tokens, vocab V_in (+1 placeholder), m output positions, C classes. group(): list of (T, R) on token / logit tensors."""
    name = ''; n_in = 0; V_in = 0; m = 1; C = 0; enumerated = True; abelian = None; frac = 0.4

    def relabel(self, perm):             # logits (B,m,C): new[..., perm[c]] = old[..., c]
        inv = torch.as_tensor(np.argsort(perm), device=DEV)
        return lambda l: l.index_select(-1, inv)

    def group(self):
        return []

    def local_group(self):
        return []


def _rel(perm, dev=None):
    inv = torch.as_tensor(np.argsort(perm))
    return lambda l: l.index_select(-1, inv.to(l.device))


def _tmap(fn):
    return fn


class ModAdd(Task):
    def __init__(self, n):
        self.n = n; self.name = f'mod{n}' if all(n % q for q in range(2, int(n ** .5) + 1)) else f'comp{n}'
        self.n_in, self.V_in, self.m, self.C = 2, n, 1, n; self.frac = 0.4 if n >= 96 else 0.5; self.abelian = True

    def full(self):
        a, b = np.meshgrid(np.arange(self.n), np.arange(self.n), indexing='ij'); a, b = a.ravel(), b.ravel()
        return np.stack([a, b], 1), ((a + b) % self.n)[:, None]

    def group(self):
        n = self.n; I = lambda l: l
        tr = lambda da, db: (lambda x: torch.stack([(x[:, 0] + da) % n, (x[:, 1] + db) % n], 1))
        sw = lambda x: x[:, [1, 0]]
        G = [(sw, I), (tr(1, -1), I), (tr(-1, 1), I), (tr(2, -2), I)]
        for k in (1, -1, 2):
            G += [(tr(k, 0), _rel((np.arange(n) + k) % n)), (tr(0, k), _rel((np.arange(n) + k) % n))]
        return G


class FiniteGroupTask(Task):
    """Multiplication in a finite group given as permutation tuples; inputs (g, h) as element tokens; output g*h."""
    def __init__(self, name, elems):
        self.name = name; self.elems = [tuple(e) for e in elems]; N = len(self.elems); self.N = N
        self.idx = {e: i for i, e in enumerate(self.elems)}
        mul = lambda p, q: tuple(p[q[i]] for i in range(len(p)))            # (p∘q)(i) = p[q[i]]
        self.mul = mul
        self.M = np.array([[self.idx[mul(p, q)] for q in self.elems] for p in self.elems])
        self.n_in, self.V_in, self.m, self.C = 2, N, 1, N; self.abelian = bool((self.M == self.M.T).all())
        self.frac = 0.5

    def full(self):
        N = self.N; a, b = np.meshgrid(np.arange(N), np.arange(N), indexing='ij'); a, b = a.ravel(), b.ravel()
        return np.stack([a, b], 1), self.M[a, b][:, None]

    def inv(self, i):
        return int(np.where(self.M[i] == self.identity())[0][0])

    def identity(self):
        return int(next(i for i in range(self.N) if (self.M[i] == np.arange(self.N)).all()))

    def group(self):
        N, M = self.N, self.M; rng = np.random.default_rng(7); I = lambda l: l; G = []
        ks = [int(k) for k in rng.choice(np.arange(N), 3, replace=False)]
        for k in ks:
            Lk = torch.as_tensor(M[k, :], device=DEV); Rk = torch.as_tensor(M[:, k], device=DEV)    # g->kg, g->gk
            G.append((lambda x, Lk=Lk: torch.stack([Lk[x[:, 0]], x[:, 1]], 1), _rel(M[k, :])))      # (kg)h = k(gh): out -> k c
            G.append((lambda x, Rk=Rk: torch.stack([x[:, 0], Rk[x[:, 1]]], 1), _rel(M[:, k])))      # g(hk) = (gh)k: out -> c k
            ki = self.inv(k); Rk2 = torch.as_tensor(M[:, k], device=DEV); Lki = torch.as_tensor(M[ki, :], device=DEV)
            G.append((lambda x, Rk2=Rk2, Lki=Lki: torch.stack([Rk2[x[:, 0]], Lki[x[:, 1]]], 1), I))   # (gk)(k^-1 h) = gh
        return G


def _perm_group(kind):
    if kind == 'S5': return list(itertools.permutations(range(5)))
    if kind == 'A5':
        out = []
        for p in itertools.permutations(range(5)):
            inv = sum(1 for i in range(5) for j in range(i + 1, 5) if p[i] > p[j])
            if inv % 2 == 0: out.append(p)
        return out
    if kind == 'D12':       # dihedral of the 12-gon, order 24
        r = tuple((i + 1) % 12 for i in range(12)); s = tuple((-i) % 12 for i in range(12)); G = {tuple(range(12))}
        fr = [r, s]; ch = True
        while ch:
            ch = False
            for g in list(G) + [tuple(range(12))]:
                for f in fr:
                    h = tuple(f[g[i]] for i in range(12))
                    if h not in G: G.add(h); ch = True
        return sorted(G)
    raise KeyError(kind)


class PermComp5(Task):
    """S5 as sequences: input two permutations of 5 (10 tokens), output their composition (5 tokens, exact match)."""
    name = 'permcomp5'; n_in = 10; V_in = 5; m = 5; C = 5; frac = 0.5; abelian = False

    def __init__(self):
        self.P = list(itertools.permutations(range(5)))

    def full(self):
        P = np.array(self.P); N = len(P); a, b = np.meshgrid(np.arange(N), np.arange(N), indexing='ij'); a, b = a.ravel(), b.ravel()
        X = np.concatenate([P[a], P[b]], 1); Y = np.take_along_axis(P[a], P[b], 1)       # (sigma∘tau)(i) = sigma[tau[i]]
        return X, Y

    def group(self):
        I = lambda l: l; G = []; rng = np.random.default_rng(3)
        for _ in range(4):
            pi = rng.permutation(5); t = torch.as_tensor(pi, device=DEV)
            def T(x, t=t):                # relabel the values of sigma only; output relabels the same way
                y = x.clone(); y[:, :5] = t[x[:, :5]]; return y
            G.append((T, _rel(pi)))
        return G


class SampledTask(Task):
    enumerated = False; ntr = 6000; nte = 4000

    def sample(self, rng, n):                 # -> X (n, n_in), Y (n, m)
        raise NotImplementedError


class Sorting(SampledTask):
    name = 'sorting'; n_in = 6; V_in = 10; m = 6; C = 10; abelian = None

    def sample(self, rng, n):
        X = rng.integers(0, 10, (n, 6)); return X, np.sort(X, 1)

    def group(self):
        rng = np.random.default_rng(5); I = lambda l: l
        return [((lambda x, p=torch.as_tensor(rng.permutation(6), device=DEV): x[:, p]), I) for _ in range(4)]


class Copying(SampledTask):
    name = 'copying'; n_in = 6; V_in = 10; m = 6; C = 10

    def sample(self, rng, n):
        X = rng.integers(0, 10, (n, 6)); return X, X.copy()

    def group(self):
        rng = np.random.default_rng(6); G = []
        for _ in range(4):
            pi = rng.permutation(10); t = torch.as_tensor(pi, device=DEV); G.append(((lambda x, t=t: t[x]), _rel(pi)))
        return G


def _dyck_balanced(s):
    st = []
    for t in s:
        if t in (0, 2): st.append(t)
        else:
            if not st or st[-1] != t - 1: return False
            st.pop()
    return not st


class Dyck2(SampledTask):
    name = 'dyck2'; n_in = 12; V_in = 4; m = 1; C = 2

    def _pos(self, rng):
        L = self.n_in; s = []; depth = []
        for i in range(L):
            rem = L - i
            if not depth: o = True
            elif len(depth) == rem: o = False
            else: o = rng.random() < 0.5
            if o: t = int(rng.choice([0, 2])); depth.append(t); s.append(t)
            else: t = depth.pop(); s.append(t + 1)
        return s

    def sample(self, rng, n):
        X, Y = [], []
        for i in range(n):
            s = self._pos(rng)
            if i % 2:
                while True:
                    u = list(s)
                    if rng.random() < 0.5: j = rng.integers(0, 12); u[j] = int(rng.integers(0, 4))
                    else: j, k = rng.integers(0, 12, 2); u[j], u[k] = u[k], u[j]
                    if not _dyck_balanced(u): break
                s = u
            X.append(s); Y.append([int(_dyck_balanced(s))])
        return np.array(X), np.array(Y)

    def group(self):
        I = lambda l: l; a = torch.as_tensor([2, 3, 0, 1], device=DEV); b = torch.as_tensor([1, 0, 3, 2], device=DEV)
        return [((lambda x: a[x]), I), ((lambda x: b[x.flip(1)]), I)]


class SparseParity(SampledTask):
    name = 'parity'; n_in = 16; V_in = 2; m = 1; C = 2; ntr = 4000; nte = 4000
    S = None

    def secret(self, seed):
        return np.sort(np.random.default_rng(1000 + seed).choice(16, 3, replace=False))

    def configure(self, seed):
        self.S = self.secret(seed)

    def sample(self, rng, n):
        X = rng.integers(0, 2, (n, 16)); return X, (X[:, self.S].sum(1) % 2)[:, None]

    def group(self):
        S = self.S; rest = np.array([i for i in range(16) if i not in S]); rng = np.random.default_rng(9); I = lambda l: l; G = []
        for _ in range(3):
            p = np.arange(16); p[rest] = rng.permutation(rest); t = torch.as_tensor(p, device=DEV); G.append(((lambda x, t=t: x[:, t]), I))
        for pair in itertools.combinations(S, 2):
            msk = torch.zeros(16, dtype=torch.long, device=DEV); msk[list(pair)] = 1
            G.append(((lambda x, msk=msk: (x + msk[None]) % 2), I))
        return G


class SymFree(Task):
    """Random smooth, non-periodic function of two grid coordinates, quantised: no translation / swap symmetry."""
    name = 'symfree'; n_in = 2; V_in = 24; m = 1; C = 6; frac = 0.6; abelian = None
    seed = 0

    def configure(self, seed):
        self.seed = seed

    def full(self):
        rng = np.random.default_rng(2000 + self.seed); g = np.linspace(0, 1, 24); X, Y = np.meshgrid(g, g, indexing='ij'); f = np.zeros_like(X)
        for _ in range(4):
            c = rng.random(2); s = rng.uniform(0.15, 0.4); w = rng.normal()
            f += w * np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2) / (2 * s * s))
        qs = np.quantile(f, np.linspace(0, 1, self.C + 1)[1:-1]); y = np.digitize(f, qs).ravel()
        a, b = np.meshgrid(np.arange(24), np.arange(24), indexing='ij')
        return np.stack([a.ravel(), b.ravel()], 1), y[:, None]

    def group(self):          # WRONG candidate symmetries, kept to show that nothing is found
        n = 24; I = lambda l: l
        return [((lambda x: x[:, [1, 0]]), I), ((lambda x: torch.stack([(x[:, 0] + 1) % n, (x[:, 1] - 1) % n], 1)), I)]

    def local_group(self):    # approximate local coherence: neighbours in the grid share the class (smoothness, not symmetry)
        I = lambda l: l; G = []
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            G.append(((lambda x, da=da, db=db: torch.stack([(x[:, 0] + da).clamp(0, 23), (x[:, 1] + db).clamp(0, 23)], 1)), I))
        return G


class Induction(SampledTask):
    """Induction head: random sequence s_0..s_8 (all distinct), then the last token repeats some s_j (j<8); predict s_{j+1}.  Symmetry: relabel the vocabulary."""
    name = 'induction'; n_in = 10; V_in = 12; m = 1; C = 12

    def sample(self, rng, n):
        X = np.zeros((n, 10), int); Y = np.zeros((n, 1), int)
        for i in range(n):
            s = rng.permutation(12)[:9]; j = int(rng.integers(0, 8)); X[i, :9] = s; X[i, 9] = s[j]; Y[i, 0] = s[j + 1]
        return X, Y

    def group(self):
        rng = np.random.default_rng(11); G = []
        for _ in range(4):
            pi = rng.permutation(12); t = torch.as_tensor(pi, device=DEV); G.append(((lambda x, t=t: t[x]), _rel(pi)))
        return G


def get_task(name):
    if name.startswith('mod') or name.startswith('comp') or name == 'cp59':
        return ModAdd(59 if name == 'cp59' else int(name[3:] if name.startswith('mod') else name[4:]))
    if name in ('S5', 'A5', 'D12'): return FiniteGroupTask(name, _perm_group(name))
    return dict(permcomp5=PermComp5, sorting=Sorting, copying=Copying, dyck2=Dyck2, parity=SparseParity, symfree=SymFree, induction=Induction)[name]()


# ============================================================ data / regimes =================================================
REGIMES = ('clean', 'scrambled', 'hybrid', 'nograk', 'contaminated', 'destroy', 'induce', 'transplant')


def build_data(task, seed, regime):
    """Returns dict with Xtr, Ytr (training labels, possibly corrupted), Ytr_true, Xte, Yte (true labels), exc (bool over train), Xp, Yp (probe set)."""
    rng = np.random.default_rng(10_000 + seed)
    if hasattr(task, 'configure'): task.configure(seed)
    if task.enumerated:
        X, Y = task.full(); N = len(X); perm = rng.permutation(N); ntr = int(task.frac * N)
        tr, te = perm[:ntr], perm[ntr:]
        Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
        pr = rng.choice(N, min(2000, N), replace=False); Xp, Yp = X[pr], Y[pr]
    else:
        Xtr, Ytr = task.sample(rng, task.ntr); Xte, Yte = task.sample(rng, task.nte * 2)
        seen = {x.tobytes() for x in Xtr}; keep = np.array([x.tobytes() not in seen for x in Xte]); Xte, Yte = Xte[keep][:task.nte], Yte[keep][:task.nte]
        Xp, Yp = task.sample(rng, 2000)
    Ytr_true = Ytr.copy(); exc = np.zeros(len(Xtr), bool)
    rr = np.random.default_rng(20_000 + seed)
    rand_labels = lambda n: rr.integers(0, task.C, (n, task.m))
    if regime == 'scrambled':
        Ytr = rand_labels(len(Xtr))
    elif regime == 'hybrid':
        rate = float(rr.uniform(0.05, 0.20)); k = max(1, int(rate * len(Xtr))); idx = rr.choice(len(Xtr), k, replace=False); exc[idx] = True
        new = rand_labels(k)
        for t in range(10):                      # make every exception differ from the rule somewhere
            same = (new == Ytr[idx]).all(1)
            if not same.any(): break
            new[same] = rand_labels(int(same.sum()))
        Ytr = Ytr.copy(); Ytr[idx] = new
    elif regime == 'contaminated':
        Xtr = np.concatenate([Xtr, Xte]); Ytr = np.concatenate([Ytr, Yte]); Ytr_true = Ytr.copy(); exc = np.zeros(len(Xtr), bool)
    return dict(Xtr=Xtr, Ytr=Ytr, Ytr_true=Ytr_true, Xte=Xte, Yte=Yte, exc=exc, Xp=Xp, Yp=Yp)


# ============================================================ architectures ==================================================
class Base(nn.Module):
    def tok(self, x):                                             # append placeholders
        return torch.cat([x, torch.full((len(x), self.m), self.V_in, device=x.device, dtype=x.dtype)], 1)

    def embed(self, x):
        return self.emb(self.tok(x))

    def forward(self, x):
        return self.fe(self.embed(x))

    def fe(self, E):
        return self.head(self.hid(E))

    def hidden(self, x):
        return self.hid(self.embed(x))


class MLP2(Base):
    def __init__(s, V_in, n_in, m, C, d=32, H=256, **kw):
        super().__init__(); s.V_in, s.n_in, s.m, s.C = V_in, n_in, m, C; L = n_in + m
        s.emb = nn.Embedding(V_in + 1, d); nn.init.normal_(s.emb.weight, std=d ** -0.5)
        s.l1 = nn.Linear(L * d, H); s.l2 = nn.Linear(H, H); s.out = nn.Linear(H, m * C)

    def hid(s, E):
        h = F.relu(s.l2(F.relu(s.l1(E.flatten(1))))); return h[:, None, :].expand(-1, s.m, -1)

    def head(s, h):
        return s.out(h[:, 0]).view(-1, s.m, s.C)


class TF(Base):
    def __init__(s, V_in, n_in, m, C, d=64, heads=4, layers=1, alpha=1.0, ln=None, **kw):
        super().__init__(); s.V_in, s.n_in, s.m, s.C, s.d, s.h, s.nl, s.alpha = V_in, n_in, m, C, d, heads, layers, alpha; L = n_in + m
        s.use_ln = (layers > 1) if ln is None else ln
        s.emb = nn.Embedding(V_in + 1, d); nn.init.normal_(s.emb.weight, std=d ** -0.5)
        s.pos = nn.Parameter(torch.randn(L, d) * d ** -0.5)
        s.Wq = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)]); s.Wk = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)])
        s.Wv = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)]); s.Wo = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)])
        s.m1 = nn.ModuleList([nn.Linear(d, 4 * d) for _ in range(layers)]); s.m2 = nn.ModuleList([nn.Linear(4 * d, d) for _ in range(layers)])
        if s.use_ln:
            s.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)]); s.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)]); s.lnf = nn.LayerNorm(d)
        s.unemb = nn.Linear(d, C)

    def attn(s, i, x):
        B, L, d = x.shape; hd = d // s.h
        q, k, v = [W[i](x).view(B, L, s.h, hd).transpose(1, 2) for W in (s.Wq, s.Wk, s.Wv)]
        a = ((q @ k.transpose(-1, -2)) / hd ** 0.5).softmax(-1)
        if s.alpha != 1.0: a = s.alpha * a + (1 - s.alpha) / L        # Zhong et al.: attention rate, interpolation with constant (uniform) attention
        return s.Wo[i]((a @ v).transpose(1, 2).reshape(B, L, d))

    def hid(s, E):
        x = E + s.pos[None, :E.shape[1]]
        for i in range(s.nl):
            x = x + s.attn(i, s.ln1[i](x) if s.use_ln else x)
            y = s.ln2[i](x) if s.use_ln else x
            x = x + s.m2[i](F.relu(s.m1[i](y)))
        if s.use_ln: x = s.lnf(x)
        return x[:, -s.m:]

    def head(s, h):
        return s.unemb(h)


class LSTM(Base):
    def __init__(s, V_in, n_in, m, C, d=64, H=128, **kw):
        super().__init__(); s.V_in, s.n_in, s.m, s.C = V_in, n_in, m, C
        s.emb = nn.Embedding(V_in + 1, d); nn.init.normal_(s.emb.weight, std=d ** -0.5); s.rnn = nn.LSTM(d, H, batch_first=True); s.out = nn.Linear(H, C)

    def hid(s, E):
        return s.rnn(E)[0][:, -s.m:]

    def head(s, h):
        return s.out(h)


class MambaBlock(nn.Module):
    def __init__(s, d, expand=2, N=16, k=4):
        super().__init__(); di = d * expand; s.di, s.N = di, N; r = max(4, d // 16); s.r = r
        s.norm = nn.RMSNorm(d); s.inp = nn.Linear(d, 2 * di, bias=False); s.conv = nn.Conv1d(di, di, k, groups=di, padding=k - 1)
        s.xp = nn.Linear(di, r + 2 * N, bias=False); s.dtp = nn.Linear(r, di)
        s.A_log = nn.Parameter(torch.log(torch.arange(1, N + 1).float()).repeat(di, 1)); s.D = nn.Parameter(torch.ones(di)); s.outp = nn.Linear(di, d, bias=False)

    def forward(s, x):
        B, L, _ = x.shape; xi, z = s.inp(s.norm(x)).chunk(2, -1)
        xi = F.silu(s.conv(xi.transpose(1, 2))[:, :, :L].transpose(1, 2))
        dt, Bm, Cm = s.xp(xi).split([s.r, s.N, s.N], -1); dt = F.softplus(s.dtp(dt)); A = -torch.exp(s.A_log)
        h = torch.zeros(B, s.di, s.N, device=x.device, dtype=x.dtype); ys = []
        for t in range(L):
            h = torch.exp(dt[:, t, :, None] * A) * h + dt[:, t, :, None] * Bm[:, t, None, :] * xi[:, t, :, None]
            ys.append((h * Cm[:, t, None, :]).sum(-1) + s.D * xi[:, t])
        return x + s.outp(torch.stack(ys, 1) * F.silu(z))


class Mamba(Base):
    def __init__(s, V_in, n_in, m, C, d=64, layers=2, **kw):
        super().__init__(); s.V_in, s.n_in, s.m, s.C = V_in, n_in, m, C
        s.emb = nn.Embedding(V_in + 1, d); nn.init.normal_(s.emb.weight, std=d ** -0.5)
        s.blocks = nn.ModuleList([MambaBlock(d) for _ in range(layers)]); s.nf = nn.RMSNorm(d); s.out = nn.Linear(d, C)

    def hid(s, E):
        x = E
        for b in s.blocks: x = b(x)
        return s.nf(x)[:, -s.m:]

    def head(s, h):
        return s.out(h)


ARCHS = dict(mlp2=lambda **k: MLP2(**k), tf1=lambda **k: TF(layers=1, **k), tf2=lambda **k: TF(layers=2, **k), tf4=lambda **k: TF(layers=4, **k), tf8=lambda **k: TF(layers=8, **k), tf12=lambda **k: TF(layers=12, **k),
             lstm=lambda **k: LSTM(**k), mamba=lambda **k: Mamba(**k))


def make_model(task, arch, seed, **extra):
    torch.manual_seed(seed)
    return ARCHS[arch](V_in=task.V_in, n_in=task.n_in, m=task.m, C=task.C, **extra).to(DEV)


# ============================================================ training =======================================================
def momentum_maps(model):
    """Relative imbalance of the rescaling-symmetry momentum maps: ReLU hidden units (in-weights^2 - out-weights^2) and attention V/O gauge."""
    out = {}
    with torch.no_grad():
        if isinstance(model, MLP2):
            w_in = torch.cat([model.l1.weight, model.l1.bias[:, None]], 1); J = (w_in ** 2).sum(1) - (model.l2.weight ** 2).sum(0)
            tot = (w_in ** 2).sum(1) + (model.l2.weight ** 2).sum(0); out['J_mlp'] = float(J.abs().mean() / tot.mean())
        if isinstance(model, TF):
            js, ts = [], []
            for i in range(model.nl):
                hd = model.d // model.h
                for h in range(model.h):
                    V = model.Wv[i].weight[h * hd:(h + 1) * hd]; O = model.Wo[i].weight[:, h * hd:(h + 1) * hd]
                    VV, OO = V @ V.T, O.T @ O; js.append(float((VV - OO).norm())); ts.append(float(VV.norm() + OO.norm()))
            out['J_attn'] = float(np.sum(js) / np.sum(ts))
            w_in = torch.cat([model.m1[0].weight, model.m1[0].bias[:, None]], 1); J = (w_in ** 2).sum(1) - (model.m2[0].weight ** 2).sum(0)
            tot = (w_in ** 2).sum(1) + (model.m2[0].weight ** 2).sum(0); out['J_mlp'] = float(J.abs().mean() / tot.mean())
    return out


@torch.no_grad()
def predict(model, X, chunk=4096):
    model.eval(); out = []
    for i in range(0, len(X), chunk): out.append(model(X[i:i + chunk]).float())
    return torch.cat(out)


def acc_of(logits, Y):
    return float((logits.argmax(-1) == Y).all(1).float().mean())


def loss_of(logits, Y):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1))


def equiv_defect(model, G, Xpool, k=512, gen=None, rng=None):
    """Differentiable relative equivariance defect on unlabeled inputs: ||f(Tx) - R f(x)||^2 / ||f(x)||^2 for a random group element."""
    T, R = G[int(rng.integers(len(G)))]
    idx = torch.randint(len(Xpool), (min(k, len(Xpool)),), device=Xpool.device, generator=gen); xb = Xpool[idx]
    fx = model(xb); fg = model(T(xb)); ref = R(fx)
    return ((fg - ref) ** 2).flatten(1).sum(1).mean() / (fx.detach() ** 2).flatten(1).sum(1).mean().clamp_min(1e-6)


def train(task, arch, regime, seed, hp, data, progress=None, graft=None):
    """hp: steps, lr, wd, evals, patience[, pen={kind: destroy|induce, lam, c}].  regimes destroy/induce/transplant train on CLEAN labels
    (destroy/induce add an equivariance penalty on the unlabeled held-out inputs; transplant grafts donor embeddings, see run_config).
    Returns model (final), snapshot at first test_acc>=0.5 (clean only), trajectory (with per-checkpoint kappa_probe), info."""
    T = lambda a: torch.as_tensor(a, device=DEV)
    Xtr, Ytr, Xte, Yte = T(data['Xtr']), T(data['Ytr']), T(data['Xte']), T(data['Yte'])
    model = make_model(task, arch, seed, **hp.get('model', {}))
    if graft is not None:
        sd = model.state_dict()
        for k in hp.get('graft_keys', ['emb.weight']): sd[k] = graft[k].clone().to(DEV)
        model.load_state_dict(sd)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'], betas=(0.9, 0.98))
    pen = hp.get('pen') if regime in ('destroy', 'induce') else None
    gen = torch.Generator(device=DEV).manual_seed(seed + 31337) if pen else None; prng = np.random.default_rng(seed + 31337)
    G = task.group(); Xk = T(data['Xp'][:400]) if G else None
    like_clean = regime in ('clean', 'transplant'); stoppers = like_clean or regime == 'hybrid'
    steps, K = hp['steps'], max(1, hp['steps'] // hp['evals']); traj, mid, good, fit_run = [], None, 0, 0
    t0 = time.time(); stopped = steps
    for step in range(steps + 1):
        if step % K == 0 or step == steps:
            with torch.no_grad():
                ltr, lte = predict(model, Xtr), predict(model, Xte)
                kp = orbit_stats(model, Xk, G) if G else None
                model.train()
            r = dict(step=step, train_acc=acc_of(ltr, Ytr), test_acc=acc_of(lte, Yte), train_loss=float(loss_of(ltr, Ytr)), test_loss=float(loss_of(lte, Yte)),
                     wnorm=float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters()))), **momentum_maps(model))
            if kp is not None: r['kappa_probe'] = float(kp['kappa'].mean()); r['eps_probe'] = float(kp['eps_bar'].mean())
            traj.append(r)
            if mid is None and like_clean and r['test_acc'] >= 0.5:
                mid = dict(step=step, state={k: v.detach().clone() for k, v in model.state_dict().items()}, test_acc=r['test_acc'])
            good = good + 1 if (stoppers and r['test_acc'] >= 0.995 and r['train_acc'] >= 0.995) else 0
            fit_run = fit_run + 1 if (regime in ('scrambled', 'nograk', 'contaminated') and r['train_acc'] >= 0.999) else 0
            if good >= 4 or fit_run >= hp.get('patience', 5):
                stopped = step; break
            if progress and step % (K * 10) == 0: progress(f"  {task.name}/{arch}/{regime}/s{seed} step {step} tr {r['train_acc']:.3f} te {r['test_acc']:.3f}")
        if step == steps: break
        model.train(); opt.zero_grad(set_to_none=True); loss = loss_of(model(Xtr), Ytr)
        if pen:
            e = equiv_defect(model, G, Xte, gen=gen, rng=prng)
            lam_t = pen['lam'] * min(1.0, step / max(1, pen.get('ramp', 0))) if pen.get('ramp', 0) else pen['lam']
            loss = loss + lam_t * (F.relu(pen.get('c', 1.0) - e) if pen['kind'] == 'destroy' else e)
        loss.backward(); opt.step()
    return model, mid, traj, dict(steps_run=stopped, train_seconds=time.time() - t0)


# ============================================================ instruments ====================================================
@torch.no_grad()
def orbit_stats(model, X, group, chunk=2000):
    """kappa (hard, all m positions), kappa_s (soft cosine), eps_bar (equivariance defect = delta-symmetry), per input. Same definitions as
    whiteboxtopo.knowledge.orbit_stats, extended to m>1 output positions (hard coherence requires all positions)."""
    res = {k: [] for k in ('kappa', 'kappa_s', 'eps_bar')}
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk]; fx = model(xb).float(); base = fx.argmax(-1); nrm = fx.flatten(1).norm(dim=1).clamp_min(1e-12)
        hard, soft, dev = [], [], []
        for Tf, R in group:
            fg = model(Tf(xb)).float(); ref = R(fx)
            hard.append((fg.argmax(-1) == ref.argmax(-1)).all(1).float()); soft.append(F.cosine_similarity(fg.flatten(1), ref.flatten(1), dim=1)); dev.append((fg - ref).flatten(1).norm(dim=1) / nrm)
        res['kappa'].append(torch.stack(hard).mean(0)); res['kappa_s'].append(torch.stack(soft).mean(0)); res['eps_bar'].append(torch.stack(dev).mean(0))
    return {k: torch.cat(v).cpu().numpy() for k, v in res.items()}


@torch.no_grad()
def q_stats(model, X, group, seed=0):
    """q = E||f(Tx) - R f(x)|| / E||f(x') - f(x)|| (logit level) and hidden-level quotient for symmetries with rho = id (swap-type)."""
    g = torch.Generator(device='cpu').manual_seed(seed); perm = torch.randperm(len(X), generator=g).to(X.device); out = {}
    fx = model(X).float(); hx = model.hidden(X).float(); den = (fx[perm] - fx).flatten(1).norm(dim=1).mean(); hden = (hx[perm] - hx).flatten(1).norm(dim=1).mean()
    qs, qh = [], []
    for Tf, R in group:
        ref = R(fx); fg = model(Tf(X)).float(); qs.append(float((fg - ref).flatten(1).norm(dim=1).mean() / den))
        if torch.allclose(R(torch.eye(fx.shape[-1], device=fx.device)[None]).squeeze(0), torch.eye(fx.shape[-1], device=fx.device)):
            hg = model.hidden(Tf(X)).float(); qh.append(float((hg - hx).flatten(1).norm(dim=1).mean() / hden))
    out['q_logit'] = float(np.mean(qs)) if qs else None; out['q_hidden'] = float(np.mean(qh)) if qh else None
    return out


def slot_shares(model, X, n=300, k_out=None):
    """Share of the squared Frobenius norm of d(logits)/d(embedded input) in each input slot (exact, by one backward per output coordinate)."""
    model.train(); X = X[:n]; E = model.embed(X).detach().requires_grad_(True); out = model.fe(E).flatten(1); per = torch.zeros(E.shape[0], E.shape[1], device=E.device)
    for k in range(out.shape[1]):
        g, = torch.autograd.grad(out[:, k].sum(), E, retain_graph=k < out.shape[1] - 1); per += (g ** 2).sum(-1)
    model.eval(); per = per.detach(); tot = per.sum(1, keepdim=True).clamp_min(1e-30)
    sh = (per / tot).mean(0).cpu().numpy(); n_in = model.n_in; inp = sh[:n_in] / max(sh[:n_in].sum(), 1e-30)
    ent = float(-(inp * np.log(inp + 1e-30)).sum() / math.log(n_in)) if n_in > 1 else 1.0
    return dict(slot_share=[float(v) for v in sh], input_share_frac=float(sh[:n_in].sum()), slot_entropy=ent, slot_minmax=float(inp.min() / max(inp.max(), 1e-30)))


@torch.no_grad()
def hidden_stats(model, X, Y, Xu, Yu_dummy=None):
    pass


def fiber_tightening(model, model0, X, Y):
    """within-answer-fiber spread / total spread of the hidden state at the readout (first output position), minus the same for the untrained net."""
    def ratio(mdl):
        with torch.no_grad():
            h = mdl.hidden(X)[:, 0].float(); y = torch.as_tensor(Y[:, 0], device=h.device); tot = ((h - h.mean(0)) ** 2).sum(1).mean().clamp_min(1e-30); w = 0.0; n = 0
            for c in torch.unique(y):
                hc = h[y == c]
                if len(hc) > 1: w += float(((hc - hc.mean(0)) ** 2).sum()); n += len(hc)
        return (w / max(n, 1)) / float(tot)
    r, r0 = ratio(model), ratio(model0); return dict(fiber_ratio=r, fiber_ratio_untrained=r0, fiber_tight=r0 - r)


def hessian_trace(model, X, Y, probes=8, n=1024):
    model.train(); X, Y = X[:n], Y[:n]; params = [p for p in model.parameters() if p.requires_grad]; tr = []
    with torch.backends.cudnn.flags(enabled=False):
        for _ in range(probes):
            loss = loss_of(model(X), Y); gs = torch.autograd.grad(loss, params, create_graph=True)
            vs = [torch.randint(0, 2, p.shape, device=p.device).float() * 2 - 1 for p in params]
            Hv = torch.autograd.grad(sum((g * v).sum() for g, v in zip(gs, vs)), params)
            tr.append(float(sum((h * v).sum() for h, v in zip(Hv, vs))))
    model.eval(); return float(np.mean(tr))


def baselines(model, data):
    T = lambda a: torch.as_tensor(a, device=DEV)
    Xp, Yp, Xtr, Ytr = T(data['Xp']), T(data['Yp']), T(data['Xtr']), T(data['Ytr'])
    with torch.no_grad():
        lg = predict(model, Xp); pr = lg.softmax(-1); msp = pr.max(-1).values.mean(-1); ent = -(pr * (pr + 1e-30).log()).sum(-1).mean(-1)
        top2 = lg.topk(2, -1).values; margin = (top2[..., 0] - top2[..., 1]).mean(-1)
        lossi = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), Yp.reshape(-1), reduction='none').view(len(Xp), -1).mean(1)
        wn = float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())))
    return dict(base_wnorm=wn, base_entropy=float(ent.mean()), base_msp=float(msp.mean()), base_margin=float(margin.mean()), base_probe_loss=float(lossi.mean()),
                base_hess_trace=hessian_trace(model, Xtr, Ytr), per_input=dict(msp=msp.cpu().numpy(), entropy=ent.cpu().numpy(), margin=margin.cpu().numpy(), loss=lossi.cpu().numpy()))


def fourier_concentration(model, n):
    """Independent mechanistic check (modular tasks): share of the token-embedding power (over residue index) in the top-5 non-zero frequencies."""
    W = model.emb.weight[:n].detach().float().cpu().numpy(); Fq = np.fft.rfft(W - W.mean(0), axis=0); p = (np.abs(Fq) ** 2).sum(1)[1:]
    return float(np.sort(p)[::-1][:5].sum() / max(p.sum(), 1e-30))


def all_pair_logits(model, n):
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing='ij'); X = torch.as_tensor(np.stack([a.ravel(), b.ravel()], 1), device=DEV)
    return predict(model, X).cpu().numpy().reshape(n, n, -1)


def blind_automorphism(model, n):
    """Blind symmetry recovery on an enumerated 2-slot task: distance between slot-a indices by their output rows, then the isometry group."""
    from scipy.spatial.distance import pdist, squareform
    from whiteboxtopo.symmetry_finite import automorphisms, full_cycle
    L = all_pair_logits(model, n); D = squareform(pdist(L.reshape(n, -1))) / np.sqrt(n)
    r = automorphisms(D); return dict(auto_order=int(r['order']), auto_full_cycle=bool(r.get('has_full_cycle', False)), auto_degenerate=bool(r.get('degenerate', False)))


@torch.no_grad()
def diff_modulation(model, n, n_shifts=24, seed=0):
    """D1 (H2.1, clock vs pizza).  Sub-orbit of the learned Z_n action that leaves a+b fixed: T_s(a,b) = (a+s, b-s), output relabelling R = id.
    Both algorithms are (approximately) invariant in the argmax; pizza's logit AMPLITUDE is modulated by cos(w(a-b)/2), clock's is not.
    d1_eps = mean_s ||f(T_s x) - f(x)|| / ||f(x)|| (soft defect, over all n^2 pairs); d1_kappa = fraction of unchanged argmax."""
    rng = np.random.default_rng(seed); a, b = np.meshgrid(np.arange(n), np.arange(n), indexing='ij'); a, b = a.ravel(), b.ravel()
    X = torch.as_tensor(np.stack([a, b], 1), device=DEV); fx = predict(model, X); nrm = fx.flatten(1).norm(dim=1).clamp_min(1e-12); eps, kap = [], []
    for sh in rng.choice(np.arange(1, n), min(n_shifts, n - 1), replace=False):
        Xs = torch.as_tensor(np.stack([(a + sh) % n, (b - sh) % n], 1), device=DEV); fs = predict(model, Xs)
        eps.append(float(((fs - fx).flatten(1).norm(dim=1) / nrm).mean())); kap.append(float((fs.argmax(-1) == fx.argmax(-1)).float().mean()))
    lg = fx.view(n, n, -1); amp = lg.max(-1).values                                   # max-logit surface over (a, b)
    ab = torch.as_tensor((a + b) % n, device=DEV).view(n, n); amp_cv = 0.0           # CV of max-logit across a-b at fixed a+b, averaged over a+b
    for sm in range(n):
        v = amp[ab == sm]; amp_cv += float(v.std() / v.mean().abs().clamp_min(1e-12))
    swap = float((fx.view(n, n, -1) - fx.view(n, n, -1).transpose(0, 1)).norm() / fx.norm())
    return dict(d1_eps=float(np.mean(eps)), d1_kappa=float(np.mean(kap)), d1_ampcv=amp_cv / n, swap_asym=swap)


@torch.no_grad()
def cover_numbers(model, X, n=1500, radii=(0.25, 0.5, 1.0)):
    """H2.7 proxy for the covering number of the learned quotient: greedy r-net size of the probe set under the chord distance of s = 2 sqrt(p)
    (the Fisher-Rao embedding of the output at the first position).  Points the model treats alike collapse into one ball."""
    X = X[:n]; p = predict(model, X)[:, 0].softmax(-1); S = 2 * p.clamp_min(1e-12).sqrt(); out = {}
    for r in radii:
        alive = torch.ones(len(S), dtype=torch.bool, device=S.device); cnt = 0
        while alive.any():
            i = int(alive.nonzero()[0]); cnt += 1; alive &= (S - S[i]).norm(dim=1) > r
        out[f'cover_r{r}'] = cnt / len(S)
    return out


# ---- non-abelian blind recovery: principal isotope (Albert) of the model's prediction table, then invariants -----------------
def table_invariants(K, rng, n_assoc=4000):
    N = len(K); I = np.arange(N)
    if (K < 0).any(): return None
    ij = rng.integers(0, N, (n_assoc, 3)); u, v, w = ij.T; assoc = float((K[K[u, v], w] == K[u, K[v, w]]).mean())
    comm = float((K == K.T).mean())
    ident = [i for i in range(N) if (K[i] == I).mean() > .9 and (K[:, i] == I).mean() > .9]
    e = ident[0] if ident else None
    hist = np.zeros(N + 1)
    if e is not None:
        for x in range(N):
            y, k = x, 1
            while y != e and k <= N: y = K[y, x]; k += 1
            hist[min(k, N)] += 1
    centre = float(np.mean([(K[z] == K[:, z]).mean() > 0.97 for z in range(N)]))
    return dict(assoc=assoc, comm=comm, centre=centre, hist=hist, identity_found=e is not None)


def recover_group(table, seed=0, tries=6):
    """table: (N,N) predicted products of labelled elements.  Principal isotopes at several base points; returns invariants of the best one (by associativity)."""
    N = len(table); rng = np.random.default_rng(seed); best = None
    for _ in range(tries):
        a0, b0 = rng.integers(0, N, 2)
        alpha = table[:, b0]; beta = table[a0, :]              # row x -> value, col y -> value
        ia = -np.ones(N, int); ib = -np.ones(N, int)
        for x in range(N):
            if ia[alpha[x]] < 0: ia[alpha[x]] = x
            if ib[beta[x]] < 0: ib[beta[x]] = x
        K = -np.ones((N, N), int); okr = ia >= 0; okc = ib >= 0
        for u in range(N):
            if not okr[u]: continue
            for v in range(N):
                if okc[v]: K[u, v] = table[ia[u], ib[v]]
        # missing rows/cols (non-Latin) fill with 0 so invariants stay computable but poor
        bad = (K < 0).mean(); K = np.where(K < 0, 0, K)
        inv = table_invariants(K, rng)
        if inv is None: continue
        inv['missing'] = float(bad)
        if best is None or inv['assoc'] > best['assoc']: best = inv
    return best


def candidate_signatures(order):
    from sympy.combinatorics import named_groups as ng
    from sympy.combinatorics.group_constructs import DirectProduct as DP
    C = lambda n: ng.CyclicGroup(n); D = lambda n: ng.DihedralGroup(n); S = lambda n: ng.SymmetricGroup(n); A = lambda n: ng.AlternatingGroup(n)
    lists = {24: dict(D12=D(12), S4=S(4), Z24=C(24), Z2xA4=DP(C(2), A(4)), Z3xD4=DP(C(3), D(4)), Z4xS3=DP(C(4), S(3)), Z2xD6=DP(C(2), D(6)), Z2xZ12=DP(C(2), C(12)), Z2Z2xS3=DP(C(2), C(2), S(3))),
             60: dict(A5=A(5), D30=D(30), Z60=C(60), Z2xZ30=DP(C(2), C(30)), S3xZ10=DP(S(3), C(10)), A4xZ5=DP(A(4), C(5)), D6xZ5=DP(D(6), C(5)), Z3xD10=DP(C(3), D(10)), Z6xD5=DP(C(6), D(5))),
             120: dict(S5=S(5), A5xZ2=DP(A(5), C(2)), Z120=C(120), D60=D(60), A4xZ10=DP(A(4), C(10)), S4xZ5=DP(S(4), C(5)), S3xD10=DP(S(3), D(10)), S3xZ20=DP(S(3), C(20)), Z2xZ60=DP(C(2), C(60)))}[order]
    out = {}
    for k, G in lists.items():
        els = list(G.generate()); idx = {tuple(e.array_form): i for i, e in enumerate(els)}; N = len(els); assert N == order, (k, N)
        M = np.array([[idx[tuple((p * q).array_form)] for q in els] for p in els]); out[k] = M
    return out


_SIGS = {}


def signature_of_table(M, rng=None):
    rng = rng or np.random.default_rng(0); N = len(M); inv = table_invariants(M, rng); return inv


def classify_recovered(inv, order, truth_name):
    if inv is None or not inv['identity_found'] or inv['assoc'] < 0.9: return dict(group_recovered=False, group_pred=None, assoc=None if inv is None else inv['assoc'])
    if order not in _SIGS:
        _SIGS[order] = {k: signature_of_table(M) for k, M in candidate_signatures(order).items()}
    best, bd = None, 1e9
    for k, s in _SIGS[order].items():
        d = np.abs(inv['hist'][:order] - s['hist'][:order]).sum() / order + abs(inv['comm'] - s['comm']) + abs(inv['centre'] - s['centre'])
        if d < bd: best, bd = k, d
    return dict(group_recovered=(best == truth_name), group_pred=best, assoc=inv['assoc'], sig_dist=float(bd))


# ============================================================ one row ========================================================
def git_commit():
    try: return subprocess.check_output(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return 'nogit'


def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def instruments(task, arch, model, model0, data, regime, seed, hp, tag):
    """All instruments on one model state, evaluated on the probe set Xp (+ test/train subsets).  Returns flat dict + per-input arrays."""
    T = lambda a: torch.as_tensor(a, device=DEV)
    Xp, Yp = T(data['Xp']), T(data['Yp']); model.eval(); row = {}
    grp = task.group(); G = grp if grp else []
    lg = predict(model, Xp); row['probe_acc'] = acc_of(lg, Yp)
    if G:
        st = orbit_stats(model, Xp, G); row['kappa'] = float(st['kappa'].mean()); row['kappa_s'] = float(st['kappa_s'].mean()); row['eps_bar'] = float(st['eps_bar'].mean())
        row.update(q_stats(model, Xp, G))
        # non-group null: random token substitutions with the same size as the group
        rng = np.random.default_rng(seed + 77); Gn = []
        for _ in range(len(G)):
            sub = torch.as_tensor(rng.permutation(task.V_in), device=DEV)
            Gn.append(((lambda x, sub=sub: sub[x]), (lambda l: l)))
        sn = orbit_stats(model, Xp, Gn); row['kappa_null'] = float(sn['kappa'].mean()); row['eps_null'] = float(sn['eps_bar'].mean())
    lgrp = task.local_group()
    if lgrp:
        sl = orbit_stats(model, Xp, lgrp); row['kappa_local'] = float(sl['kappa'].mean())
    row.update(slot_shares(model, Xp)); row.update(fiber_tightening(model, model0, Xp, data['Yp'])); row.update(cover_numbers(model, Xp))
    bs = baselines(model, data); per_input = bs.pop('per_input'); row.update(bs)
    if isinstance(task, ModAdd):
        row['fourier_top5'] = fourier_concentration(model, task.n); row.update(diff_modulation(model, task.n))
        if task.n <= 113: row.update(blind_automorphism(model, task.n))
    if isinstance(task, SymFree): row.update(blind_automorphism(model, 24))
    if isinstance(task, FiniteGroupTask):
        L = all_pair_logits(model, task.N); tab = L.argmax(-1); row['table_acc'] = float((tab == task.M).mean())
        row.update(classify_recovered(recover_group(tab, seed), task.N, task.name))
    return row, per_input, (st if G else None)


def exception_scores(task, model, data):
    """H2.2: per-training-input scores for exception localisation (higher = more exception-like): 1-kappa, eps_bar, 1-msp, entropy, -margin, loss."""
    T = lambda a: torch.as_tensor(a, device=DEV); G = task.group(); X = T(data['Xtr']); Y = T(data['Ytr'])
    st = orbit_stats(model, X, G); lg = predict(model, X); pr = lg.softmax(-1); msp = pr.max(-1).values.mean(-1).cpu().numpy(); ent = (-(pr * (pr + 1e-30).log()).sum(-1)).mean(-1).cpu().numpy()
    t2 = lg.topk(2, -1).values; mar = (t2[..., 0] - t2[..., 1]).mean(-1).cpu().numpy()
    loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), Y.reshape(-1), reduction='none').view(len(X), -1).mean(1).cpu().numpy()
    return {'inc_kappa': 1 - st['kappa'], 'eps_bar': st['eps_bar'], 'inc_kappa_s': 1 - st['kappa_s'], 'one_minus_msp': 1 - msp, 'entropy': ent, 'neg_margin': -mar, 'loss': loss}


def auroc(score, label):
    from scipy.stats import rankdata
    label = np.asarray(label, bool); n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0: return float('nan')
    r = rankdata(score); return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_config(cfg, progress=print):
    """cfg: dict(task, arch, regime, seed, hp).  Returns list of result rows (final row, plus midgrok row if available)."""
    t0 = time.time(); task = get_task(cfg['task']); hp = cfg['hp']; seed = cfg['seed']; regime = cfg['regime']
    data = build_data(task, seed, regime)
    graft = None
    if regime == 'transplant':      # donor = a CLEAN model of the same task/arch on another seed (donor hp: hp['donor'] or hp with wd=1)
        dhp = dict(hp.get('donor', hp)); dhp.setdefault('wd', 1.0); dhp.pop('graft_keys', None)
        dm, _, _, dinfo = train(task, cfg['arch'], 'clean', seed + 500, dhp, build_data(task, seed + 500, 'clean'), progress)
        graft = {k: v.detach().clone() for k, v in dm.state_dict().items()}
    model, mid, traj, info = train(task, cfg['arch'], regime, seed, hp, data, progress, graft=graft)
    model0 = make_model(task, cfg['arch'], seed, **hp.get('model', {}))
    rows = []
    def finish(m, step, tag, test_acc_override=None):
        row, per_input, st = instruments(task, cfg['arch'], m, model0, data, regime, seed, hp, tag)
        base = dict(config_hash=config_hash(cfg), git_commit=git_commit(), task=cfg['task'], arch=cfg['arch'], regime=regime, seed=seed, snapshot=tag, step=step,
                    hp=json.dumps(hp, sort_keys=True), n_params=sum(p.numel() for p in m.parameters()))
        T = lambda a: torch.as_tensor(a, device=DEV)
        base['test_acc'] = acc_of(predict(m, T(data['Xte'])), T(data['Yte'])); base['train_acc'] = acc_of(predict(m, T(data['Xtr'])), T(data['Ytr']))
        base['train_acc_true'] = acc_of(predict(m, T(data['Xtr'])), T(data['Ytr_true']))
        if regime == 'hybrid':
            ex = data['exc']; lg = predict(m, T(data['Xtr'])); base['rule_acc_nonexc'] = acc_of(lg[T(~ex)], T(data['Ytr_true'])[T(~ex)]); base['exc_memorised'] = acc_of(lg[T(ex)], T(data['Ytr'])[T(ex)])
            base['exc_rate'] = float(ex.mean()); sc = exception_scores(task, m, data)
            for k, v in sc.items(): base[f'auroc_exc_{k}'] = auroc(v, ex)
        row = {**base, **row}; return row
    save_dir = os.environ.get('WB_SAVE_DIR'); cid = os.environ.get('WB_CFG_ID', config_hash(cfg))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True); torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, f'{save_dir}/{cid}_final.pt')
    rows.append(finish(model, info['steps_run'], 'final'))
    rows[0].update(info); rows[0]['traj'] = traj; rows[0]['wall_seconds'] = None
    if mid is not None:
        m2 = make_model(task, cfg['arch'], seed, **hp.get('model', {})); m2.load_state_dict(mid['state']); m2.eval()
        if save_dir: torch.save({k: v.detach().cpu() for k, v in mid['state'].items()}, f'{save_dir}/{cid}_mid.pt')
        r2 = finish(m2, mid['step'], 'midgrok'); r2['regime'] = 'midgrok'; r2['mid_test_acc'] = mid['test_acc']; rows.append(r2)
    w = time.time() - t0
    for r in rows: r['wall_seconds'] = w
    return rows
