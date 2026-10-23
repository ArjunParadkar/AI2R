"""Planted-answer systems (master prompt §5). Estimator tests only; never evidence.

Every planted system is an exact ReLU/clamp residual network built from a
*spec*: a list of variables, each written by one block into its own block of
the residual stream (then rotated by a random orthogonal Q so that realisers
are not axis aligned). The spec is the ground truth: nodes, parent sets,
derivation tables, and (by computing activity with the true realisers) the
ground-truth generality of every node.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from whitebox_ee.systems import Block, ResidualMLPSystem

DT = torch.float64


@dataclass
class PVar:
    name: str
    card: int
    layer: int                      # 0 = input variable
    parents: Tuple[str, ...] = ()
    fn: Optional[Callable] = None   # fn(*parent_values) -> value
    enc: str = "onehot"             # onehot | fourier | random
    enc_dim: Optional[int] = None   # for 'random'
    concept: bool = True            # False for planted decoys / sub-threshold detectors


def templates(card, enc, rng, enc_dim=None):
    if enc == "onehot":
        E = np.eye(card)
    elif enc == "fourier":
        K = max(1, min(3, card // 2))
        v = np.arange(card)[:, None]
        k = np.arange(1, K + 1)[None, :]
        E = np.concatenate([np.cos(2 * np.pi * k * v / card), np.sin(2 * np.pi * k * v / card)], 1)
    elif enc == "random":
        d = enc_dim or max(2, card // 2)
        E = rng.standard_normal((card, d))
    else:
        raise ValueError(enc)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return E


@dataclass
class Planted:
    system: ResidualMLPSystem
    vars: Dict[str, PVar]
    U: Dict[str, np.ndarray]                  # true realiser (D x k) of every variable
    E: Dict[str, np.ndarray]
    site_of: Dict[str, str]
    behaviours: Dict[str, str]                # behaviour -> variable it reads
    sample: Callable                          # sample(n, rng) -> X
    truth_parents: Dict[str, Tuple[str, ...]]
    meta: dict = field(default_factory=dict)

    def values(self, X) -> Dict[str, np.ndarray]:
        """Ground-truth values of every variable on inputs X."""
        X = np.asarray(X)
        vals = {}
        inputs = [v for v in self.vars.values() if v.layer == 0]
        for j, v in enumerate(inputs):
            vals[v.name] = X[:, j].astype(np.int64)
        for v in sorted([v for v in self.vars.values() if v.layer > 0], key=lambda v: v.layer):
            ps = [vals[p] for p in v.parents]
            vals[v.name] = np.array([v.fn(*t) for t in zip(*ps)], dtype=np.int64)
        return vals

    def decode(self, name, states):
        """Read variable `name` from states at its site via its templates."""
        z = states @ self.U[name]
        return (z @ self.E[name].T).argmax(1)


def build(vars_: Sequence[PVar], behaviours: Dict[str, str], sample: Callable, seed: int = 0,
          n_distract: int = 16, distract_scale: float = 0.3, noise: float = 0.0,
          beta_out: float = 8.0, rotate: bool = True, name: str = "planted") -> Planted:
    rng = np.random.default_rng(seed)
    V = {v.name: v for v in vars_}
    E = {v.name: templates(v.card, v.enc, rng, v.enc_dim) for v in vars_}
    # stream layout
    offs, D = {}, 0
    for v in vars_:
        offs[v.name] = (D, D + E[v.name].shape[1])
        D += E[v.name].shape[1]
    d_off = D
    D += n_distract
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0] if rotate else np.eye(D)
    U = {}
    for v in vars_:
        a, b = offs[v.name]
        U[v.name] = Q[:, a:b]          # orthonormal (D x k)
    # value templates in stream coordinates: (card x D)
    T = {n: E[n] @ U[n].T for n in V}

    inputs = [v for v in vars_ if v.layer == 0]
    emb = [torch.tensor(T[v.name], dtype=DT) for v in inputs]

    # deterministic distractors + noise: sin features of the one-hot inputs
    A_d = rng.standard_normal((sum(v.card for v in inputs), n_distract)) * 1.5
    c_d = rng.uniform(0, 2 * np.pi, n_distract)
    A_n = rng.standard_normal((sum(v.card for v in inputs), D)) * 1.7
    c_n = rng.uniform(0, 2 * np.pi, D)
    Qd = Q[:, d_off:]
    cards = [v.card for v in inputs]

    def onehots(X):
        X = np.asarray(X)
        return np.concatenate([np.eye(c)[X[:, j]] for j, c in enumerate(cards)], 1)

    def extra(X):
        oh = onehots(X)
        out = distract_scale * np.sin(oh @ A_d + c_d) @ Qd.T
        if noise > 0:
            out = out + noise * np.sin(oh @ A_n + c_n) / np.sqrt(D)
        return out

    blocks = []
    L = max(v.layer for v in vars_)
    for layer in range(1, L + 1):
        lv = [v for v in vars_ if v.layer == layer]
        W1, b1, W2, b2, W3 = [], [], [], [], []
        n_ind = 0
        rows_and = []
        for v in lv:
            ind_index = {}
            for p in v.parents:
                Ep = E[p]
                sims = Ep @ Ep.T
                m = np.max(sims - 2 * np.eye(len(sims))) if len(sims) > 1 else 0.0
                theta = (1 + m) / 2
                beta = 1 / (1 - theta)
                for t in range(V[p].card):
                    W1.append(beta * T[p][t])
                    b1.append(-beta * theta)
                    ind_index[(p, t)] = n_ind
                    n_ind += 1
            k = len(v.parents)
            for combo in itertools.product(*[range(V[p].card) for p in v.parents]):
                rows_and.append(([ind_index[(p, t)] for p, t in zip(v.parents, combo)], k, T[v.name][v.fn(*combo)]))
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.zeros((len(rows_and), n_ind))
        b2 = np.zeros(len(rows_and))
        W3 = np.zeros((D, len(rows_and)))
        for r, (idx, k, write) in enumerate(rows_and):
            W2[r, idx] = 1.0
            b2[r] = -(k - 1)
            W3[:, r] = write
        blocks.append(Block("and", torch.tensor(W1, dtype=DT), torch.tensor(b1, dtype=DT),
                            torch.tensor(W3, dtype=DT), torch.tensor(W2, dtype=DT), torch.tensor(b2, dtype=DT)))
    readouts = {}
    for b, vname in behaviours.items():
        readouts[b] = (torch.tensor(beta_out * T[vname], dtype=DT), torch.zeros(V[vname].card, dtype=DT))
    sys_ = ResidualMLPSystem(emb, blocks, readouts, [v.name for v in inputs], extra, name)
    site_of = {v.name: f"h{v.layer}" for v in vars_}
    truth = {v.name: tuple(v.parents) for v in vars_ if v.layer > 0}
    return Planted(sys_, V, U, E, site_of, behaviours, sample, truth, dict(seed=seed, D=D, noise=noise))


# ---------------------------------------------------------------------------
# V1: multi-digit addition (digits, column sums, carries, result digits)
# ---------------------------------------------------------------------------

def v1_addition(n_digits: int = 3, seed: int = 0, enc_sum: str = "onehot", noise: float = 0.0,
                with_decoy: bool = True) -> Planted:
    vs = []
    for i in range(n_digits):
        vs.append(PVar(f"a{i}", 10, 0))
    for i in range(n_digits):
        vs.append(PVar(f"b{i}", 10, 0))
    for i in range(n_digits):
        vs.append(PVar(f"s{i}", 19, 1, (f"a{i}", f"b{i}"), lambda a, b: a + b, enc=enc_sum))
    if with_decoy:  # represented, never read
        vs.append(PVar("decoy", 7, 1, ("a0", "b1"), lambda a, b: (a * b + a) % 7, concept=False))
    vs.append(PVar("c0", 2, 2, ("s0",), lambda s: int(s >= 10)))
    vs.append(PVar("r0", 10, 2, ("s0",), lambda s: s % 10))
    for i in range(1, n_digits):
        vs.append(PVar(f"c{i}", 2, 2 + i, (f"s{i}", f"c{i-1}"), lambda s, c: int(s + c >= 10)))
        vs.append(PVar(f"r{i}", 10, 2 + i, (f"s{i}", f"c{i-1}"), lambda s, c: (s + c) % 10))
    beh = {f"out{i}": f"r{i}" for i in range(n_digits)}
    beh[f"out{n_digits}"] = f"c{n_digits-1}"

    def sample(n, rng):
        return rng.integers(0, 10, size=(n, 2 * n_digits))

    return build(vs, beh, sample, seed=seed, noise=noise, name=f"V1_add{n_digits}_{enc_sum}")


# ---------------------------------------------------------------------------
# V2: planted memorized subset + planted rule subset
#   y = (a + b) mod p through a sum node, except on a planted subset M where
#   y is an arbitrary label. M is implemented by one detector per memorized
#   pair (each its own leaf, entropy far below h_min) feeding an override
#   node chain; detectors are planted non-concepts.
# ---------------------------------------------------------------------------

def v2_memo(p: int = 11, frac_memo: float = 0.2, seed: int = 0, n_detect_groups: int = 0) -> Planted:
    rng = np.random.default_rng(1000 + seed)
    pairs = [(a, b) for a in range(p) for b in range(p)]
    nm = int(round(frac_memo * len(pairs)))
    idx = rng.choice(len(pairs), nm, replace=False)
    memo = {pairs[i]: int(rng.integers(0, p)) for i in idx}
    # make sure memorized labels differ from the rule
    for k in memo:
        if memo[k] == (k[0] + k[1]) % p:
            memo[k] = (memo[k] + 1 + int(rng.integers(0, p - 1))) % p
    memo_list = sorted(memo)
    vs = [PVar("a", p, 0), PVar("b", p, 0),
          PVar("sum", p, 1, ("a", "b"), lambda a, b: (a + b) % p)]
    # one binary detector per memorized pair: m_j = [ (a,b) == pair_j ]  (leaf, H << h_min)
    for j, pr in enumerate(memo_list):
        vs.append(PVar(f"m{j}", 2, 1, ("a", "b"), (lambda pr: (lambda a, b: int((a, b) == pr)))(pr), concept=False))
    # override value: label of the memorized pair, or p (= none). Built as a
    # chain so that each block has a small fan-in.
    def ov_fn(*ms, labels=[memo[k] for k in memo_list]):
        for m, l in zip(ms, labels):
            if m:
                return l
        return p
    # chain in groups of 4 detectors to keep tables small
    groups = [list(range(i, min(i + 4, nm))) for i in range(0, nm, 4)]
    prev = None
    for gi, g in enumerate(groups):
        names = [f"m{j}" for j in g]
        labels = [memo[memo_list[j]] for j in g]
        if prev is None:
            fn = (lambda labels: (lambda *ms: next((l for m, l in zip(ms, labels) if m), p)))(labels)
            vs.append(PVar(f"ov{gi}", p + 1, 2 + gi, tuple(names), fn, concept=False))
        else:
            fn = (lambda labels: (lambda o, *ms: o if o != p else next((l for m, l in zip(ms, labels) if m), p)))(labels)
            # carry detectors forward: they live in the stream already (residual)
            vs.append(PVar(f"ov{gi}", p + 1, 2 + gi, (prev,) + tuple(names), fn, concept=False))
        prev = f"ov{gi}"
    Lf = 2 + len(groups)
    vs.append(PVar("y", p, Lf, ("sum", prev), lambda s, o: s if o == p else o))
    beh = {"out": "y"}

    def sample(n, rng):
        return rng.integers(0, p, size=(n, 2))

    P = build(vs, beh, sample, seed=seed, name=f"V2_memo_p{p}")
    P.meta["memo"] = memo
    P.meta["p"] = p

    def is_memo(X):
        X = np.asarray(X)
        return np.array([(int(a), int(b)) in memo for a, b in X])
    P.meta["is_memo"] = is_memo
    return P


# ---------------------------------------------------------------------------
# V3: planted sideways. C = (A + B) mod q with A from (a1,a2), B from (b1,b2);
# training covers every A value and every B value but withholds a set W of
# (A,B) combinations. Two variants:
#   compositional: the C-block is the full table (correct on W)
#   lookup:        the C-block is correct only on seen combinations; on W it
#                  writes a wrong value (a network that never needed W).
# ---------------------------------------------------------------------------

def v3_sideways(q: int = 5, seed: int = 0, compositional: bool = True, n_withheld: int = 6) -> Planted:
    rng = np.random.default_rng(2000 + seed)
    combos = [(a, b) for a in range(q) for b in range(q)]
    # withheld set: one combination per row/column-ish, every value still seen
    W = set()
    perm = rng.permutation(len(combos))
    for i in perm:
        a, b = combos[i]
        if len(W) >= n_withheld:
            break
        if sum(1 for (x, y) in W if x == a) < 1 and sum(1 for (x, y) in W if y == b) < 1:
            W.add((a, b))
    wrong = {(a, b): ((a + b) % q + 1 + int(rng.integers(0, q - 1))) % q for (a, b) in W}

    def cfn(a, b):
        if (not compositional) and (a, b) in W:
            return wrong[(a, b)]
        return (a + b) % q
    vs = [PVar("a1", q, 0), PVar("a2", 2, 0), PVar("b1", q, 0), PVar("b2", 2, 0),
          PVar("A", q, 1, ("a1", "a2"), lambda x, s: (x + s) % q),
          PVar("B", q, 1, ("b1", "b2"), lambda x, s: (x + 2 * s) % q),
          PVar("C", q, 2, ("A", "B"), cfn),
          PVar("y", q, 3, ("C",), lambda c: c)]
    beh = {"out": "y"}

    def a_of(x):
        return (x[:, 0] + x[:, 1]) % q

    def b_of(x):
        return (x[:, 2] + 2 * x[:, 3]) % q

    def sample(n, rng, withheld_only=False, train=True):
        out = []
        while sum(len(o) for o in out) < n:
            X = np.stack([rng.integers(0, q, 4 * n), rng.integers(0, 2, 4 * n),
                          rng.integers(0, q, 4 * n), rng.integers(0, 2, 4 * n)], 1)
            ab = list(zip(a_of(X), b_of(X)))
            inW = np.array([t in W for t in ab])
            keep = inW if withheld_only else (~inW if train else np.ones(len(X), bool))
            out.append(X[keep])
        return np.concatenate(out)[:n]

    P = build(vs, beh, sample, seed=seed, name=f"V3_sideways_{'comp' if compositional else 'lookup'}")
    P.meta.update(W=W, q=q, compositional=compositional, a_of=a_of, b_of=b_of)
    return P


# ---------------------------------------------------------------------------
# V5: planted creation. Before: a network with nodes A, B (from inputs) and
# behaviours reading A and B. After exposure: the same network plus a new
# composite node N = (A + B) mod q feeding a new behaviour.
# Controls (false-creation types, theory Def. 4.6):
#   'decoy'   : after-network adds a node computed from A,B but never read (a)
#   'leaf'    : after-network's new behaviour is a per-input lookup through a
#               new node derived from the raw inputs only, not from A,B (b)
#   'relabel' : after-network adds a node that is an invertible relabelling of A (c)
#   'generic' : control exposure — same addition happens under the control
#               problem too (d)  (simulated by returning the 'valid' network for both)
# ---------------------------------------------------------------------------

def v5_creation(q: int = 5, seed: int = 0, variant: str = "valid") -> Tuple[Planted, Planted]:
    base = [PVar("x1", q, 0), PVar("x2", 2, 0), PVar("x3", q, 0), PVar("x4", 2, 0),
            PVar("A", q, 1, ("x1", "x2"), lambda u, s: (u + s) % q),
            PVar("B", q, 1, ("x3", "x4"), lambda u, s: (u + 2 * s) % q),
            PVar("yA", q, 2, ("A",), lambda a: a), PVar("yB", q, 2, ("B",), lambda b: b)]
    beh0 = {"outA": "yA", "outB": "yB"}

    def sample(n, rng):
        return np.stack([rng.integers(0, q, n), rng.integers(0, 2, n), rng.integers(0, q, n), rng.integers(0, 2, n)], 1)

    before = build(base + [PVar("yN", q, 3, ("A",), lambda a: a, concept=False)],
                   {**beh0, "outN": "yN"}, sample, seed=seed, name="V5_before")
    if variant == "valid":
        extra = [PVar("N", q, 2, ("A", "B"), lambda a, b: (a + b) % q),
                 PVar("yN", q, 3, ("N",), lambda n: n)]
    elif variant == "decoy":
        extra = [PVar("N", q, 2, ("A", "B"), lambda a, b: (a + b) % q, concept=False),
                 PVar("yN", q, 3, ("A",), lambda a: a)]
    elif variant == "leaf":
        rng = np.random.default_rng(seed + 77)
        tab = rng.integers(0, q, size=(q, 2, q, 2))
        extra = [PVar("N", q, 2, ("x1", "x2", "x3", "x4"), lambda a, b, c, d: int(tab[a, b, c, d])),
                 PVar("yN", q, 3, ("N",), lambda n: n)]
    elif variant == "relabel":
        extra = [PVar("N", q, 2, ("A",), lambda a: (3 * a + 1) % q),
                 PVar("yN", q, 3, ("N",), lambda n: n)]
    else:
        raise ValueError(variant)
    after = build(base + extra, {**beh0, "outN": "yN"}, sample, seed=seed, name=f"V5_after_{variant}")
    return before, after
