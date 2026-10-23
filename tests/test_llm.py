"""Tests for whitebox_ee.llm.HFSystem, bench.tasks and whitebox_ee.baselines.

Run:  .venv/bin/python -m pytest tests/test_llm.py -q
Downloads: pythia-70m (~160 MB), Qwen2.5-0.5B (~1 GB, skipped with WB_SKIP_QWEN=1),
OLMo-2 tokenizer only.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.tasks import AdditionTask, AgreementTask, MultiplicationTask, digit_format  # noqa: E402
from whitebox_ee.baselines import (ccgp, ccgp_per_input, margin, max_softmax, neg_entropy,  # noqa: E402
                                   per_input_loss, sharpness_per_input, weight_norm, _loss_inputs)
from whitebox_ee.llm import HFSystem  # noqa: E402
from whitebox_ee.systems import Patch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PYTHIA = "EleutherAI/pythia-70m"
QWEN = "Qwen/Qwen2.5-0.5B"
OLMO_TOK = "allenai/OLMo-2-1124-7B"
SKIP_QWEN = os.environ.get("WB_SKIP_QWEN") == "1"


@pytest.fixture(scope="module")
def tokenizers():
    from transformers import AutoTokenizer
    return {n: AutoTokenizer.from_pretrained(n) for n in (PYTHIA, QWEN, OLMO_TOK)}


@pytest.fixture(scope="module")
def pythia_add():
    return HFSystem(PYTHIA, task=AdditionTask(3), dtype="float32", device=DEV)


@pytest.fixture(scope="module")
def qwen_add():
    if SKIP_QWEN:
        pytest.skip("WB_SKIP_QWEN=1")
    return HFSystem(QWEN, task=AdditionTask(3), dtype="bfloat16", device=DEV)


# ---------------------------------------------------------------- (c) + (e) tokenization / roles

@pytest.mark.parametrize("name,fmt", [(PYTHIA, "spaced"), (QWEN, "native"), (OLMO_TOK, "spaced")])
def test_digit_format(tokenizers, name, fmt):
    assert digit_format(tokenizers[name]) == fmt


@pytest.mark.parametrize("name", [PYTHIA, QWEN, OLMO_TOK])
@pytest.mark.parametrize("make", [lambda: AdditionTask(3), lambda: AdditionTask(2, n_shots=0),
                                  lambda: MultiplicationTask(2)])
def test_arith_roles_single_digit(tokenizers, name, make):
    tok = tokenizers[name]
    task = make()
    task.bind(tok)
    X = task.sample(50, np.random.default_rng(1))
    iv, tg = task.input_values(X), task.targets(X)
    cls = task.class_token_ids(tok)
    for n, ex in enumerate(X):
        ids, roles, readouts = task.render(ex, tok)
        # one token per digit: the natural tokenisation of the text equals the rendered ids
        assert ids == tok(ex["text"])["input_ids"]
        for i in range(task.n1):
            assert tok.decode([ids[roles[f"op1_d{i}"]]]).strip() == str(iv[f"a{i}"][n])
        for i in range(task.n2):
            assert tok.decode([ids[roles[f"op2_d{i}"]]]).strip() == str(iv[f"b{i}"][n])
        assert "=" in tok.decode([ids[roles["eq"]]])
        for j in range(task.n_out):
            p = roles[f"pre_ans_d{j}"]
            assert readouts[f"out{j}"] == p
            assert ids[p + 1] == cls[tg[f"out{j}"][n]]          # next token = answer digit j's class token
        assert len(set(roles[f"pre_ans_d{j}"] for j in range(task.n_out))) == task.n_out


@pytest.mark.parametrize("name", [PYTHIA, QWEN, OLMO_TOK])
def test_agreement_tokens_and_roles(tokenizers, name):
    tok = tokenizers[name]
    task = AgreementTask(n_shots=2)
    X = task.sample(80, np.random.default_rng(2))
    for ex in X:
        ids, roles, ro = task.render(ex, tok)
        assert tok.decode([ids[roles["subj"]]]).strip() == task._noun(ex["subj_id"], ex["subj_num"])
        assert tok.decode([ids[roles["attr"]]]).strip() == task._noun(ex["attr_id"], ex["attr_num"])
        assert len(set(roles.values())) == 4
        ci = task.example_class_ids(ex, tok)
        assert ids[ro["verb"] + 1] == ci[ex["subj_num"]]
    for w in task.nouns:
        for f in (w, w + "s"):
            assert len(tok.encode(" " + f, add_special_tokens=False)) == 1, f


def test_edit_rerenders(pythia_add):
    t = pythia_add.task
    X = t.sample(4, np.random.default_rng(3))
    Y = pythia_add.edit_inputs(X, {"a0": np.array([9, 9, 9, 9]), "b2": np.array([0, 1, 2, 3])})
    iv = t.input_values(Y)
    assert (iv["a0"] == 9).all() and (iv["b2"] == [0, 1, 2, 3]).all()
    assert (t.input_values(X)["a1"] == iv["a1"]).all()
    assert all(t.targets(Y)["out0"] == (iv["a0"] + iv["b0"]) % 10)


def test_labels_addition():
    t = AdditionTask(3)
    X = t.sample(200, np.random.default_rng(4))
    L, tg = t.labels(X), t.targets(X)
    for i in range(3):
        assert (L[f"r{i}"] == tg[f"out{i}"]).all()
    assert (L["c2"] == tg["out3"]).all()
    Xn = t.sample(50, np.random.default_rng(5), carries=False)
    assert all((t.labels(Xn)[f"c{i}"] == 0).all() for i in range(3))
    Xc = t.sample(50, np.random.default_rng(5), carries=True)
    Lc = t.labels(Xc)
    assert (np.stack([Lc[f"c{i}"] for i in range(3)]).sum(0) > 0).all()


def test_labels_multiplication():
    t = MultiplicationTask(2)
    X = t.sample(200, np.random.default_rng(6))
    L, tg = t.labels(X), t.targets(X)
    assert (L["r0"] == tg["out0"]).all() and (L["r1"] == tg["out1"]).all() and (L["k1"] == tg["out2"]).all()


# ---------------------------------------------------------------- (a) + (b) patching

def _check_patching(sys_, X, atol):
    lg, st = sys_.run(X)
    site = [s for s in sys_.sites if s.startswith("L2@")][0]
    later = [s for s in sys_.sites if s.startswith(f"L{sys_.n_layers}@")][-1]
    d = sys_.d_model
    # (a) self-patch is a no-op
    lg2, st2 = sys_.run(X, [Patch(site, np.eye(d, dtype=np.float32), st[site])])
    for b in lg:
        np.testing.assert_allclose(lg2[b], lg[b], atol=atol)
    np.testing.assert_allclose(st2[later], st[later], atol=atol)
    # (b) full identity basis from another example reproduces that example's state
    perm = np.roll(np.arange(len(X)), 1)
    src = st[site][perm]
    lg3, st3 = sys_.run(X, [Patch(site, np.eye(d, dtype=np.float32), src)])
    scale = np.abs(src).max()
    np.testing.assert_allclose(st3[site], src, atol=1e-2 * scale if sys_.dtype_name != "float32" else 1e-5)
    change = max(np.abs(st3[later] - st[later]).max(), max(np.abs(lg3[b] - lg[b]).max() for b in lg))
    assert change > 1e-2
    # a random 1-d basis: only that coordinate is set
    rng = np.random.default_rng(0)
    u = rng.standard_normal((d, 1)).astype(np.float32)
    u /= np.linalg.norm(u)
    _, st4 = sys_.run(X, [Patch(site, u, src)])
    np.testing.assert_allclose(st4[site] @ u, src @ u, atol=2e-2 * scale if sys_.dtype_name != "float32" else 1e-4)
    # patches compose: patching twice at the same site with full basis = last source wins
    _, st5 = sys_.run(X, [Patch(site, np.eye(d, dtype=np.float32), src),
                          Patch(site, np.eye(d, dtype=np.float32), st[site])])
    # (x + (src - x)) + (x - src') == x only up to rounding, which later blocks amplify
    tol = 1e-3 if sys_.dtype_name == "float32" else 5e-2
    np.testing.assert_allclose(st5[later], st[later], atol=tol * np.abs(st[later]).max())


def test_patching_pythia(pythia_add):
    X = pythia_add.task.sample(24, np.random.default_rng(7))
    _check_patching(pythia_add, X, atol=1e-5)


def test_patching_qwen(qwen_add):
    X = qwen_add.task.sample(16, np.random.default_rng(8))
    _check_patching(qwen_add, X, atol=1e-3)


def test_padding_invariance(pythia_add):
    """Logits do not depend on batch composition / padding (float32)."""
    t = AgreementTask(n_shots=1)
    s = HFSystem(PYTHIA, task=t, dtype="float32", device=DEV, _model=pythia_add.model, _tokenizer=pythia_add.tok)
    X = t.sample(12, np.random.default_rng(9))
    lg, st = s.run(X)
    for i in range(3):
        lgi, sti = s.run([X[i]])
        np.testing.assert_allclose(lgi["verb"][0], lg["verb"][i], atol=2e-3)
    full = s.full_logits(X[:2])
    for i in range(2):
        ci = t.example_class_ids(X[i], s.tok)
        np.testing.assert_allclose(full[i, 0, ci], lg["verb"][i], atol=2e-3)


def test_layer_zero_is_embedding(pythia_add):
    X = pythia_add.task.sample(3, np.random.default_rng(10))
    _, st = pythia_add.run(X, sites=["L0@eq", "L1@eq"])
    ids, rp, _, _ = pythia_add.encode(X)
    k = pythia_add.task.roles.index("eq")
    E = pythia_add.model.get_input_embeddings().weight.detach().float().cpu().numpy()
    np.testing.assert_allclose(st["L0@eq"], E[[ids[i][rp[i, k]] for i in range(3)]], atol=1e-6)
    with torch.inference_mode():
        hs = pythia_add.model(input_ids=torch.as_tensor([ids[0]], device=DEV), output_hidden_states=True).hidden_states
    np.testing.assert_allclose(st["L1@eq"][0], hs[1][0, rp[0, k]].float().cpu().numpy(), atol=1e-4)


def test_untrained_copy(pythia_add):
    u = pythia_add.untrained_copy(seed=0)
    assert u.sites == pythia_add.sites
    w0 = pythia_add.model.get_input_embeddings().weight
    w1 = u.model.get_input_embeddings().weight
    assert not torch.allclose(w0, w1)
    X = pythia_add.task.sample(4, np.random.default_rng(11))
    lg, _ = u.run(X, sites=[])
    assert lg["out0"].shape == (4, 10)
    del u


# ---------------------------------------------------------------- integration with Probe

def test_probe_integration():
    from whitebox_ee.probe import Member, Node, Probe
    task = AdditionTask(2)
    s = HFSystem(PYTHIA, task=task, dtype="bfloat16", device=DEV)
    X = task.sample(64, np.random.default_rng(12))
    pr = Probe(s, X)
    assert pr.N == 64 and set(pr.states) == set(s.sites)
    site = "L3@op1_d0"
    U = np.linalg.qr(np.random.default_rng(0).standard_normal((s.d_model, 4)))[0]
    node = Node("c", "concept", [Member(site, U, np.zeros((2, 4)), np.array([0, 1]))], None, s.sites.index(site))
    rows = np.arange(16)
    lg, st = pr.run_patched(rows, srcs=[(node, rows[::-1])])
    assert lg["out0"].shape == (16, 10)
    np.testing.assert_allclose(st[site] @ U, pr.states[site][rows[::-1]] @ U, atol=0.1 * np.abs(pr.states[site]).max())
    lg0, _ = pr.run_patched(rows, freeze=[node])
    np.testing.assert_allclose(lg0["out0"], pr.logits["out0"][rows], atol=0.1)
    # input edits through the probe
    a0 = Node("a0", "input", [], pr.inputs["a0"], -1)
    lgi, _ = pr.run_patched(rows, srcs=[(a0, rows[::-1])])
    assert lgi["out0"].shape == (16, 10)


# ---------------------------------------------------------------- baselines

def test_softmax_baselines():
    z = np.array([[3.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    assert max_softmax(z)[0] > max_softmax(z)[1]
    assert neg_entropy(z)[0] > neg_entropy(z)[1]
    np.testing.assert_allclose(margin(z), [2.0, 0.0])
    d = max_softmax({"a": z, "b": z[::-1]}, reduce="min")
    assert d.shape == (2,)


def test_ccgp_planted():
    """Abstract (factorised) code -> CCGP ~ 1; random per-condition code -> ~ chance."""
    rng = np.random.default_rng(0)
    conds = np.array([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    cid = np.repeat(np.arange(8), 30)
    lab = {"x": conds[cid, 0], "y": conds[cid, 1]}
    D = 40
    axes = rng.standard_normal((3, D))
    H_abs = conds[cid] @ axes * 3 + rng.standard_normal((len(cid), D))
    H_rand = rng.standard_normal((8, D))[cid] * 3 + rng.standard_normal((len(cid), D))
    a = ccgp(H_abs, lab, cid)
    r = ccgp(H_rand, lab, cid)
    assert a["x"] > 0.9 and a["y"] > 0.9
    assert r["x"] < 0.75
    tr, te = np.arange(0, len(cid), 2), np.arange(1, len(cid), 2)
    sc = ccgp_per_input({"s": H_abs, "r": H_rand}, lab, cid, tr, te)
    assert sc.shape == (len(te),) and np.nanmean(sc) > 0
    sc_r = ccgp_per_input({"r": H_rand}, lab, cid, tr, te)
    assert np.nanmean(sc) > np.nanmean(sc_r)


# ---------------------------------------------------------------- (d) gradients and Hutchinson

@pytest.fixture(scope="module")
def tiny_neox():
    from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM
    tok = AutoTokenizer.from_pretrained(PYTHIA)
    cfg = GPTNeoXConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=64, max_position_embeddings=256)
    torch.manual_seed(0)
    m = GPTNeoXForCausalLM(cfg).to(DEV).double()
    m.eval()
    return m, tok


def test_gradient_vs_finite_difference():
    """Real pythia-70m (float64), SDPA MATH backend: directional derivative of the per-input
    loss matches central finite differences."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PYTHIA)
    m = AutoModelForCausalLM.from_pretrained(PYTHIA).to(DEV).double().eval()
    task = AdditionTask(2)
    ex = task.sample(1, np.random.default_rng(13))[0]
    ids, pos, tgt = _loss_inputs(task, tok, ex)
    p = m.gpt_neox.layers[-1].mlp.dense_4h_to_h.weight
    g = torch.Generator(device=DEV).manual_seed(0)
    v = torch.randn(p.shape, generator=g, device=DEV, dtype=p.dtype)
    with sdpa_kernel(SDPBackend.MATH):
        loss = per_input_loss(m, ids, pos, tgt)
        (gr,) = torch.autograd.grad(loss, [p])
        dd = float((gr * v).sum())
        eps = 1e-4
        with torch.no_grad():
            p.add_(eps * v)
            lp = float(per_input_loss(m, ids, pos, tgt))
            p.sub_(2 * eps * v)
            lm = float(per_input_loss(m, ids, pos, tgt))
            p.add_(eps * v)
    fd = (lp - lm) / (2 * eps)
    assert abs(fd - dd) <= 1e-5 * max(1.0, abs(dd)), (fd, dd)
    del m


def test_hutchinson_vs_exact(tiny_neox):
    m, tok = tiny_neox
    task = AdditionTask(2, n_shots=1)
    X = task.sample(2, np.random.default_rng(14))
    # parameters entering the residual stream additively (block output biases): their Hessian is
    # close to Gauss-Newton, so 1000 probes give a tight Hutchinson estimate on the tiny model
    P = [m.gpt_neox.layers[-1].mlp.dense_4h_to_h.bias, m.gpt_neox.layers[-1].attention.dense.bias]
    est, se = sharpness_per_input(m, tok, task, X, params=P, n_hutch=1000, return_se=True)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for n, ex in enumerate(X):
        ids, pos, tgt = _loss_inputs(task, tok, ex)
        with sdpa_kernel(SDPBackend.MATH):
            loss = per_input_loss(m, ids, pos, tgt)
            grads = torch.autograd.grad(loss, P, create_graph=True)
            flat = torch.cat([g.reshape(-1) for g in grads])
            tr = 0.0
            for i in range(flat.numel()):
                row = torch.autograd.grad(flat[i], P, retain_graph=True, allow_unused=True)
                rowf = torch.cat([(r if r is not None else torch.zeros_like(q)).reshape(-1) for r, q in zip(row, P)])
                tr += float(rowf[i])
            # HVP vs finite difference of gradients along a random direction
            v = [torch.randn_like(q) for q in P]
            gv = sum((g * vv).sum() for g, vv in zip(grads, v))
            Hv = torch.autograd.grad(gv, P)
            eps = 1e-5
            with torch.no_grad():
                for q, vv in zip(P, v):
                    q.add_(eps * vv)
            gp = torch.autograd.grad(per_input_loss(m, ids, pos, tgt), P)
            with torch.no_grad():
                for q, vv in zip(P, v):
                    q.sub_(2 * eps * vv)
            gm = torch.autograd.grad(per_input_loss(m, ids, pos, tgt), P)
            with torch.no_grad():
                for q, vv in zip(P, v):
                    q.add_(eps * vv)
            fd = torch.cat([((a - b) / (2 * eps)).reshape(-1) for a, b in zip(gp, gm)])
            hv = torch.cat([h.reshape(-1) for h in Hv])
            rel = float((fd - hv).norm() / hv.norm())
            assert rel < 1e-5, rel
        assert abs(est[n] - tr) <= 0.15 * abs(tr), (est[n], tr, se[n])


def test_weight_norm(pythia_add):
    w_all = weight_norm(pythia_add)
    w_2 = weight_norm(pythia_add, k_blocks=2)
    assert w_all > w_2 > 0
