"""GPT-NeoX / Pythia gauge backend (FINAL_SPEC §2.3 transformer gauge, §3 step 1).

Every renaming in ``whiteboxtopo.symmetry.neox`` is claimed *exact*. These tests check that
claim on a real model (EleutherAI/pythia-70m: 6 layers, 8 heads, d_head 64, rotary_pct 0.25
⇒ rotary_ndims 16 + 48 non-rotary tail dims, parallel residual, untied unembedding):

* each renaming kind separately and combined leaves the logits unchanged (fp32 floor);
* the same in float64, to show the fp32 numbers are a round-off floor and not an approximation;
* **kills**: permuting heads without permuting ``dense``; a Q/K map that does not commute with
  RoPE (an unrestricted GL(d_head) instead of the block-diagonal restriction); a V/O map applied
  to V but not compensated in ``dense`` — each must move the logits far more than the floor;
* ``inverse_on_units`` round-trips; ``sample(kind="combined")`` produces valid elements;
* ``invariants.audit_quantities`` runs on the backend.
"""
import glob
import os

import numpy as np
import pytest

from whiteboxtopo import symmetry as S
from whiteboxtopo.symmetry.group import GaugeElement

RTOL = 1e-4          # the bar in the task; the measured floor is ~1e-6 (fp32) / ~1e-14 (fp64)
# two short prompts, pythia tokenizer ids (content is irrelevant — only that they are real tokens)
TOKENS = np.array([[510, 3158, 8516, 30013, 4830, 689, 253, 22658, 4370, 15, 187, 42],
                   [1552, 310, 247, 1071, 273, 253, 14604, 1332, 13, 285, 352, 4245]])


def _snapshot():
    """A cached pythia-70m snapshot that actually carries weights (several revisions are cached)."""
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots/*/model.safetensors")))
    return os.path.dirname(hits[0]) if hits else None


@pytest.fixture(scope="module")
def pythia():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    path = _snapshot()
    if path is None:
        pytest.skip("EleutherAI/pythia-70m not cached")
    torch.set_num_threads(4)
    return tf.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()


def _reldev(model, g, tokens=TOKENS):
    be = S._backend(model)
    y0 = be.forward(model, tokens)
    y1 = be.forward(be.apply(model, g), tokens)
    return float(np.max(np.abs(y1 - y0)) / np.max(np.abs(y0)))


def _restrict(g, family):
    """The sub-element of ``g`` touching only keys whose family name is ``family``."""
    out = GaugeElement(kind=g.kind)
    name = lambda k: k[0] if isinstance(k, tuple) else k
    for src, dst in ((g.perms, out.perms), (g.scales, out.scales), (g.rots, out.rots)):
        for k, v in src.items():
            if name(k) == family:
                dst[k] = v
    return out


def test_dispatch_and_describe(pythia):
    be = S._backend(pythia)
    assert be.__name__.endswith("symmetry.neox")
    D = S.describe(pythia)
    assert D["type"] == "gpt_neox"
    assert (D["n_layers"], D["n_heads"], D["d_head"], D["d_model"], D["d_mlp"]) == (6, 8, 64, 512, 2048)
    assert D["rotary_ndims"] == 16 and D["n_pass"] == 48 and D["use_parallel_residual"]
    assert D["untied_unembedding"] and "lnf" in D["keys"]
    # known_dim = L * (H d_h^2 + H r + H n_pass^2 + 2d) + d   (V/O GL, per-plane (s, phi), tail GL, LNs, final LN)
    assert D["known_dim"] == 6 * (8 * 64 ** 2 + 8 * 16 + 8 * 48 ** 2 + 2 * 512) + 512
    assert any("GELU" in s for s in D["approximate_or_absent"])


@pytest.mark.parametrize("kind", ["perm", "scale", "rot", "combined"])
@pytest.mark.parametrize("orthogonal", [True, False])
def test_identity_per_kind(pythia, kind, orthogonal, record_property):
    """Each gauge kind alone leaves the logits unchanged (fp32 round-off floor)."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2):
        g = S._backend(pythia).sample(pythia, rng, kind=kind, orthogonal=orthogonal)
        worst = max(worst, _reldev(pythia, g))
    record_property(f"reldev_{kind}_orth{int(orthogonal)}", worst)
    print(f"\n[neox] kind={kind:9s} orthogonal={orthogonal}  max relative Δlogit = {worst:.3e}")
    assert worst < RTOL


@pytest.mark.parametrize("family", ["head", "mlp", "vo", "qk", "qkpass", "ln1", "ln2", "lnf"])
def test_identity_per_key_family(pythia, family, record_property):
    """Each individual key family (isolated from the others) is exact on its own."""
    rng = np.random.default_rng(1)
    g = _restrict(S._backend(pythia).sample(pythia, rng, kind="combined", orthogonal=False), family)
    assert not g.is_identity(), f"sampler produced no {family} keys"
    dev = _reldev(pythia, g)
    record_property(f"reldev_{family}", dev)
    print(f"\n[neox] family={family:7s} max relative Δlogit = {dev:.3e}")
    assert dev < RTOL


def test_float64_eager_is_bitwise_exact():
    """The fp32 numbers above are round-off, not an approximate symmetry.

    In float64 with the *eager* attention kernel the combined renaming leaves the logits
    **bitwise identical** (measured max |Δ| = 0.0 exactly, all 2 x 12 x 50304 of them).
    With the default `sdpa` kernel the same element floors at ~8e-9 instead, because the
    fused/blocked kernel is not bit-reproducible under a reordering of its inputs (median
    deviation is still 0 and only ~2e-6 of the logits move at all) — that is a property of
    the kernel, not of the symmetry.
    """
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    path = _snapshot()
    if path is None:
        pytest.skip("EleutherAI/pythia-70m not cached")
    torch.set_num_threads(4)
    m64 = tf.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float64, attn_implementation="eager").eval()
    rng = np.random.default_rng(2)
    for kind in ("perm", "scale", "rot", "combined"):
        g = S._backend(m64).sample(m64, rng, kind=kind, orthogonal=False)
        dev = _reldev(m64, g)
        print(f"\n[neox] float64/eager kind={kind:9s} max relative Δlogit = {dev:.3e}")
        assert dev < 1e-12


# ----------------------------------------------------------------- kills (the test can fail)
def test_kill_head_permutation_without_dense(pythia):
    """Permuting the heads of ``query_key_value`` but NOT the columns of ``dense`` is not a
    symmetry — if this passed, the identity tests above would be vacuous."""
    torch = pytest.importorskip("torch")
    import copy
    D = S.describe(pythia); H, dh = D["n_heads"], D["d_head"]
    broken = copy.deepcopy(pythia)
    p = np.random.default_rng(3).permutation(H)
    with torch.no_grad():
        for blk in S._backend(broken)._blocks(broken):
            W = blk.attention.query_key_value.weight; b = blk.attention.query_key_value.bias
            rows = torch.as_tensor(np.concatenate([np.arange(j * 3 * dh, (j + 1) * 3 * dh) for j in p]))
            W.copy_(W[rows]); b.copy_(b[rows])          # dense.weight columns deliberately left alone
    be = S._backend(pythia)
    y0 = be.forward(pythia, TOKENS); y1 = be.forward(broken, TOKENS)
    dev = float(np.max(np.abs(y1 - y0)) / np.max(np.abs(y0)))
    print(f"\n[neox] KILL head-perm-without-dense relative Δlogit = {dev:.3e}")
    assert dev > 1e-2


def test_kill_qk_map_that_does_not_commute_with_rope(pythia):
    """An unrestricted GL(d_head) on Q/K (instead of blockdiag(rotary planes, tail)) breaks RoPE:
    the rotary restriction in ``neox.py`` is necessary, not decorative."""
    torch = pytest.importorskip("torch")
    import copy
    D = S.describe(pythia); H, dh = D["n_heads"], D["d_head"]
    from whiteboxtopo.symmetry.transformer import _rand_rot
    rng = np.random.default_rng(4)
    broken = copy.deepcopy(pythia)
    with torch.no_grad():
        for blk in S._backend(broken)._blocks(broken):
            W = blk.attention.query_key_value.weight; b = blk.attention.query_key_value.bias
            for h in range(H):
                M = torch.as_tensor(_rand_rot(rng, dh, True), dtype=torch.float64)   # full GL(64), mixes rotary planes
                MinvT = torch.linalg.inv(M).T
                base = h * 3 * dh
                q = slice(base, base + dh); k = slice(base + dh, base + 2 * dh)
                W[q] = (M @ W[q].double()).float(); W[k] = (MinvT @ W[k].double()).float()
                b[q] = (M @ b[q].double()).float(); b[k] = (MinvT @ b[k].double()).float()
    be = S._backend(pythia)
    y0 = be.forward(pythia, TOKENS); y1 = be.forward(broken, TOKENS)
    dev = float(np.max(np.abs(y1 - y0)) / np.max(np.abs(y0)))
    print(f"\n[neox] KILL unrestricted-GL-on-QK relative Δlogit = {dev:.3e}")
    assert dev > 1e-2


def test_kill_vo_without_output_compensation(pythia):
    """Rotating V without applying B^{-1} to the head's columns of ``dense`` changes the function."""
    torch = pytest.importorskip("torch")
    import copy
    D = S.describe(pythia); H, dh = D["n_heads"], D["d_head"]
    from whiteboxtopo.symmetry.transformer import _rand_rot
    rng = np.random.default_rng(5)
    broken = copy.deepcopy(pythia)
    with torch.no_grad():
        for blk in S._backend(broken)._blocks(broken):
            W = blk.attention.query_key_value.weight; b = blk.attention.query_key_value.bias
            for h in range(H):
                B = torch.as_tensor(_rand_rot(rng, dh, True), dtype=torch.float64)
                v = slice(h * 3 * dh + 2 * dh, (h + 1) * 3 * dh)
                W[v] = (B @ W[v].double()).float(); b[v] = (B @ b[v].double()).float()
    be = S._backend(pythia)
    y0 = be.forward(pythia, TOKENS); y1 = be.forward(broken, TOKENS)
    dev = float(np.max(np.abs(y1 - y0)) / np.max(np.abs(y0)))
    print(f"\n[neox] KILL V-rotation-without-dense relative Δlogit = {dev:.3e}")
    assert dev > 1e-2


# ----------------------------------------------------------------- bookkeeping
def test_sample_combined_is_a_valid_element(pythia):
    D = S.describe(pythia)
    L, H, dh, d, m, r, npass = (D["n_layers"], D["n_heads"], D["d_head"], D["d_model"],
                                D["d_mlp"], D["rotary_ndims"], D["n_pass"])
    g = S._backend(pythia).sample(pythia, np.random.default_rng(6), kind="combined", orthogonal=False)
    assert g.kind == "combined" and not g.is_identity()
    for l in range(L):
        assert sorted(g.perms[("head", l)].tolist()) == list(range(H))
        assert sorted(g.perms[("mlp", l)].tolist()) == list(range(m))
        assert g.scales[("ln1", l)].shape == (d,) and (g.scales[("ln1", l)] > 0).all()
        assert g.scales[("ln2", l)].shape == (d,) and (g.scales[("ln2", l)] > 0).all()
        assert g.scales[("qk", l)].shape == (H, r // 2) and (g.scales[("qk", l)] > 0).all()
        assert g.rots[("qk", l)].shape == (H, r // 2)
        assert len(g.rots[("vo", l)]) == H
        assert all(np.abs(np.linalg.det(R)) > 1e-8 for R in g.rots[("vo", l)])
        assert len(g.rots[("qkpass", l)]) == H and g.rots[("qkpass", l)][0].shape == (npass, npass)
    assert g.scales["lnf"].shape == (d,)
    # no MLP unit scalings are ever sampled (GELU is not homogeneous)
    assert not any(isinstance(k, tuple) and k[0] == "mlp" for k in g.scales)
    with pytest.raises(NotImplementedError):
        S.uniform_scaling(pythia, 2.0)


def test_inverse_on_units_round_trips(pythia):
    """``inverse_on_units`` takes a per-unit array measured on the renamed model back to base order
    (MLP hidden units: permutation) and base magnitude (LayerNorm outputs: scaling)."""
    rng = np.random.default_rng(7)
    D = S.describe(pythia); m, d = D["d_mlp"], D["d_model"]
    g = S._backend(pythia).sample(pythia, rng, kind="combined", orthogonal=False)
    base_mlp = rng.normal(size=m)
    renamed = base_mlp[g.perms[("mlp", 2)]]                                   # what the renamed model reports
    assert np.allclose(S.inverse_on_units(g, ("mlp", 2), renamed, axis=0), base_mlp)
    base_ln = rng.normal(size=d)
    renamed_ln = base_ln * g.scales[("ln1", 1)]
    back = S.inverse_on_units(g, ("ln1", 1), renamed_ln, axis=0, scale_power=1.0)
    assert np.allclose(back, base_ln)
    # 2-d arrays, unit axis 1
    V = rng.normal(size=(5, m))
    assert np.allclose(S.inverse_on_units(g, ("mlp", 2), V[:, g.perms[("mlp", 2)]], axis=1), V)


def test_audit_quantities_runs_on_neox(pythia):
    """``invariants.audit_quantities`` drives this backend with no extra hook: an exactly gauge-invariant
    quantity (the residual-stream norm) is `invariant`, a gauge-dependent one (a raw MLP hidden coordinate)
    is not."""
    torch = pytest.importorskip("torch")
    from whiteboxtopo.invariants import audit_quantities

    def quantities(model):
        be = S._backend(model)
        with torch.no_grad():
            out = model(torch.as_tensor(TOKENS[:1], dtype=torch.long), output_hidden_states=True)
            h = out.hidden_states[-1][0].double().numpy()   # final residual stream: gauge-invariant
            acts = []
            blk = be._blocks(model)[3]
            x = torch.as_tensor(out.hidden_states[3][:1])
            a = blk.mlp.dense_h_to_4h(blk.post_attention_layernorm(x))[0].double().numpy()
        return {"resid_norm": np.linalg.norm(h, axis=-1), "mlp_preact_first8": a[-1, :8]}

    res = audit_quantities(quantities, pythia, n=3, kinds=("perm", "combined"), orthogonal=False, layers=[3])
    print("\n[neox] audit_quantities:",
          {k: (v.verdict, {kk: round(float(c), 3) for kk, c in v.median_cv.items()}) for k, v in res.items()})
    assert res["resid_norm"].verdict == "invariant"
    assert res["mlp_preact_first8"].verdict == "artifact"
