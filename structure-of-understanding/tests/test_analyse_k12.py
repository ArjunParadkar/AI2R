"""Synthetic test of the Phase 2 analysis units (no model, no real data)."""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _k1(nodes, n_cand=100, acc=0.9):
    outs = {"out:out0": ["C0"], "out:out1": ["C1", "C3"], "out:out2": ["C2", "C4"]} if nodes else {}
    gin = {k: 0.1 for k in ("a0", "a1", "a2", "b0", "b1", "b2")} if nodes else {}
    return dict(meta=dict(train_accuracy={"out0": acc, "out1": acc}), nodes=nodes, outputs=outs, input_generality=gin,
                verdicts={}, n_candidates=n_cand)


def _k2(rng, n=400, strength=1.0, acc=0.9):
    y = rng.random(n) < 0.5
    t = dict(correct=y, depth_lambda=y * strength + rng.standard_normal(n), ccgp=rng.standard_normal(n),
             neg_sharpness=rng.standard_normal(n), max_softmax=rng.standard_normal(n))
    return dict(meta=dict(train_accuracy={"out0": acc}), table=t)


def _cell(root, name, k1=None, k2=None):
    d = root / name
    d.mkdir()
    if k1 is not None:
        (d / "k1.json").write_text(json.dumps(k1))
    if k2 is not None:
        pickle.dump(k2, open(d / "k2.pkl", "wb"))


def test_units_and_decisions(tmp_path):
    rng = np.random.default_rng(0)
    good = {f"C{i}": dict(label=l, g=0.5, parents=p) for i, (l, p) in enumerate(
        [("s0", ["a0", "b0"]), ("s1", ["a1", "b1"]), ("s2", ["a2", "b2"]), ("c0", ["C0"]), ("c1", ["C1", "C3"])])}
    _cell(tmp_path, "Qwen__Qwen2.5-7B_add3_s0", _k1(good), _k2(rng))
    _cell(tmp_path, "Qwen__Qwen2.5-7B_add3_s0_untrained", _k1({}, n_cand=100))
    _cell(tmp_path, "Qwen__Qwen2.5-7B_add3_s1", _k1(good), _k2(rng, strength=-5))     # not a unit
    _cell(tmp_path, "EleutherAI__pythia-160m_add2_s0", _k1({}), _k2(rng, strength=-5))  # development model
    _cell(tmp_path, "EleutherAI__pythia-1.4b_add3_s0", _k1({}, acc=0.1))                # gate failed
    _cell(tmp_path, "Qwen__Qwen2.5-32B_add3_s0_8bit", _k1(good), _k2(rng))              # no quant check -> not a unit
    for m in ("allenai__OLMo-2-1124-7B", "mistralai__Mistral-7B-v0.1", "Qwen__Qwen2.5-14B"):
        for t in ("mul21", "agree"):
            _cell(tmp_path, f"{m}_{t}_s0", _k1({}), _k2(rng))
    out = tmp_path / "r.json"
    subprocess.run([sys.executable, "-m", "bench.analyse_k12", str(tmp_path), "--out", str(out)], cwd=ROOT,
                   check=True, capture_output=True)
    r = json.loads(out.read_text())
    units = r["K2_decision"]["units"]
    assert "Qwen__Qwen2.5-7B_add3_s1" not in units and "EleutherAI__pythia-160m_add2_s0" not in units
    assert "Qwen__Qwen2.5-32B_add3_s0_8bit" not in units
    assert len(units) == 7 and r["K2_decision"]["n_cells"] == 7
    k1 = {c["cell"]: c for c in r["K1_decision"]["cells"]}
    assert k1["Qwen__Qwen2.5-7B_add3_s0"]["K1_pass"] and k1["Qwen__Qwen2.5-7B_add3_s0"]["null_fp"] == 0.0
    assert not k1["EleutherAI__pythia-1.4b_add3_s0"]["K1_pass"]
    assert r["K1_decision"]["verdict"] == "PASS" and r["K1_decision"]["family_pass"]["Qwen2.5"]
    assert r["quantization_check"]["verdict"] == "NOT AVAILABLE"
