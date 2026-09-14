"""whiteboxtopo.open — the open() pipeline and the scorecard (FINAL_SPEC 3, 5)."""
import json
import numpy as np
import pytest

from whiteboxtopo.invariants.model import ReluMLP
from whiteboxtopo.open import (Behavior, CellResult, Scorecard, OpenState, open as wb_open, grid,
                               step, step_gate, step_report, CELLS, STATUSES, GATE_RULE)

FAST = {"certify_n": 4, "hidden_symmetry": False, "gate_null": False, "seed": 0}


def tiny_mlp(seed=0):
    r = np.random.default_rng(seed)
    return ReluMLP([r.standard_normal((4, 8)) / 2, r.standard_normal((8, 6)) / 2,
                    r.standard_normal((6, 3)) / 2],
                   [r.standard_normal(8) * 0.1, r.standard_normal(6) * 0.1, r.standard_normal(3) * 0.1])


def behavior(n=120, seed=0):
    X = np.random.default_rng(seed + 1).standard_normal((n, 4))
    return Behavior("random_batch", X, (X.sum(1) > 0).astype(int), kind="general")


# ---------------------------------------------------------------- Behavior
def test_behavior_validates_kind():
    with pytest.raises(ValueError):
        Behavior("b", np.zeros((3, 2)), kind="nonsense")
    b = behavior()
    assert b.n == 120 and b.to_dict()["kind"] == "general"


# ------------------------------------------------------------ the full run
def test_open_returns_a_complete_scorecard():
    card = wb_open(tiny_mlp(), behavior(), config=dict(FAST, model_name="tinyMLP"))
    assert isinstance(card, Scorecard)
    assert set(card.cells) == set(CELLS) and len(card.cells) == 8
    assert all(c.status in STATUSES for c in card.cells.values())
    assert all(isinstance(c, CellResult) and c.problem == p for p, c in card.cells.items())
    # every cell that is not certified says why
    for p, c in card.cells.items():
        if c.status != "certified":
            assert c.provenance, f"{p} has no provenance"
    # P0 always runs on a ReLU MLP (gauge-fixed weight graph, no torch instrumentation needed)
    assert card.cells["P0"].status in ("certified", "partial")
    assert card.cells["P0"].value in ("tractable", "marginal", "intractable")
    assert GATE_RULE in card.cells["P0"].evidence["rule"]
    # P3/P7 cannot be reached without checkpoints, and say so
    assert card.cells["P3"].status == "not_reached" and "checkpoint" in card.cells["P3"].provenance
    assert card.cells["P7"].status == "not_reached"


def test_open_outputs_render():
    card = wb_open(tiny_mlp(), behavior(), config=dict(FAST, model_name="tinyMLP"))
    row = card.to_grid()
    assert isinstance(row, list) and len(row) == 10 and row[0] == "tinyMLP"
    assert all(x in STATUSES for x in row[1:9])
    md = card.to_markdown()
    assert md.count("\n|") >= 9 and "degree of opening" in md
    js = json.loads(card.to_json())
    assert set(js["cells"]) == set(CELLS) and js["model"] == "tinyMLP"
    assert js["degree_of_opening"] is None or 0.0 <= js["degree_of_opening"] <= 1.0
    assert js["status_counts"]["certified"] + js["status_counts"]["not_reached"] \
        + js["status_counts"]["partial"] + js["status_counts"]["failed"] == 8
    g = grid([card, card])
    assert g.count("tinyMLP") == 2 and "1 - γ_cov" in g


def test_degree_of_opening_tracks_gamma():
    card = Scorecard(model="m", gamma_cov=0.25)
    assert card.degree_of_opening == pytest.approx(0.75)
    assert Scorecard(model="m").degree_of_opening is None       # unknown gap is never a zero gap


def test_seeds_give_a_mechanism_count():
    m = tiny_mlp()
    card = wb_open(m, behavior(), seeds=[m, tiny_mlp(3)], config=dict(FAST))
    mc = card.cells["P1"].evidence.get("mechanism_count")
    assert mc is not None and mc["N"] == 2 and mc["K"] == 2     # two different nets, two mechanisms


def test_glue_attaches_cech_numbers_to_P2():
    card = wb_open(tiny_mlp(), behavior(), config=dict(FAST, cover_k=4, cover_n_min=4))
    ev = card.cells["P2"].evidence
    assert "cech" in ev and "cover" in ev
    assert ev["cech"]["H0"] >= 1 and ev["cech"]["H1"] >= 0
    assert isinstance(ev["sanity_gate_H1_zero"], bool)


# -------------------------------------------- negatives are deliverables
def test_a_raising_step_marks_its_cells_not_reached():
    @step("boom", ("P0", "P4"))
    def boom(state):
        raise RuntimeError("planted failure")

    card = wb_open(tiny_mlp(), behavior(), steps=[boom], config=dict(FAST))
    assert card.cells["P0"].status == "not_reached"
    assert card.cells["P4"].status == "not_reached"
    assert "planted failure" in card.cells["P0"].provenance
    assert any("planted failure" in e for e in card.errors)
    assert all(c.status in STATUSES for c in card.cells.values())


def test_a_raising_step_downgrades_an_answered_cell_instead_of_wiping_it():
    @step("boom_after", ("P0",))
    def boom_after(state):
        raise RuntimeError("late failure")

    card = wb_open(tiny_mlp(), behavior(), steps=[step_gate, boom_after], config=dict(FAST))
    assert card.cells["P0"].status == "partial"          # certified -> partial, evidence kept
    assert card.cells["P0"].value in ("tractable", "marginal", "intractable")
    assert "late failure" in card.cells["P0"].provenance


def test_unreachable_modules_are_reported_not_raised(monkeypatch):
    import whiteboxtopo.open as O
    monkeypatch.setattr(O, "_lazy", lambda modpath: (_ for _ in ()).throw(ImportError(f"no {modpath}")))
    card = wb_open(tiny_mlp(), behavior(), config=dict(FAST))
    for p in ("P2", "P4", "P5"):
        assert card.cells[p].status == "not_reached"
        assert "unavailable (module pending)" in card.cells[p].provenance
    assert card.gamma_cov is None and card.degree_of_opening is None


def test_steps_can_be_named_and_report_always_runs():
    card = wb_open(tiny_mlp(), behavior(), steps=["step_gate"], config=dict(FAST))
    assert card.cells["P0"].status in ("certified", "partial")
    assert all(card.cells[p].status == "not_reached" for p in CELLS if p != "P0")
    with pytest.raises(ValueError):
        wb_open(tiny_mlp(), behavior(), steps=["not_a_step"], config=dict(FAST))


def test_open_state_helpers():
    st = OpenState(cells={}, provenance={})
    st.set_cell("P1", "certified", value=3.0, provenance="x")
    assert st.cell("P1").status == "certified"
    st.not_reached("P2", "because")
    assert st.cell("P2").status == "not_reached" and st.cell("P2").provenance == "not reached: because"
    with pytest.raises(ValueError):
        CellResult("P1", "bogus_status")
