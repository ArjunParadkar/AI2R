# E5 — Cross-substrate legibility: FINDINGS

> **NOT FOR EXTERNAL CLAIMS until cleared with the Fiete lab.** Every number,
> figure and sentence below is preliminary, for discussion with the owner's
> mentors only. Nothing here may be cited, posted or shipped. This applies
> to the fly connectome result in particular (data owned by the Fiete-lab
> collaboration, `MIT Fiete Lab/InvProj1`).

**Status: COMPLETE (2026-08-30).** One run (with a mid-run reboot, see
Amendment 2), 32.4 min wall after resume, on CPU (4 threads, no GPU). 10
seeds of the main N=148 vanilla-RNN arm, 3 seeds each of GRU/size-arms/
controls, the fly HD core (148 neurons) + full connectome (439) + a planted
ring attractor, all measured on E4's exact instrument
(`e4lib.graph`/`e4lib.discovery`, unforked — `e5lib/profile.py` is a thin
adapter). Files: `results/e5_points.csv`, `e5_points_raw.csv`,
`e5_sweep.csv`, `e5_nulls.csv`/`e5_nulls_raw.csv`, `e5_tests.csv`,
`e5_rnn_training.csv`, `fig_e5_phase.png`, `fig_e5_support.png`, `run.log`,
`analyze_stdout.txt`. Tests: `tests/test_e5.py` **4/4 green** (see §Tests —
one stale pre-Amendment-1 literal was fixed during close-out).

## Verdict

| Hypothesis | SPEC criterion | Result (d=10%, primary) |
|---|---|---|
| **H-E5a** | "Pass requires both: (i) d_M(connectome,𝒮) ≤ 95th pctile of leave-one-out d_M of 𝒮's own members, and (ii) d_M(connectome,𝒮) < d_M(connectome,ℱ)." | **FAILS both.** d_M(fly, 𝒮)=3.526 > loo95=2.589 (i fails); d_M(fly,ℱ)=2.662 < d_M(fly,𝒮) (ii fails — the fly is *closer to the failure cluster*). Secondary (5-NN mean S): the fly's 5 Mahalanobis-nearest E4 neighbours (all `mlp\|copy` variants) have mean S=0.222, below the learned-model ensemble mean S=0.391. |
| **H-E5b** | "Pass = d_M(RNN̄, fly) < median-of-within-architecture-medians (thr=1.113 at d10)." | **FAILS, both graphs.** Weight-graph RNN centroid: d_M=3.958 (3.6× the threshold). Functional (EAP) graph centroid: d_M=3.020 (2.7× threshold). Secondary NN test also fails: the fly ranks 47th (weight) / 73rd (functional) of 88 nearest-neighbour ranks, not top-5. |
| **H-E5c** | "Agree if d_M(weight, functional) per seed < H-E5b threshold." (exploratory, no pass/fail gate, reported either way) | **Weight and functional placements disagree for every one of the 10 main seeds** at every threshold (frac_agree_main = 0/10 at d5, d10, d20). Mean per-seed d_M(weight,functional)=2.976 (d10), ~3× the threshold. Edge-level agreement is moderate, not strong: Spearman ρ(|weight edge|, |functional edge|) = 0.512 (mean over seeds). Per SPEC's own framing: **since weight and functional graphs disagree, the fly-vs-RNN comparison in H-E5a/b is a weight-graph-only comparison** — there is no independent "functional" placement of biology to check it against, and the RNN's own functional placement is a different (though still non-matching) point from its weight placement.|

**Threshold robustness (kill 5):** all three verdicts are unchanged at
d=5% and d=20% (H-E5a fails both criteria at every d; H-E5b fails at every
d for both graphs, d_M ranging 2.73–5.30 across the six weight/functional ×
d5/d20 cells vs thresholds 1.03–1.35; H-E5c frac_agree_main=0 at every
d). Nothing here is threshold-dependent.

**Bottom line: the fly HD ring-attractor core does not land in E4's
"legible" region, and the task-matched artificial RNN does not land near
the fly.** This is a double negative, not a single failure to replicate:
neither the biology-in-legible-region claim nor the RNN-approximates-biology
claim survives contact with E4's own axes.

## Kill-control story: the structure is real, E4's axes just don't see it as the connectome's

The literal SPEC-mandated kill attempts all ran and are informative, and
they point the same direction: **the connectome is decisively non-random
structure, but the specific numbers E4's pipeline reports (Q₁₀, tw₁₀/N)
are dominated by size/density effects that make direct comparison across
substrates close to meaningless without the size confound in the
foreground.**

1. **Rewiring/ER nulls (kill 1/4ii) — connectome is not degree- or
   density-driven modularity.** All three null families put the observed
   fly-core values many SDs from chance: Q₁₀ z = +42.9 (directed rewire),
   +37.6 (undirected top-10% rewire), +36.3 (ER, matched N and edge count);
   tw₁₀/N z = −15.6, −15.5, −16.6 respectively (all p ≈ 0.0099, the floor
   of 100 draws). This is unambiguous: whatever region of (Q,tw) space the
   fly core sits in, it did not get there by chance given its degree
   sequence or density. SPEC's own gate ("if |z(Q₁₀)| < 2, do not report as
   structure") is not triggered — the opposite, extremely far from it.
2. **The same rewiring null applied to the trained RNNs is much weaker
   (kill 2)** — an asymmetry worth flagging on its own: RNN weight-graph
   z(Q₁₀) mean = 1.185 (range −0.92 to 3.26 across 10 seeds), and only
   **3/10 seeds** clear z > 2. The fly connectome's modularity is far
   stronger relative to its own null than any trained RNN's is relative to
   its own null. Trained RNNs are, on this measure, close to their own
   degree/density expectation; the fly is not. That is itself evidence
   against "the RNN approximates the fly's structure" — even before
   comparing their absolute positions.
3. **Label-shuffle NMI (kill 3) is a genuine split result.** Leiden's
   10%-density partition of the fly core (8 communities) has **no**
   relationship to cell type: NMI(partition, type) = 0.048 vs a
   label-permutation null mean 0.059, p = 0.727 (not significant — type is
   not what Leiden is finding at this threshold). But it strongly tracks
   the protocerebral-bridge glomerulus/wedge label: NMI(partition, wedge) =
   0.556 vs null mean 0.248, p = 0.001. **The communities Leiden finds are
   retinotopic-position modules (which PB wedge a neuron belongs to), not
   cell-type modules** — biologically sensible for a ring attractor (each
   wedge is a local copy of the EPG/PEG/PEN/Δ7 circuit) but a genuinely
   different kind of "modularity" than E4's Q was built to characterise
   (feature/layer separability in a trained network). On the full 439-
   neuron matrix this reverses in relative strength (nmi_type=0.740,
   nmi_wedge=0.335) — the 148-core truncation trades a type signal for a
   wedge signal, a sensitivity result in its own right (see below).
4. **Size confound (kill 4) — this is the finding that must be read first
   when interpreting H-E5a/b.** Across the vanilla-RNN size arms (N ∈
   {64,148,256}, matched in every other respect), tw₁₀/N rises monotonically
   with N (Spearman ρ = 0.872 vs N across the 16 vanilla models; means
   0.333/0.545/0.664 at N=64/148/256) and **d_M(RNN centroid, fly) grows
   mechanically with N**: 1.734 (N=64) → 3.958 (N=148, the task-matched
   size) → 5.151 (N=256). The RNN that is *closest* to the fly on these
   axes is the smallest one, which is not size-matched to the connectome at
   all and is a worse task-performance match (still learns, acc_tail≈0.73).
   This is the opposite of what the size-matching in SPEC's RNN design was
   meant to buy: matching N=148 to the connectome's N does not make the RNN
   land near the connectome; if anything the relationship goes the wrong
   way. Q₁₀ itself falls with N (ρ=−0.55; 0.363→0.234→0.221) while n_edges10
   rises exactly with N (ρ=1.0, since density is defined relative to
   existing edges and the RNN is dense) — **tw₁₀/N in this pipeline is
   substantially an artifact of how many edges are being fed to the
   min-fill bound at a fixed relative threshold, not a substrate-specific
   circuit property.** The matched-edge-count secondary column confirms
   this is not merely an artifact of the *relative* threshold: even at
   equal edge count (607 edges) the RNN's Q_matched=0.329 and
   tw_matched_norm=0.369 remain far from the fly's Q₁₀=0.581,
   tw₁₀/N=0.236 — so it is not purely a "different edge count" problem,
   but the edge-count/density confound is real and additive to whatever
   the remaining gap is.
5. **The planted ring attractor is the closest artificial point to the
   fly** (d_M = 1.03, versus 3.96–4.57 for every trained/untrained/
   GRU/shuffled RNN arm), and closer than any trained network. A hand-built
   "legible by construction" bump attractor (Q₁₀=0.550, tw₁₀/N=0.182) sits
   near the fly's own coordinates (Q₁₀=0.581, tw₁₀/N=0.236) even though it
   was never trained and has no forward-pass task performance. This is
   consistent with the fly's Q/tw signature coming from its **spatial/
   ring topology** (each PB wedge locally recurrent, weakly coupled to
   neighbours) rather than from anything a gradient-trained integrator
   learns to do with a dense unconstrained recurrent matrix.
6. **Sensitivity to the 148-core truncation (kill 6):** the full 439-neuron
   matrix sits at Q₁₀=0.578, tw₁₀/N=0.201, sparsity=0.294 (vs core's
   Q₁₀=0.581, tw₁₀/N=0.236, sparsity=0.073) — d_M(full, core)=1.38, i.e.
   the truncation is a real but moderate move in this 3-D space (smaller
   than the fly-vs-RNN gap by ~3×), and it does **not** flip either
   verdict: d_M(full, 𝒮)=2.84 > loo95, d_M(full, ℱ)=1.96 < d_M(full,𝒮), so
   H-E5a still fails on the full network, slightly worse than the core.
   The sparsity axis moves the most (0.073→0.294) since the full 439×439
   matrix includes many non-HD-core rows/columns.
7. **RNN discovery is much smaller than E4's typical learned circuit.**
   S_raw (fraction of units the greedy prune keeps at F≥0.9) averages 0.103
   ± 0.014 across the 10 main seeds — i.e. E4's discovery pipeline
   compresses these HD-integrator RNNs to under 15 units out of 148 — well
   below E4's learned-model mean S=0.391. Task accuracy at discovery time
   (acc_tail=0.761 mean) is comfortably above the guard (≥0.5) but well
   short of ceiling; this small but real discoverable circuit (mean
   Jaccard across 3 discovery seeds = 0.932, so the discovered subset is
   highly stable across seeds) is itself interesting and orthogonal to the
   phase-diagram placement question — it says these RNNs *are* compressible
   by E4's own method even though their (Q,tw,sparsity) coordinates don't
   land near the fly's.

## Limitations (own read, not sugar-coated)

* **10 seeds** for the main arm (3 for every control/size arm). CIs on the
  centroid distances were not bootstrapped in `analyze_e5.py` (the SPEC
  criterion is a point comparison against a fixed threshold, not a CI
  test) — the `d_seed_fly_min/max` spread reported (weight: 3.79–4.14 at
  d10) shows the 10 seeds are tight relative to the gap to the threshold
  (1.11), so more seeds would not close this gap, but 3-seed arms (GRU,
  size, controls) are underpowered for anything beyond the qualitative
  read given here.
* **One task family.** Everything is HD integration with K=16 bins;
  whether the RNN's placement would change under a different task or
  architecture family is untested (that is exactly what E4's own ensemble
  already showed is architecture-dependent, and E5 has only one
  architecture per cell type here).
* **CPU-scale RNNs, one architecture per cell/size cell.** N∈{64,148,256}
  vanilla RNN, 3-gate GRU at N=148 only; no transformer arm (a transformer
  recurrent state doesn't map onto a single N×N weight graph the way SPEC's
  §Data (b) [JC] requires, so it was out of scope by design, not an
  oversight — but it means H-E5b's "same region as one E4 architecture's
  own scatter" threshold is computed from E4's mlp/tf2/tf4 architectures,
  none of which is the RNN's own architecture family, and E5 has no
  RNN-architecture analogue of "within-architecture scatter" to compare
  against).
* **Accuracy ceiling.** Main-arm acc_tail sits at 0.746–0.774 (mean 0.761),
  not near 1.0 — the leaky-CTRNN + curriculum recipe (Amendment 1) reaches
  the ≥0.5 learned guard comfortably and gets well past chance (1/16=0.0625)
  but does not reach the ≥0.9 early-stop target within the 4000-step CPU
  budget for most seeds (only the size-arm N=64 seeds and the GRU arm
  clear 0.9; GRU acc_tail 0.88–0.90). One N=256 seed (`s1`) never reached
  the learned guard within budget (acc_tail=0.474, `learned=False`) and is
  still included in the N=256 size-arm centroid (kill 4) — a 1/3 failure
  rate at the largest size arm that itself illustrates the CPU-budget
  constraint, and slightly weakens the size-arm centroid's precision at
  N=256 (though the N=64→148→256 monotonic trend in tw10_norm/d_centroid
  does not depend on that one seed alone: N=256's own two learners already
  show the same direction, see `e5_rnn_training.csv`).
* **No forward pass on the connectome** — as pre-registered, biology gets
  structural coordinates only; there is no way to run E4's discovery
  pipeline or a functional graph on the fly, so H-E5a is a weight-graph-
  only test by construction and H-E5c's "functional vs weight" comparison
  is RNN-only and cannot itself be checked against biology.
* **Type-block/NMI results depend on the 148-core choice** (see kill 6):
  the type signal that is weak in the core (NMI=0.048, ns) is strong in the
  full 439 (NMI=0.740) — a caution against over-reading the core-only NMI
  numbers as "the fly connectome has no type structure"; it has type
  structure, just not at the 10%-density Leiden partition of the 148-core
  subgraph specifically.

## Amendments (relative to SPEC.md)

SPEC.md carries **Amendment 1** (2026-08-30, pre-run: leaky step, σ=0.08,
K=16, T-curriculum — all reflected in `e5lib/hdtask.py` and `e5lib/rnn.py`
and used throughout the run and this document) and **Amendment 2**
(2026-08-30, post-run, informational: the mid-run reboot and the
checkpoint-resume code path — resumed models' eval metrics re-derived
deterministically, bit-for-bit identical to pre-reboot values, `steps=-1`
markers in `e5_rnn_training.csv`; no result changed).

## Tests

`OMP_NUM_THREADS=1 /home/god/whitebox-lib/.venv/bin/python -m pytest tests/ -q`
→ **4 passed** (`tests/test_e5.py`):

* `test_connectome_loads_with_expected_block_signs` — PASS.
* `test_planted_ring_gets_high_Q` — PASS.
* `test_rewiring_null_runs_and_preserves_degrees` — PASS.
* `test_rnn_interface_and_ablation` — PASS after a one-line close-out fix:
  the test asserted `logits.shape == (8, 20, 32)`, a stale hardcoded
  literal from the pre-Amendment-1 K=32 design; `HDRNN`/`HDTask` both
  default to K=16 per Amendment 1, and every actual measurement in
  `results/` was produced with K=16 consistently. The literal was fixed to
  16 (test-file bug, not a code or result bug — the remaining assertions
  in that test do not depend on K).

## Library consequence for `whitebox-lib`

**No new API, and this reinforces rather than adds to E4's negative.** E5
was explicitly the transfer test of E4's `legibility_profile` measures
(Q/tw/sparsity) to a substrate E4 never touched (a real biological
connectome) and to a task-matched but untrained-on-E4's-benchmark RNN
family. The result: the instrument runs cleanly across substrates (no
adapter bugs, no NaNs, kill controls behave sensibly — the fly is
decisively non-random by every rewiring/ER null even though its absolute
Q/tw numbers don't match anything in E4's ensemble), but **the coordinates
it produces do not transfer as a "legible region" locator**: they are
dominated by network size/density (kill 4, tw10_norm ρ=0.87 vs N) at least
as much as by anything substrate-specific, and the fly's own strong,
non-random modularity (kill 1–3) tracks spatial/wedge structure that E4's
Q was never built to detect (kill 3's NMI split), not a "trained-network
attribution graph" kind of modularity. My own read: this is exactly the
kind of result that should stay a documented negative in `E5/FINDINGS.md`
and not motivate a `whitebox.legibility.cross_substrate_distance()` or
similar API — shipping a "distance to biology" score built on these three
coordinates would imply a transferability that kill attempt 4 actively
contradicts (the score is largely reporting N). E4 already concluded
`e4lib.legibility_profile` ships as "measures only, no prediction"; E5
adds the same caveat one level up — measures only, **not comparable across
substrates without first controlling for size**, which the library does
not currently do and which SPEC's matched-edge-count column shows is only
a partial fix (kill 4's matched-edge Q/tw are still far apart). If a future
Stage wants a genuine cross-substrate legibility instrument, the
prerequisite is a size-normalized or size-conditioned version of Q/tw
(e.g. partial correlation controlling N, or Q/tw regressed against
matched random-graph nulls of the *same* N and density before comparing
across substrates) — not present in `whitebox-lib` today, and not built
here since SPEC scoped E5 as a measurement/comparison exercise, not an
instrument-design one.
