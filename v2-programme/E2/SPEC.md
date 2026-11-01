# E2 — Gauge-invariant observables (pre-registration)

Status: SPEC written 2026-08-30 **before** any E2 analysis code exists or runs.
Changes after the first run are logged as numbered, dated amendments at the
bottom, never edited in place. Rule-5 pause points were lifted by the owner for
E2; judgment calls are stated here instead of asked.

## Question

Which interpretability quantities are constant on the gauge orbit of a trained
network — and are therefore well-defined statements about the *function* — and
which popular ones are properties of one arbitrary representative? Frame (from
PROGRAM.md): an interpretability claim is well-defined only if it is
gauge-invariant; this is the empirical audit. The symmetry facts are textbook
(2025 survey "Symmetry in Neural Network Parameter Spaces"; Grigsby, Lindsey,
Meyerhoff, Wu 2022 on functional dimension); the contribution is the measured
table and the reusable audit tool, not the group theory.

## Networks

Every network saved by E1 in `E1/results/networks/*.npz` (E1 saves converged
seeds only; convergence is re-checked on load: zero error and |logit| ≥ 0.5 on
the truth table, anything else is dropped and counted). Snapshot taken at the
start of the E2 run; the still-running E1 `_part2` cells contribute whatever
they have written by then (count recorded in FINDINGS). Grouping: function ×
depth × width, as in E1.

Untrained control networks: for every (function, d, w) cell, 24 random-init
networks of the same shape (E1's init recipe, seed 1000+i).

## Symmetry group and orbit sampling

G = hidden-unit permutations within each hidden layer × per-hidden-unit positive
scalings (w_in, b) → α(w_in, b), w_out → w_out/α. Sampled with E1's
`e1lib.gauge.apply_gauge` (verified in E1's control tests). Three groups are
sampled separately so the table says WHICH symmetry breaks a quantity:

* `perm`: uniform random permutations per layer, all α = 1;
* `scale`: identity permutation, log α ~ N(0, 1) i.i.d. per unit;
* `combined`: both, independently.

n_reps = 100 representatives per network per group, rng seeded by
(network index, group). Representative 0 is the network itself.

Additionally a **uniform-scaling sweep**: all units of one hidden layer ℓ
scaled by the same α ∈ {10^-2, 10^-1.5, …, 10^2} (9 values), other layers
untouched, to fit covariance exponents (below).

**Hidden symmetries: not sampled.** E1 found Jacobian-rank deficits beyond
scalings+dead units in 92 % of nets. Those are *infinitesimal* flat directions
of the realisation map at θ; a finite step along a null-space vector changes f
at second order unless the direction integrates to a genuine symmetry, which is
not known and was explicitly not characterised in E1. Sampling them would
confound "the quantity moved" with "the function moved", which is exactly what
a CV audit cannot separate. They are therefore excluded, and the scope of every
verdict below is "invariance under the known group G". Stated, not silently
dropped.

## Candidate quantities

All computed with the same code path on the base network and on each
representative. Activation point clouds are the post-ReLU activations of one
hidden layer, one point per input; two input sets are used (both are cheap):

* `tt`: the 2^n truth-table inputs (8 or 16 points);
* `box`: 200 points uniform in [−1, 2]^n (E1's functional-dimension box),
  fixed per network (seeded), so the only thing that varies across
  representatives is the gauge.

Persistence: v1 `whitebox.compute_diagrams` (Vietoris–Rips, H0 and H1, VR
threshold at the 95th percentile of pairwise distances — v1's convention, kept
so E2's TC is the library's TC; the percentile threshold is itself
scale-covariant, so it does not confound the scaling test). Metrics from v1:
`topological_complexity` (TC = Σ lifetime², per dimension), `betti_at_scale` at
`median_scale` (β0, β1 at the median bar midpoint), `persistent_entropy` (PE,
scale-free by construction — included as the expected scaling-invariant
companion).

**Expected invariant (the candidate list):**

| id | quantity | expected under perm | under scale |
|---|---|---|---|
| I1 | logits on the truth table (sanity: f itself) | exact | exact |
| I2 | input-gradient of the logit at each truth-table point (the per-activation-region linear map of the ReLU net) | exact | exact |
| I3 | product of weight matrices W0·W1·…·Wd (the *linear part* only — for a ReLU net this is not the function, and it is only exactly the end-to-end map inside the all-active region; stated) | exact | exact (α cancels) |
| I4 | singular value spectrum of each hidden layer map W_ℓ (and of the output row) | exact | **not** |
| I5 | TC(H0), TC(H1) of layer-ℓ activation cloud, `tt` and `box` | exact | **not** (per-unit α) |
| I6 | β0, β1 at median scale, `tt` and `box` | exact | not (per-unit α) |
| I7 | PE(H0), PE(H1) | exact | not (per-unit α) |
| I8 | TC on the **per-unit standardised** cloud (each unit z-scored over the inputs; dead units dropped) — an exploratory *repair*: standardisation removes the per-unit scale | exact | exact (up to float) |

**Expected NOT invariant (the indictment list).** Each is reported in two
readings, because the failure mode differs: *raw-index* (the quantity is a
vector indexed by neuron position — the way "neuron 3" is used in a paper),
and *identity-tracked* (the permutation is undone before comparing, so only
the scaling can act) plus *sorted* (a canonicalisation). The claim under audit
is the raw-index one; the other readings say which repair works.

| id | quantity |
|---|---|
| N1 | per-neuron mean |activation| over the truth table |
| N2 | per-neuron gradient × activation attribution to the logit, summed over the truth table (per input, then summed) |
| N3 | per-neuron zero-ablation effect: mean over inputs of |logit − logit_ablated| |
| N4 | raw weight magnitudes: per-neuron incoming-weight norm ‖(w_in, b)‖, and all |W| entries |
| N5 | "top neuron": argmax over hidden units of N1 (magnitude-based) and of N3 (ablation-based); reported as agreement rate with the base network |
| N6 | per-neuron gate assignment: binarised truth table b_j = [a_j > 0] over the 2^n inputs in the full network (E1's binarisation; the Méloux per-unit separation count is also recorded); "gate of neuron i" agreement rate = fraction of representatives in which the bit-vector at raw index i equals the base one, averaged over i. Note: under pure permutation this is a relabelling; identity-tracked it is invariant by construction. The claim "neuron 3 computes AND" is what is audited. |

## Statistics

**CV of a quantity** q over representatives r = 0..R−1 of one network:
for scalars, std/|mean|; for vectors (diagrams are summarised to the scalars
above; spectra and per-neuron vectors stay vectors) the relative dispersion
CV = sqrt(mean_r ‖q_r − q̄‖²) / ‖q̄‖ (reduces to std/|mean| for scalars).
If ‖q̄‖ = 0 and all q_r = 0, CV = 0; if ‖q̄‖ = 0 otherwise, CV = ∞ (reported
as such). Integer-valued quantities (β, argmax, gates) additionally get an
**agreement rate** = fraction of representatives equal to the base value.
Per network the CV is computed on all 100 reps; across networks we report the
median CV per (function, depth, width) cell and pooled, with a 95 % bootstrap
CI (1000 resamples of networks with replacement — networks are independent
seeds, no self-duplication issue arises for a per-network statistic).

**Between-network CV** (the vacuity check): the same dispersion of the base
values across the networks of a cell. An invariance verdict is only
*informative* if the quantity actually varies across networks; a quantity
with within-orbit CV ≈ 0 and between-network CV ≈ 0 is constant, not
invariant, and is flagged as such.

**Scaling behaviour of TC** (uniform-scaling sweep): for each network and
layer ℓ, fit log TC_ℓ = a + k·log α over the 9 α values (H0 and H1, `tt` and
`box`). Verdict per quantity: *scaling-invariant* if the per-unit-scaling CV
is < 1e-6; *scaling-covariant of degree k* if the per-unit CV is ≥ 1e-6 but
the uniform-sweep exponent is the same across all networks (|k − round(k)| <
0.01, fit R² > 0.999, and the same integer for every network); *neither*
otherwise. Also reported: whether the per-unit-scaling CV is non-zero (it
should be — per-unit scalings are not a similarity of the cloud).

**Verdict thresholds (pre-registered):** *exactly invariant*: median CV < 1e-6
(float64 arithmetic; the sanity quantities I1–I3 calibrate this) with the 95 %
CI upper bound < 1e-6; *not invariant*: median CV > 0.05; anything in between
is reported as "approximately invariant" with the number, no verdict claimed.
Agreement-rate verdicts: *stable* ≥ 0.95, *unstable* < 0.5.

## Pre-registered hypotheses

* **H-E2a** (permutation invariance of the candidate list): under `perm`,
  every quantity I1–I8 (and the sorted readings of N1–N4, N6) is exactly
  invariant (median CV < 1e-6) in every cell. Any failure is a bug or a
  finding — investigated before anything else is believed.
* **H-E2b** (TC scaling behaviour): under `scale`, TC (I5), β (I6) and PE (I7)
  are *not* invariant (median CV > 0.05). Under the uniform sweep TC is
  covariant with exponent k = 2 (squared lifetimes of a cloud whose distances
  scale by α), β at median scale and PE are invariant (k = 0). Pre-stated
  reading: "TC is scaling-covariant of degree 2 under uniform scaling and
  neither invariant nor covariant under the per-unit scalings that ReLU
  actually admits" — i.e. TC of a raw activation cloud is not gauge-invariant.
  I8 (standardised cloud) is predicted exactly invariant under `scale`.
* **H-E2c** (spectra): singular values (I4) exactly invariant under `perm`,
  not invariant (CV > 0.05) under `scale`.
* **H-E2d** (sanity): I1, I2, I3 exactly invariant under all three groups.
* **H-E2e** (indictment, raw-index reading): under `combined`, N1–N4 have
  median CV > 0.05; N5 top-neuron raw-index agreement < 0.5; N6 raw-index gate
  agreement < 0.5. Which symmetry breaks each: `perm` breaks every raw-index
  quantity (by relabelling); `scale` breaks N1, N4 (magnitude-based) and N5
  (magnitude-based argmax).
* **H-E2f** (honest prior, stated so it cannot be retrofitted): N2
  (grad × act), N3 (ablation effect) and N6 (gates) are *scaling-invariant per
  unit* — α cancels in grad × act, zero-ablation does not see α, and [a > 0]
  does not see α. So identity-tracked they are invariant under `scale`, and
  their sorted readings are invariant under `combined`. If this holds, the
  indictment of attribution scores is about *labelling*, not values, and the
  table must say so. N1 and N4 are predicted broken under `scale` even
  identity-tracked and even sorted.

Nothing above is a foregone conclusion at the numerical level: the
persistence code path, percentile thresholding, dead units, ties in argmax,
and float64 round-off under α ranging over e^±3 are all places where "exact"
could fail. Report whatever happens.

## Kill attempts and controls

1. **Parameter distance moved.** Per representative, ‖θ_g − θ‖₂ / ‖θ‖₂ over
   the concatenated weights and biases. Report median and min per group. If
   the median is < 0.1 the orbit sample is too tame and the audit is void.
2. **Vacuity.** Between-network CV per quantity (above). A quantity that is
   invariant within orbits but constant across networks gets no "invariant"
   verdict — it gets "constant".
3. **Untrained null.** The full audit on the untrained random networks.
   Invariance is a property of the group action, not of training, so the
   verdicts must be identical; if any differ, the verdict is about the
   trained nets' geometry, not about gauge, and is downgraded.
4. **Planted metrics (unit tests, written with their expected verdict before
   the trained run):** a known-invariant metric (logits) must pass, a
   known-non-invariant metric (W0[0,0]) must fail, in every group.
5. **Broken-group control.** Apply a transformation that is *not* a symmetry
   (permute layer ℓ's units without permuting layer ℓ+1's rows). The audit
   must report the logits as non-invariant. If it does not, the tool cannot
   detect anything.
6. **Hand-built networks.** E1's hand XOR3 (known structure, known dead-free
   layers): I1–I3 CV = 0 exactly; a duplicated-and-rescaled copy of a
   sub-circuit gives a network whose TC is different from the single copy
   (the point cloud gains coordinates) — a check that TC is sensitive to the
   thing it is supposed to see.
7. **Float-precision floor.** The `scale` group's CV for I1 (logits) is
   the float floor of the whole pipeline; every "exact" verdict is judged
   against it, and it is printed in the table.

## Outputs

`results/e2_invariance_table.csv` (one row per quantity × reading × group:
median CV, CI, agreement rate, between-network CV, verdict, breaking symmetry,
n_networks), `results/e2_per_network.csv` (every CV per network), 
`results/e2_cells.csv` (per-cell medians + CIs), `results/e2_scaling.csv`
(fitted exponents per network/layer), `results/e2_untrained_table.csv`,
figures at 300 dpi: `fig_cv_per_quantity.png`, `fig_tc_scale_exponent.png`,
`fig_agreement.png`. `e2lib/` with `check_invariance(metric_fn, model, n_reps,
group)`; `tests/` with the planted-metric and hand-built tests; `FINDINGS.md`
last.

## Compute

CPU only, ≤ 14 worker processes (E1 `_part2` owns the rest). Worst case
≈ 1200 networks × 3 groups × 100 reps × (2 clouds × 2–3 layers) small PH
computations — tens of minutes.

## Amendments

(none yet)

1. (2026-08-30, at aggregation, analysis-only — no definition, threshold or
   hypothesis changed) Two reading rules made explicit. (a) The
   "identity-tracked" reading of the indictment list is realised as the `scale`
   group: undoing the permutation of a `combined` representative leaves exactly a
   scaled-only representative, so a separate identity-tracked column would
   duplicate `scale`. (b) The agreement rate is exact equality, so for
   float-valued quantities it is dominated by 1e-16 round-off and is not
   meaningful; agreement is read only for the integer-valued quantities (β,
   argmax, gates, separation counts), CV for everything else. Also, the
   scaling-verdict label for a fitted exponent of 0 was reworded from
   "covariant (degree 0)" to "invariant to uniform scaling" — same criterion.
