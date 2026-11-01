# Block F — The Spine: gauge orbits and the invariance audit (pre-registration)

_Naming: every run in this document is a horse from the Tracen roster (`experiments/HORSES.md`); the handbook codes below are the pre-registration labels and appear here only as such._

Status: SPEC written 2026-09-07 **before any run**. `results/` is empty at this
commit. Changes after results exist go to `AMENDMENTS.md`, never here.
Novelty status: **OPEN** (handbook §3) — bridge between parameter-symmetry
theory (Grigsby et al. functional dimension; 2025 parameter-symmetry survey)
and Méloux et al. 2025 non-identifiability; the audit is framed as the
mechanistic extension of Implementation Invariance (Sundararajan et al. 2017,
cited, TAKEN at input level). Prior in-house result: E1 (whitebox-v2 programme,
2026-08-30) found orbits/explanations = 1.000 under a *different* explanation
definition (Méloux threshold interpretations, ε-quantised canonical forms).
Block F re-runs the question under the handbook's own definitions and
budget; E1 is cited as the prior, not reused as evidence.

## Question
When a small network admits many valid mechanistic explanations, how much of
that multiplicity is one mechanism seen through the network's gauge group G
(unit permutations × positive ReLU scalings), and how much is genuinely
distinct algorithms? Operationally: N explanations partition into K gauge
orbits; report K/N and how K and N scale with width.

## F1 — Specimens  [LAPTOP]
* ReLU MLPs, hidden depth d ∈ {2, 3}, width w ∈ {4, 6, 8, 10} (same width every
  hidden layer), single linear logit output, f(x) = [logit > 0].
* Targets. 3-input: **XOR3**, **MAJ3**, **AND_OR3** = (x1∧x2)∨x3. The handbook
  lists PARITY3 separately from XOR3; for n = 3 they are the same function, so
  it is run once and AND_OR3 (E1's third function) fills the slot. 4-input: the
  **fixed list of 16 non-degenerate functions** in `flib/functions.py`
  (PARITY4, THR2, THR3, AND4, OR4, XORPAIR_AND, ANDPAIR_XOR, ANDPAIR_OR,
  ORPAIR_AND, MUX_XOR, EQPAIR_AND, ONEHOT, EXACT2, XOR12_AND34, AND123_OR4,
  NANDTREE); non-degeneracy (every input matters) is asserted in code.
* Training: full-batch Adam lr 1e-2, BCE-with-logits on the whole truth table,
  until zero classification error **with margin |logit| ≥ 0.5** (so
  binarisation at 0 is not on a numerical edge) or 20 000 steps. Seeds 0–23
  (24 ≥ 20) per (function, d, w) cell: 19 × 2 × 4 × 24 = 3 648 networks.
  Non-converged seeds are logged in `results/specimens.csv` (`converged=0`)
  and dropped from F2–F6.

## F2 — Enumerate explanations  [LAPTOP / 5070 for the large cells]
* **Circuit** = subset S of hidden units (all layers). Complement ablated in
  every layer at once, two criteria: `zero` (activation → 0) and `mean`
  (activation → its mean over the 2^n inputs in the full network). S is valid
  under a criterion iff the ablated network reproduces the full truth table
  (sign of logit). **Primary count uses minimal valid circuits** (no proper
  valid subset; E1's design decision, same reason: under zero-ablation every
  superset that adds a dead unit is trivially valid); all-valid counts are a
  secondary column.
* **Gate interpretation** (handbook definition). Unit u ∈ S has bit
  b_u(x) = 1[pre-activation_u(x) > 0] in the ablated network. u *implements a
  gate* over its circuit parents (previous circuit layer, or the inputs for
  layer 1) iff b_u is a function of the parents' bits on all 2^n inputs
  (negation allowed: ¬g counts as the same gate, since it is a relabelling of
  the same function). The output implements a gate over the last circuit
  layer's bits iff the target is a function of them. An **interpretation is
  valid iff every unit and the output implement a gate**; the gate table is
  then unique (its restriction to observed parent patterns), so each circuit
  carries 0 or 1 handbook-interpretation. An **explanation** = (circuit,
  valid interpretation). N = number of explanations per (specimen, criterion).
* **Co-primary definition (Méloux et al. 2025, as in E1):** a threshold
  interpretation assigns each circuit unit a threshold between two
  consecutive distinct activation values such that the resulting bit is a
  function of its parents' bits (thresholded likewise); `n_meloux` = exact
  count per circuit; explanations_meloux = Σ n_meloux. Both definitions are
  analysed in full (N, K, K/N, H-F5a/b) and reported side by side. Rationale
  (recorded before the full run, see AMENDMENTS.md #0): the handbook's
  bit-at-0 definition is strictly stronger and can leave a specimen with
  N = 0; a specimen with N = 0 under a definition is excluded from that
  definition's K/N statistics and the count of such specimens is reported.
* Budget. Exhaustive over 2^(d·w) subsets, vectorised; cells with d·w ≤ 20 on
  CPU, d·w ∈ {24, 30} (d3w8, d3w10) on the 5070 in chunks of 2^20 masks. A
  per-network wall cap of 30 min; networks over the cap are logged
  `enumerated=0` and excluded (count reported). Nothing is sampled.
* Output `results/explanations.csv`: specimen, criterion, circuit_id (bitmask),
  n_units, interpretable (0/1), n_meloux. Per specimen: |mean ∩ zero| / |mean ∪ zero|
  on minimal circuits (`results/criterion_agreement.csv`).
* **Kill F2.** Hand-built XOR3 network with exactly one circuit
  (3 hidden units: x1∧¬x2-type gates, weights in `flib/planted.py`): enumeration
  must return exactly that circuit under both criteria, with 1 interpretation.

## F3 — Gauge group per specimen  [LAPTOP]
`symmetry.describe` (known continuous dim = Σ widths) and `symmetry.hidden`
(fiber dim = corank of the realisation Jacobian on the truth table + 4×n_params
random probes in [-1, 2]^n, rel. tol 1e-8). Record known_dim, fiber_dim,
excess, and the identification (dead / duplicate / constant / unclassified).
`G_incomplete = excess > 0`. Output `results/symmetry.csv`.

## F4 — Orbit partition (H0 of the quotient)  [LAPTOP]
**Definition (exact).** Explanations E = (S, W|_S) and E' = (S', W'|_S') of
specimens in the same cell are gauge-equivalent iff some g ∈ G maps one onto
the other: a layer-wise bijection S → S' (permutation part) and positive
per-unit factors (scaling part) such that the sub-implementation weights
agree within relative tolerance 1e-6 after rescaling, and the gate tables map
unit-to-unit under the bijection. Implementation: canonical form of the
sub-implementation (scale-normalise each unit's incoming (w, b) to unit norm,
push the norm downstream; lexicographic sort of units by the quantised
incoming vector; output row normalised) hashed at quantisation 1e-6 relative;
equal hash ⇒ same orbit; union-find over pairs is then trivial. Ties at the
quantisation boundary are checked by a 1e-4-relative re-hash (count of
changed orbit assignments reported).
* Orbits are computed **within each specimen** (K_within) and **pooled across
  the cell's seeds** (K_pooled); both reported. The pooled ratio is the primary
  statistic (within a single network, G can only merge duplicate circuits).
* Output `results/orbits.csv`: specimen, criterion, N, K_within; and
  `results/orbits_pooled.csv`: cell, criterion, N, K_pooled, K/N.
* **Kill F4.** Planted networks in `flib/planted.py` with exactly 1, 2, 3
  genuinely distinct algorithms wired in parallel and selected by a fixed
  input (XOR3 via (x1∧¬x2)∨(¬x1∧x2); via (x1∨x2)∧¬(x1∧x2); via a 3-unit
  threshold decomposition), each algorithm duplicated once with a permuted,
  rescaled copy: enumeration + partition must yield K = 1, 2, 3 respectively
  with N = 2K.

## F5 — Hypotheses  [LAPTOP]
* **H-F5a**: median over specimens (pooled-cell ratio assigned to each
  specimen) of K/N ≤ 0.5. Pass/fail per criterion.
* **H-F5b**: regress log K_pooled and log N_pooled on width (per function,
  depth): slope(log N) − slope(log K) > 0 with a seed-bootstrap (1000
  resamples, seeds resampled *without* duplication, E1 amendment 5) 95 % CI
  excluding 0.
* Figures `fig/F5_KN_vs_width.png` (log-log, per function), `fig/F5_KN_hist.png`.
* **Kill F5.** Both criteria must give the same qualitative verdict; a
  disagreement is the headline.

## F6 — Orbit autopsy  [LAPTOP]
For every specimen with K_within > 1 (and for every pooled cell with
K_pooled < N): one representative per orbit, gate tables rendered as named
gates where a name exists (AND/OR/XOR/NAND/NOR/XNOR/MAJ/THR/projection/
negation), algorithm named by its gate composition string. Table
`results/F6_algorithms.csv`: function, n distinct algorithms, list. If a
"distinct" orbit pair is related by a symmetry not in G (e.g. a sign-flip
through a hidden-unit that is dead on the truth table), it is added to S1 and
F3–F5 rerun, logged in `AMENDMENTS.md`.

## F7 — Transformer extension  [5070]
1- and 2-layer `ToyTransformer` (d = 32, 4 heads, d_head 8, ReLU MLP) on the
same Boolean tasks as token sequences (tokens: x1..xn as 0/1 tokens plus an
`=` token; readout at the last position; 2 classes). To keep exhaustive
enumeration feasible the MLP width is d_mlp = 16 (1 layer: 20 nodes) and 8
(2 layers: 24 nodes). Circuits = subsets of {heads} ∪ {MLP units}; complement
mean-ablated (head output → its mean over inputs; MLP unit → its mean). No
gate interpretations for heads (they are not Boolean); explanations = valid
minimal circuits. G extended with head permutations and within-head GL(d_head)
changes of basis (S1); canonical form of a head = the rotation-invariant
bilinear forms (W_Q W_K^T, W_V W_O) restricted to the circuit, quantised.
Seeds 0–9 per (function ∈ {XOR3, MAJ3, PARITY4, THR2}, layers ∈ {1, 2}).
Output `results/orbits_transformer.csv`, same H-F5a test.

## F8 — Renaming samples  [LAPTOP]
Per F1 specimen: 100 permutation-only, 100 scaling-only (per-unit log-uniform
α ∈ [0.1, 10]), 100 combined; function identity verified on the truth table
(max |Δlogit| < 1e-6) for every sample; `results/F8_identity.csv`.

## F9 — Anchor  [LAPTOP]
Input→output weight product (Π W_l): per-component CV under all three
samples < 1e-6 on every specimen, else the pipeline stops.

## F10 — The audit table  [LAPTOP]
`check_invariance` (S2: invariant CV < 0.01; covariant = perm-invariant with a
power law in α, R² ≥ 0.99; artifact otherwise) on each candidate, per
specimen, on the truth-table activation clouds (and on a 512-point random
probe cloud in [-1,2]^n so diagrams are not degenerate — both reported):
* expected invariant: persistence summaries (β0, β1 at median scale, total
  persistence H0/H1, persistent entropy) of hidden-layer activation clouds
  under **permutation**; weight products; invariant polynomials (tr(W_lᵀW_l),
  det Gram of each layer map, tr of round-trip products W_l W_{l+1}···);
  loop products along closed paths (unit i → j → i through two layers, summed);
* expected covariant: singular values of layer maps under scaling; persistence
  summaries under scaling (**measured**, not assumed — Z5 fits the law);
* expected artifact: per-unit activations, per-unit gradient×input
  attributions, top-k (k = 3) unit set by |attribution| (encoded as a 0/1
  membership vector), raw per-unit weight norms, single-direction "features"
  (top principal direction of each hidden layer's cloud, expressed in unit
  coordinates);
* with `align=True`: the single-direction features after Procrustes on the
  unit axis, and per-unit quantities after Hungarian matching — reported as
  a second verdict column.
Output `results/audit_table.csv` (one row per candidate × specimen; plus a
summary row per candidate with the majority verdict and the fraction of
specimens agreeing), `fig/F10_audit_heatmap.png`.
* **Kill F10.** Any candidate whose class differs between permutation-only and
  combined samples is reported with both classes.

## F11 — Real-model audit  [5070]
GPT-2 small (HF `gpt2`, fp32, eval). Renamings (S1 `gpt2` backend): head
permutations, within-head GL(d_head) changes of basis, MLP unit permutations,
LayerNorm-compatible rescalings (ln_1 → c_attn rows, ln_2 → c_fc rows). Exact
scalings only; GELU is not homogeneous so no MLP scalings; ln_f is tied to the
embedding so no ln_f scaling. Documented in the backend docstring and repeated
in FINDINGS. Layer-6 residual-stream and MLP-hidden activations over 2 000
tokens of OpenWebText-10k; F10's candidate list (persistence summaries of
PCA-20 clouds, singular values of layer-6 W_in/W_out, per-unit activations,
per-unit gradient×input for the next-token loss, top-k unit sets, weight
norms, top principal direction, and the gauge-invariant per-head bilinear
forms); n = 30 renamings per kind (time-bounded). Function identity verified
(max |Δlogit| < 1e-3 in fp32 on 2 000 tokens; the fp32 floor is reported).
* **Kill F11.** Any classification that flips between MLP-toy and GPT-2 is
  flagged in the table, not smoothed.

## F12 — Frontier audit  [HPCC]
Script `hpcc/kitasan_black.py`: Llama-3-8B or Gemma-2-9B, 3 layers,
checkpointed per (layer, kind, sample), `--resume`. Not run locally.

## F13 — Ship  [LAPTOP]
Public API (`whiteboxtopo.symmetry`, `whiteboxtopo.invariants.check_invariance`)
documented, VALIDATION.md rows, NOVELTY.md with the two searches
("parameter symmetry explanation identifiability", "implementation invariance
mechanistic interpretability audit"), dates, top hits.

## Pass/fail summary (pre-registered)
| Item | Threshold |
|---|---|
| F2 kill | planted XOR net → exactly 1 circuit, 1 interpretation, both criteria |
| F4 kill | planted 1/2/3-algorithm nets → K = 1/2/3, N = 2K |
| H-F5a | median K/N ≤ 0.5 (per criterion) |
| H-F5b | slope(log N) − slope(log K) > 0, bootstrap CI excludes 0 |
| F5 kill | criteria agree qualitatively |
| F9 | weight-product CV < 1e-6 on every specimen |
| F10 | expected classes as listed; deviations reported with both classes |
| F11 | flips vs toy flagged |
