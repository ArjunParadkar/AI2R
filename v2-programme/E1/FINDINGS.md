# E1 — Gauge orbits vs circuit non-identifiability: FINDINGS

**Status: PRELIMINARY (2026-08-30).** 27 of 36 cells complete (all of XOR3,
MAJ3, AND_OR3, PARITY4; THR2_4 d2 w4–8). The first run was killed by a machine
reboot at 10:01 mid-cell; the remaining 9 cells (THR2_4 d2w10/d3w4/d3w6, RAND4 ×6)
are re-running (`run_e1.py --suffix _part2`, log `results/run_part2.log`) and
`aggregate_e1.py` merges the parts. Nothing below is expected to change: every
completed cell gives the same answer. Re-run `aggregate_e1.py` when the log says
`done` and update the cell counts here.

## Verdict

| Hypothesis | Pre-registered criterion | Result |
|---|---|---|
| **H-E1a** gauge explains ≥ 50 % of multiplicity | pooled orbits/explanations ≤ 0.5, CI < 0.5 | **FAILS in 56/56 (cell, scheme) rows.** Ratio ∈ [0.997, 1.000]; bootstrap CI lower bound ≥ 0.991. |
| **H-E1b** orbits grow slower with width than explanations | slope(log orbits) < slope(log expl), CI < 0 | **FAILS 5/5** (function, depth) pairs. Slope difference is exactly 0 (orbits = explanations at every width). |

**Explanation multiplicity in these networks is not gauge.** Under the known
symmetry group G (hidden-unit permutations × positive ReLU scalings) essentially
no two valid explanations — within a network or across 24 independently trained
seeds — lie on the same orbit, at any tolerance ε ∈ {0.01 … 0.2}. Even at ε = 0.2
the pooled ratio stays ≥ 0.83 in every cell (`results/e1_pooled.csv`,
`orbits_eps0.2/n_expl`), and at the circuit level (ignoring interpretations)
≥ 0.98 (`circuit_orbits_eps0.05/n_circuits`).

Why this is the answer and not a bug:
* Within a network, G cannot act on interpretation multiplicity at all
  (amendment 2/3): the gate-assignment tree of a circuit is gauge-invariant, so
  the within-network ratio is ≈ 1 by construction; G could only merge duplicate
  circuits, and there are none (minimal circuits are distinct neuron subsets).
* Across seeds, gauge equivalence would require two independent trainings to land
  on the same canonical parameter form up to ε. They never do: the trained
  weights are continuous and independent; coincidence to ε = 0.05 of the
  normalised sub-implementation is a measure-zero event that ε-quantisation does
  not rescue.
* The orbit code is verified on hand-built networks with planted permutations
  and scalings (8/8 control tests green, `tests/test_e1.py`), and a 1e-4
  relative-jitter re-count changes the orbit count in only 41/1056 rows
  (ε-grid boundary effects, all by < 1 %).

So the Méloux-type multiplicity is *genuine*: distinct neuron subsets carrying
distinct gate assignments, not one mechanism seen through symmetry.

## A coarser, exactly gauge-invariant key does not rescue it

`functional_key` = per layer, the sorted multiset of binarised truth tables of
the circuit's units, plus the output bits. This is exactly G-invariant and much
coarser than the parameter form (it identifies two circuits whenever their units
compute the same Boolean functions). Pooled orbits/explanations under it is
0.65–1.00 (`pooled_ratio_functional`, right panel of `fig_ratio_vs_width.png`);
at the circuit level 0.67–1.00. I.e. even "same gates, any wiring" collapses at
most ~⅓ of the multiplicity, and mostly < 15 %.

## Scale of the multiplicity (Méloux replication)

Pooled explanation counts per cell run from 10² (w=4) to 10¹³ (PARITY4 d2 w10),
with the median number of interpretations per minimal circuit reaching 10⁷.
Multiplicity is dominated by *interpretations* (gate assignments), not circuits:
pooled minimal-circuit counts are 9–1975 per cell. Exact counting (amendment 4)
was needed to get these numbers at all. Zero- and mean-ablation validity
disagree substantially (e.g. AND_OR3 d2 w8: 202 mean-valid vs 425 zero-valid
pooled circuits; explanation counts differ by 10³×) — recorded, as the brief asked.

## Controls and side results

* **Untrained control** (SPEC 5): random networks of the same shape never
  compute the function as a whole (0/1056), but for the *monotone* 3-input
  functions wide random nets do contain zero-ablation-valid sub-circuits
  (AND_OR3 d2 w10: mean 722, max 6304 per net; MAJ3 d2 w10: mean 385) — one to
  two orders of magnitude below trained nets (medians 1.8×10⁵) but not zero.
  For XOR3/PARITY4/THR2_4 the random count is essentially 0. Caveat for the
  validity criterion: on an 8-row truth table a random 10-wide layer will
  sometimes contain a monotone gate by chance.
* **Functional dimension** (Grigsby): Jacobian rank of the realisation map on
  uniform inputs in [-1, 2]ⁿ. 92 % of trained networks have a rank deficit
  beyond scaling symmetries + dead units (median 2–15 directions, growing with
  width; `fig_funcdim.png`). Hidden symmetries / flat directions are the norm,
  not the exception — reported, not characterised, as the brief specified. They
  were not used in the orbit partition; adding them could only merge orbits in
  the continuous-parameter sense, which the ε-tolerance already probes.
* **Convergence**: XOR3 70 %, PARITY4 57 % of seeds reach zero error (w=4 cells
  have only 7–9 converged seeds and PARITY4 w=4 has *no* interpretable
  circuits — 100 % uninterpretable rate); monotone functions 94–100 %.
* **Bootstrap defect found and fixed** (amendment 5): a with-replacement seed
  resample duplicates networks, which are gauge-identical to themselves, and
  produced CI lower bounds ≈ 0.35 — the *only* place in E1 where "gauge collapse"
  ever appeared, and it was an artifact of the resampling. Unique-seed subsampling
  bootstrap gives CIs of [0.99, 1.00].

## What this means for the programme

* The E1 bridge "parameter symmetry ⇒ explanation multiplicity" is refuted for
  this setting; the honest headline is *non-identifiability is not a gauge
  artifact*. That is a clean, citeable negative and sharpens Méloux et al.:
  the incompatible explanations are not symmetry-related copies.
* The positive residue: ε-orbit counting and the functional key are working
  gauge-invariant *measurements*, which is exactly what E2 needs. E2 can proceed
  on the saved networks (`results/networks/*.npz`).
* Not done / needs Arjun: figure "two gauge-equivalent circuits" cannot be
  drawn from data (none exist across seeds); it can only be a planted example.
  Figure "two genuinely distinct circuits" — every cell provides thousands;
  not yet rendered.

Files: `results/e1_cells.csv` (per-cell verdicts + CIs), `e1_h1b.csv`,
`e1_pooled*.csv`, `e1_per_network*.csv`, `e1_funcdim*.csv`,
`fig_orbits_vs_expl.png`, `fig_ratio_vs_width.png`, `fig_funcdim.png`.

## Example figures (added 2026-08-30)

`make_examples.py` → `fig_example_gauge_pair.png` (a *planted* gauge-equivalent
pair: circuit 0x317 of XOR3 d2 w6 seed 1 and its image under a random
permutation + scaling; canonical forms coincide, so the orbit code merges them —
the only way to exhibit such a pair, since none occur between independently
trained networks) and `fig_example_distinct_pair.png` (two genuinely distinct
minimal mean-valid circuits of the same network: different neuron subsets,
different gate assignments, no element of G maps one to the other).
