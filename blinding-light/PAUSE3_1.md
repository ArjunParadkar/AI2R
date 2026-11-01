# PAUSE 1 (Run 3) — did the group get discovered?  · 2026-09-20 14:35

**Said first, as the brief requires: by the letter, the group is NOT discoverable on Z113 without being supplied.** With the registered
max-norm tolerance (10% of the median distance) the isometry search returns groups of order 1, 1, 2, 13 and 5 on the five grokked
models; the criterion was order ≥ 113 with a 113-cycle on ≥ 4 of 5. The ring RNN returns order 1 on 3 of 3 seeds.

**Why (diagnostic, post-hoc).** A true translation of the residues changes the output-pullback distance matrix by only 3% of the median
*on average* on a grokked model (memoriser: 22%; random relabelling: 21–33%) but by 14–42% in its worst entry. The function is
translation-equivariant in its predictions exactly (every orbit is prediction-constant, P2.1 = 1.000 on 5/5) and in its logit geometry
only approximately. Max-norm isometry is the wrong notion of "symmetry of a trained network"; the network's symmetry is statistical.

**Contradicting the brief's framing.**
1. Stated in SPEC3 before computing: under the output-pullback metric this "group" is a property of the *function* (any model that is
   right everywhere has D_ij = f(i−j)); it is not evidence about the representation. On the raw embedding rows (Euclidean, secondary)
   nothing is found on any model (orders 1, 1, 1, 1, 3): **the token embedding is not an isometric copy of Z_113** — it is a sum of a few
   characters with unequal, non-translation-invariant weights plus residue.
2. **The Z₂ swap quotient is not simply "learned".** It is present at initialisation (q = 0.012 at step 0: attention pooling at the read-off
   position is permutation-invariant up to small positional embeddings), *destroyed* by memorisation (memoriser: q = 0.58; clean seed 1
   at step 3000, mid-memorisation: 0.29) and *restored* at generalisation (0.002–0.005 on 5/5). Run 2 called it a learned invariance; the
   accurate statement is that generalisers keep (or return to) a symmetry the architecture offers and memorisers break it.
3. My own post-hoc replacement statistics were wrong twice (A-9, A-10): a mean-norm tolerance passes 804 permutations on an untrained
   model, and a chance-relative one passes 3,406 on the memoriser, because a sparse permutation (one transposition) barely moves a mean.
   Neither yields a group order. The only post-hoc object that carries information is whether a **full 113-cycle** is among the candidates.

| cell | pre-registered | measured | met? |
|---|---|---|---|
| P1.1 group | order ≥ 113 and a 113-cycle on ≥ 4/5 (D^g, max-norm tol 0.10) | orders 1, 1, 2, 13, 5; a 113-cycle on 2/5 (the two decoys) | **NOT MET** |
| P1.1 frequencies | ≥ 4 of 5 at r > 0.9 under one unit, from the discovered generator, on ≥ 4/5 | letter generator exists on 2 models: decoy_s0 4 at r>0.9, decoy_s1 3 | **NOT MET** (1/5) |
| P1.1, Euclidean on embedding rows (secondary) | reported | orders 1, 1, 1, 1, 3 | — |
| P1.2 memoriser | order ≤ 2 | 1 | met |
| P1.3 ring | cyclic order ≥ 36 on 3/3; shuffled null ≤ 2 | order 1, 1, 1 (null 1, 1, 1) | **NOT MET** |
| P1.5 swap quotient | q(swap) < 0.05 at resid_mid on 5/5 (RE-SCORE) | 0.0050, 0.0019, 0.0024, 0.0040, 0.0045; embed 0.61–0.78; memoriser 0.58 | met |
| P2.1 orbit explanation | ≥ 99% of inputs orbit- and coset-constant on 5/5; memoriser < 60% | 1.000 ×5; memoriser 0.000 | met |
| P1.4 rot-MNIST Killing | — | agent IMG, pending | — |

**POST-HOC (A-10), cannot upgrade a verdict.** Accepting candidates whose *mean* defect is below a quarter of a random relabelling's:
a full 113-cycle generator is found on **5/5 grokked models and not on the memoriser**. Ordering the tokens along that discovered
generator and taking the DFT of the embedding recovers 5, 4, 4, 4, 4 of each model's five inventory frequencies, at r > 0.9 for 5, 4, 4, 4, 3
of them (r = 0.990–0.999; one at 0.35), under a single unit u that is unique (1 of 56 units reaches ≥ 4 matches on every model) and equals
the unit implied by the generator itself (u = 1, 1, 1, 43, 42 — the decoy models' generators are translations by 43⁻¹ and 42⁻¹, not by 1, and
the frequencies only line up after that change of generator). That is the unsupervised naming run 2 failed, by the group route — post-hoc,
with an acceptance rule invented after seeing the letter fail, on the same models.

**Kill tests.** Planted truth before networks: regular 12-, 31-, 113-gons → dihedral group of order 2p; a noisy three-frequency Z113 embedding
presented in scrambled order → order 226, generator found, planted frequencies recovered up to one unit; random cloud → trivial; simplex →
flagged degenerate. Two instrument faults were found there and fixed before any network was touched (A-8).

**Claimable.** (i) The orbit explanation is exact: every input's translation orbit under the discovered generator, and its swap coset,
receive one prediction on all five generalisers and on none of the memoriser's inputs. (ii) Invariance depth of the swap is resid_mid on
5/5 (q 0.6–0.8 at the embedding → 0.002–0.005). (iii) NOT claimable by the letter: a discovered group. Claimable only as a post-hoc
observation to be pre-registered on fresh seeds: a 113-cycle generator and its characters are recoverable from outputs alone.
Neighbouring work: LieGG (Moskalev et al., NeurIPS 2022) extracts Lie generators from trained networks; nothing here is claimed as first
beyond the finite, metric formulation.

Timings: group search 15–130 s per model on CPU under load (≈450 anchor triples, two refinements each); swap quotient 10 s.
Files: `results/run3/blockZ_finals.json` (+ `_letter_only.json`), `blockRing.json`, `whiteboxtopo/symmetry_finite.py`, `tests/test_run3_finite.py`.
