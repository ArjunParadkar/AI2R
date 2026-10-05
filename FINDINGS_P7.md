# FINDINGS P7 — formation of the group

**Plain English.** By the letter nothing forms: the max-norm automorphism group never reaches order 113 on any checkpoint of any clean Z113
run, and never reaches a cyclic element of order 36 on any ring checkpoint. P7.1 and P7.3 are NOT MET. Under the post-hoc full-cycle rule
(A-10, fixed at 14:17 — before the fresh seeds had finished training) the picture is sharp: a 113-cycle generator first appears within one
checkpoint of the grokking checkpoint on five of five generalisers, persists afterwards, and never appears in a memoriser.
Neighbouring work: LieGG (Moskalev et al., NeurIPS 2022).

**Contradicting the brief.** P7.2's prediction — swap-invariance precedes grokking — is wrong. q(swap) at resid_mid starts *low* at
initialisation (0.012–0.019, architectural), rises to 0.20–0.46 while the network memorises, and falls back below 0.05 at the grokking
checkpoint or one to two checkpoints after it (offsets +1, +2, 0, 0, +1). Commutativity is restored with generalisation, not before it.

| cell | pre-registered | measured | verdict |
|---|---|---|---|
| P7.1 ring | cyclic order ≥ 36 within 2 checkpoints of error < 5° on 3/3 | error < 5° at checkpoints 5, 8, 9; letter order 1 at every checkpoint | **NOT MET** |
| P7.2 swap trajectory | report | first q < 0.05 (after step 0) minus grok checkpoint: clean_s0 +1, clean_s1 +2, decoy_s1 0, fresh s3 0, fresh s4 +1; memorisers never (max q 0.56–0.75) | — (prediction wrong) |
| P7.3 decoy_s1 | letter order ≥ 113 at every checkpoint ≥ grok+2 | 16% of those checkpoints (max order 173) | **NOT MET** |

POST-HOC (full 113-cycle among mean-accepted candidates):

| model | grok ckpt | first checkpoint with a 113-cycle | offset | share of checkpoints ≥ grok+2 with a 113-cycle |
|---|---|---|---|---|
| clean_s0 | 16 | 16 | 0 | 97% |
| clean_s1 | 10 | 11 | +1 | 100% |
| decoy_s1 (the model whose circles flicker, run 2) | 17 | 17 | 0 | **100%** |
| fresh clean_s3 (sparse sweep, A-11) | 6 | 5 | −1 | 100% |
| fresh clean_s4 (sparse sweep) | 7 | 7 | 0 | 100% |
| scrambled_s0 (all 51 checkpoints), fresh scrambled_s3, s4 | — | never (0 of 73 checkpoints) | — | — |

On decoy_s1 the run-2 vineyard showed frequency circles knocked down at checkpoints 28, 34 and 46; the post-hoc generator is present at
every one of them. "The symmetry is the invariant, the circle is its shadow" is supported — post-hoc, on one model. Ring, post-hoc: the
relative rule over-counts (300+ candidates vs the true dihedral 144) and already fires at checkpoint 0–1; it is not informative there.

**Claimable.** Only the P7.2 trajectory (a measurement, no criterion) and the memoriser controls. The post-hoc grokking coincidence is the
strongest regularity in the run and must be pre-registered before it is claimed.
Files: `results/run3/sweep_*.json`, `sweep_scores.json`, `blockRing.json`, `figures/run3_group_formation.png`.
