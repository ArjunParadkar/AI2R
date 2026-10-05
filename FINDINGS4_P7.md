# FINDINGS (Run 4) P7 — formation: three signals

**Plain English.** All three signals appear on every fresh generaliser and on no memoriser, and all three registered timing criteria fail by the
letter. The generator does not appear *at* grokking — on fresh seeds it appears **before** it. Neighbouring work: LieGG (Moskalev et al., 2022).

| cell | pre-registered | measured | verdict |
|---|---|---|---|
| P7.1 generator emergence | within ±1 checkpoint of grokking on 4/4; ≥ 95% of later checkpoints; never in a memoriser | offsets −4, −1, −3, 0 (decoys, reported: −1, 0); later presence 100%, 100%, 100%, 93%; memorisers 0 of 36 swept checkpoints | **FAIL** (2/4 within ±1; one seed at 93%) |
| P7.2 symmetry restoration | q(swap) < 0.05 at init, > 0.2 in the memorisation phase, < 0.05 within 2 of grokking, on 4/4; memorisers never below 0.2 | init 0.010–0.019 ✓ 4/4; peak 0.45, —, 0.38, 0.40 (seed 6 has no swept checkpoint with train ≥ 0.99 and test < 0.1); restored ✓ 4/4; memorisers min 0.45, 0.48 ✓ | **FAIL** (3/4: one seed has no registered memorisation phase) |
| P7.3 slot collapse | hint share > 0.3 throughout memorisation; < 0.01 within ±2 of grokking, both decoys | 0.47 → 0.000 on both; below 0.01 at grok+2 (seed 6) and grok+3 (seed 5); the registered phase (test < 0.1) contains 1 and 0 checkpoints because decoys reach 20% test early | **FAIL** |

**Contradicting the brief.** "All appearing at grokking" is not what fresh seeds show. The generator is found when test accuracy is 17%, 70%, 41%
and 93% — four, one, three and zero checkpoints (500 steps each) before the largest jump. It is a *leading* signal on three of four fresh seeds
(post-hoc observation; run 3's seeds gave 0, +1, 0, −1, 0). The hint-slot share does not sit on a plateau and drop: it declines steadily through
the long pre-grokking phase (0.47 → 0.26 over 36 checkpoints on decoy_s5) and then collapses. None of this is visible in train accuracy, which
is 1.00 from checkpoint 1–3 onward.
**Claimable, as measurement:** on six of six fresh generalisers (4 clean, 2 decoy) a generator appears, swap-invariance returns to < 0.005 and
the nuisance slot's share of the geometry goes to < 0.001; on two of two true memorisers none of the three happens in 25,000 steps.
Files: `results/run4/sweep_*.json`, `score.json`, `figures/run4_formation.png`.
