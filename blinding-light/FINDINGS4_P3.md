# FINDINGS (Run 4) P3 — the unsupervised lead, with a null that can fail; the interrupt

**Plain English.** The dominant-direction rule passes exactly as registered, *including the null that was designed to make it fire wrongly*.
Flag any input slot holding more than 80% of trace(ḡ): it fires on both fresh planted-shortcut models (hint slot 0.957, 0.951), stays silent on
the four fresh clean models (max slot 0.49–0.50) and on both grokked decoys (hint 0.0002, 0.0003) — and fires on the a-only task (slot a 0.997),
where one slot is legitimately dominant because b is irrelevant by design. So the rule is unsupervised and works, and what it detects is
**dominance, not spuriousness**: whether a dominant input is a shortcut or the task is a question about the task, which the rule cannot answer.
Neighbouring work: LieGG (Moskalev et al., NeurIPS 2022).

| cell | pre-registered | measured | verdict |
|---|---|---|---|
| P3.1 fires on planted shortcut ×2 (fresh) | hint share > 0.8 | 0.957, 0.951 | met |
| P3.1 silent on clean ×4 (fresh) | max slot < 0.6 | 0.497, 0.488, 0.500, 0.489 | met |
| P3.1 silent on grokked decoys ×2 (fresh) | hint < 0.01 | 0.00018, 0.00033 | met |
| P3.1 fires on the legitimately dominant task (a-only) — registered limitation | slot a > 0.8 | 0.997 | met |
| **P3.1** | | | **PASS** |
| P3.2 interrupt | 5/5 > 0.98 at step 6000; floor silent to 12000; test ≥ 0.99 | decorrelated accuracy at step 6000: 0.999 / 1.000 / 1.000 / 1.000 / 1.000; floor re-firings in (6000, 12000]: 0; in-distribution test at 12000: 0.999 / 1.000 / 1.000 / 1.000 / 1.000 | PASS |

Also measured and not in the criterion: true memorisers are *not* flagged (max slot 0.38, 0.39) — they spread across all three slots, which is
P6.2. An untrained model gives each slot a third. So the slot profile of ḡ has four readable regimes: thirds (untrained), spread (memoriser),
two task slots (generaliser), one slot (shortcut — or a task that really has one input).

**P3.3 — "symmetry breaking = spurious structure" is dropped.** Reason, with the run-3 numbers replicated here on fresh seeds: on decoy models
the hint carries nothing, yet its share of the geometry is 0.33 at initialisation, rises to 0.44–0.47, stays above 0.25 for the entire
pre-generalisation phase (36 checkpoints on decoy_s5) and only then collapses to 0.000. A memorising network breaks every symmetry, the
nuisance's included. The monitor of record for a *known* nuisance remains the run-2 functional floor.
Files: `results/run4/z113_finals.json`, `score.json`, `figures/run4_dominance.png`, `runs/r4_interrupt_s*/history.json`.

**P3.2 note (2026-09-20 23:49).** PASS by the letter on 5/5 seeds. Not covered by the criterion and recorded for the record: seed 0's decorrelated accuracy dipped to 0.943 at step 7500 (floor silent, direct NMI 0.406; reliance −0.001) and recovered to 0.999 by step 8000; all other checks in (6000, 12000] read 0.997–1.000. One run per seed. The interrupt needs z (the nuisance must be known) and is not topological. LieGG (Moskalev et al., NeurIPS 2022) remains the cited neighbour for the symmetry instruments.
