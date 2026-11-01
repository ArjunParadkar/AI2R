| Cell | Criterion | Result | Verdict |
|---|---|---|---|
| P1a V1-onehot | node precision & recall >= 0.9 on 5/5 | P 1.00-1.00, R 1.00-1.00 | PASS |
| P1b V1-onehot | edge F1 >= 0.9 on 5/5 | 1.00-1.00 | PASS |
| P1c V1-onehot | decoy represented-not-used 5/5 | 5/5 | PASS |
| P1e V1-onehot | generality MAE vs oracle <= 0.05 on 5/5 | 0.009 (max) | PASS |
| P1a V1-noise05 | node precision & recall >= 0.9 on 5/5 | P 1.00-1.00, R 1.00-1.00 | PASS |
| P1b V1-noise05 | edge F1 >= 0.9 on 5/5 | 1.00-1.00 | PASS |
| P1c V1-noise05 | decoy represented-not-used 5/5 | 5/5 | PASS |
| P1e V1-noise05 | generality MAE vs oracle <= 0.05 on 5/5 | 0.009 (max) | PASS |
| P1d V1 | wrong hypotheses accepted <= 10% (pooled) | 0/50 | PASS |
| P1f NULL | FP <= 0.02 every seed; mean accepted <= 1 | max FP 0.025, mean accepted 0.80 (20 cells) | FAIL |
| P2 V2 | AUROC(-mu) >= 0.9 on 5/5 and lower CI > 0.8 on 4/5 | pt 0.500-0.500, lo 0.500-0.500 | FAIL |
| P3a V3-comp | move accuracy >= 0.9 (withheld and seen) 5/5 | 1.00 | PASS |
| P3b V3-comp | predicted-correctness accuracy >= 0.9 5/5 | 1.00 | PASS |
| P3a V3-lookup | move accuracy >= 0.9 (withheld and seen) 5/5 | 1.00 | PASS |
| P3b V3-lookup | predicted-correctness accuracy >= 0.9 5/5 | 1.00 | PASS |
| P4 V4 | AUROC >= 0.95 and scaffold recall >= 0.9, 5/5 | AUROC 1.000, recall 1.00 | PASS |
| P5a V5-valid | exactly one valid node N <- {A,B}, 4/5 | 5/5 | PASS |
| P5b V5-decoy | zero valid nodes, 4/5 | 5/5 | PASS |
| P5b V5-leaf | zero valid nodes and N 'new leaf (used, not derived)', 4/5 | 5/5 | PASS |
| P5b V5-relabel | zero valid nodes, 4/5 | 5/5 | PASS |
| P5b V5-generic | zero valid nodes and N 'exposure-generic', 4/5 | 5/5 | PASS |
