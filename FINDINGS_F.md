# FINDINGS F — the formation vineyard

**In plain English.** On the four models where grokking is a clean event the picture is tight: frequency circles are born in a
burst within one checkpoint of the grokking checkpoint, with the same number of vines and the same lifetimes. The fifth model —
one of the two that had never been looked at — breaks every cross-seed criterion. F fails by the letter, and it fails *because*
an unseen model was included.

**Contradicting the brief's framing, first.** "Birth order has no shared identity" was to be repaired by comparing birth point
processes. They are not comparable across all seeds: decoy_s1 (trained with uninformative hint tokens) groks at checkpoint 17
by test accuracy and its circles *do* rise to full lifetime (≈1.0) right there, like everyone else's — but from checkpoint
~27 on they are repeatedly knocked down (lifetime collapses at checkpoints 28, 34, 46; see `figures/run2_vineyards.png`), so
under the pre-registered onset rule ("certified from here to the end") only 2 vines qualify, at checkpoints 42 and 46. The
representation is disrupted again and again after test accuracy has saturated, and the onset statistic is fragile to it.
Vines are also symmetry-informed
(tracked by frequency label on the b = 0 orbit); nothing here is an unsupervised vineyard.

| criterion | measured | met? |
|---|---|---|
| onset mode within ±2 checkpoints of grokking, every seed | offsets: clean_s0 +1, clean_s1 0, cleanctrl_s2 +1, decoy_s0 +1 (UNSEEN), **decoy_s1 +27 (UNSEEN)** | **NOT MET** |
| W2 between birth processes < 5th pct of null, every pair | 6 of 10 pairs met; all 4 pairs involving decoy_s1 fail (W2 ≈ 25–28 vs null 4) ; the other 6: W2 1.8–3.9 vs null p05 7.9–18 | **NOT MET** |
| vine count spread (max−min)/mean < 0.30 | counts 5, 5, 4, 5, 2 → 0.71 | **NOT MET** |
| median peak lifetime spread < 0.30 | 1.010, 1.052, 1.064, 1.054, 1.019 → 0.051 | met |
| **F overall** | | **FAIL** |

Restricted to the four other models (POST-HOC restriction, not a verdict): mode offsets 0–1, all 6 pairs beat the null by 3–7×,
count spread 0.21, lifetime spread 0.05.

**Claimable.** (i) Peak normalised lifetime of a certified frequency circle is a constant of the architecture (1.01–1.06 on five
models). (ii) On 4 of 5 models circles are born in a burst at grokking. (iii) Test accuracy saturating does not imply the
representation has settled — decoy_s1's circles are repeatedly disrupted after grokking; it was found only because it was held
out. (iv) "First checkpoint from which the circle persists to the end" is a fragile onset rule; first-crossing would be
the obvious alternative and would have to be pre-registered on fresh seeds.

Timings: 5 models × 51 checkpoints in ~3 min. Files: `results/run2/blockF.json`, `figures/run2_vineyards.png`.
