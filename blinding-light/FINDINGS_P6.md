# FINDINGS P6 — memorisation vs understanding: is it the size of the group?

**Plain English.** By the letter, no. No registered measure of "group size" separates generaliser from memoriser: the continuous Killing
dimension is 0 for clean, scrambled and untrained ResNet-18 alike; the null fraction is 0.997 for all three (as SPEC3 said it must be);
the finite automorphism order never reaches 113 on any clean Z113 checkpoint. What *does* replicate — on fresh seeds trained for this
run and never seen before — is the generaliser half of run 2's two after-the-fact observations.
Neighbouring work: LieGG (Moskalev, Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022); details of the image half in `FINDINGS_IMG.md`.

| cell | pre-registered | measured | verdict |
|---|---|---|---|
| P6.1 Killing dimension | dim 𝔨_clean ≥ dim 𝔨_scr + 2 (input pullback, 8 generators, 500 images) | 0 vs 0 (untrained null: 0) | **NOT MET** |
| P6.2 invariance fraction | ν̄_clean > ν̄_scr + 0.1 | 0.997070 vs 0.997062 (gap 8.5e-6; unreachable by construction) | **NOT MET** |
| P6.2 secondary: participation ratio of g's spectrum | reported | clean 5.7, scrambled 7.8, untrained 9.5 | — |
| P6.3 automorphism order across training | ≥ 113 within ±2 checkpoints of grokking on both clean seeds, all earlier ≤ 2; memoriser ≤ 2 throughout | never ≥ 113 with a cycle (max 13 and 81); memoriser ≤ 1 on all 51 checkpoints | **NOT MET** (memoriser half met) |
| P6.3 UNSEEN fresh seeds (sparse sweep, A-11) | same | clean_s3, s4: never ≥ 113 with a cycle (max 103, 9); scrambled_s3, s4: 1 throughout | **NOT MET** |
| P6.4 connection-scale slope, FRESH seeds | < −0.3 on 4 generalisers, > −0.1 on 4 memorisers | generalisers −0.57, −0.53 (Z113), −0.50, −0.49 (ResNet): **4/4**; memorisers +0.01, −0.01 (Z113): 2/2, −0.17, −0.37 (ResNet): 0/2 | **NOT MET** (6 of 8) |
| P6.4 ID-profile class, FRESH seeds, ≥ 18/20 bootstraps | plateau (grokked Z113), growth (Z113 memoriser), decay (clean ResNet penultimate) | plateau 20/20 ×2 ✓; memorisers classified *plateau* 20/20 ×2 ✗ (overall slope +0.09, below the +0.2 growth threshold); decay 20/20 ×2 ✓ | **NOT MET** (4 of 6) |

**Contradicting the brief.**
1. "A memoriser has a trivial symmetry group, a generaliser a nontrivial one" cannot be read off a max-norm automorphism group or a Killing
   algebra: both are trivial for everyone. The registered definitions of "size" do not see the difference.
2. The fresh ResNet "memorisers" are not memorisers: 15 epochs on random labels reached 56–59% training accuracy (run 1's 20 epochs: 87%).
   Their connection-scale profile is non-monotone with an outlier-driven spike at layer3 (2.87), and its slope (−0.17, −0.37) says little.
3. What clean training does to the continuous side is photometric before it is geometric: every Killing residual is lower in the clean net
   than in the scrambled or untrained one (ratio ≈ 0.7 for the six warps), but the largest drops are brightness (0.09 vs 0.25) and contrast
   (0.10 vs 0.20). The memoriser's residuals sit at the untrained level.

**What replicated on unseen models (generaliser half only).** Connection scale of class fibers, in units of the layer's median distance,
falls with depth in every fresh generaliser — Z113 0.77 → 0.65 → 0.20 and 0.76 → 0.61 → 0.23; ResNet 0.98 → 0.48 and 0.99 → 0.49 — and is
flat in the fresh Z113 memorisers (0.92 → 0.92, 0.92 → 0.91). ID(k) plateaus on both fresh grokked Z113 models (6.4–7.7) and decays on both
fresh clean ResNet penultimate layers (16 → 3), 20/20 bootstraps each.

**POST-HOC (A-10).** A full 113-cycle generator appears within one checkpoint of grokking on 5/5 generalisers including both fresh seeds
(offsets 0, +1, 0, −1, 0), persists in 97–100% of later checkpoints, and appears in 0 of 73 memoriser checkpoints (FINDINGS_P7.md). By the
fact stated in SPEC3 before computing, this is a label-free statistic of the *function* on held-out inputs and is expected to coincide with
test accuracy; it does not show that "generalisation is the group".

**Claimable.** The generaliser-side depth trend and ID profile, replicated on four and four fresh models. Not claimable: any definition of
generalisation as group size.
Files: `results/run3/blockIMG.json`, `blockP64.json`, `sweep_scores.json`; `figures/run3_killing_dim.png`, `run3_fresh_seeds.png`, `run3_group_formation.png`.
