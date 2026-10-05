# PAUSE 2 — does alignment catch what H¹ missed?  (D1, D2) · 2026-09-20 10:08

**Plain English.** By the letter, no: the pre-registered detector cannot fire on anything. Its threshold (mean + 3σ of raw
Mapper NMI on clean models) comes out at 1.139, above the maximum value NMI can take, because with ~1,000 Mapper nodes for 2,000
points raw NMI sits near 0.87 on every model. It does not fire on the evasion window, and it does not fire on the ordinary
planted-shortcut models either (0/51, 0/51). D2 is NOT MET.
What does catch the re-encoded model is the statistic SPEC2 registered as the *floor*: the plain NMI between the model's
prediction and the hint on a probe where the hint is independent of the answer. It fires on every checkpoint of the evasion
window on both seeds, and never on a clean or decoy model.

**Contradicting the brief's framing, stated first.**
1. **The topological version does not match its own floor.** The Mapper alignment is *worse* than the direct statistic it was
   meant to generalise. Chance-adjusted (AMI, registered as a secondary before any number existed — AMENDMENTS A-2) it does
   separate shortcut from clean (0.96–1.00 vs ≤ 0.12), but it also fires on the **decoy** models — 19/51 and 25/51 checkpoints,
   AMI up to 0.35 — where the hint is present, the representation encodes it, and the output provably ignores it (ablation
   changes 0.0% of predictions). Mapper built on the activations inherits the shape detector's weakness: it sees that z is
   *represented*, not that z is *used*. The direct statistic has 0 false positives on the same decoys.
2. **The re-encoding was never hidden from a functional test.** Direct NMI(ŷ, z) > threshold is the ablation-reliance ground truth
   in another form (on the evasion window the prediction equals the hint on 36% of probe rows = every row whose hint is a leaked
   residue). The "detector that catches the evasion" is the measurement that defined the evasion. What the experiment shows is
   the asymmetry, not a new instrument: **shape can be evaded by re-encoding; function cannot, but function needs z.**
3. The evasion window on seed 1 includes step 500, which is late first detection (run 1: H¹ fired at step 1000 on this seed),
   not re-encoding. By the pre-registered definition it counts; it is flagged here.

| cell | pre-registered | measured | met? |
|---|---|---|---|
| A-1: evasion reproduces in the regenerated run | ≥1 checkpoint with 0 H¹ classes at reliance > 0.9 | seed 0: steps 3500, 4000, 4500, 5500, 6000 (reliance 1.000, decorrelated acc 0.014); seed 1: steps 500, 4000 | reproduced (trajectory identical to run 1 through step 4500) |
| **D2** | raw Mapper NMI above clean mean + 3σ on every evasion checkpoint | threshold 1.139 (unreachable); A in window 0.984–0.993 (s0), 0.986–0.999 (s1) | **NOT MET** |
| D nulls (raw NMI) | 0 firings on cleanctrl_s2, decoy_s0, decoy_s1 | 0 / 0 / 0 (vacuous: nothing can fire) | met, vacuously |
| floor: direct NMI(ŷ,z), threshold 0.537 | reported | evasion window 5/5 (s0), 2/2 (s1); shortcut 50/51, 50/51; nulls 0/51 ×3 | — |
| secondary: Mapper AMI, threshold 0.126 | reported (A-2) | evasion window 5/5, 2/2 (0.958–0.997); shortcut 50/51 ×2; clean ctrl 0/51; **decoys 19/51, 25/51** | — |

**Kill tests.** Clean-control and decoy nulls for the direct statistic: 0 firings in 153 checkpoints. For Mapper-AMI: clean
control 0/51, decoys 44/102 — fails its null.

**What is now claimable.**
* The evasion result is reproducible and deterministic: same seed, same code → the same trajectory; on seed 0 the H¹ detector
  is silent at five of the last six checks while reliance is 1.000 and decorrelated accuracy 0.014.
* A functional test on a decorrelated probe (prediction–nuisance dependence) is not evaded by the re-encoding and has no false
  positives on "nuisance present but unused". It requires knowing z. It is not topological, and Mapper adds nothing to it here.
* NOT claimable: that a Mapper/Reeb alignment detector works. By the letter it cannot fire; adjusted, it false-alarms on decoys.

**Headline implication.** The paper's headline cannot be "alignment detector succeeds where H¹ failed". The defensible
headline is the asymmetry itself, with the evasion as its demonstration.

Timings: ~1 s per checkpoint (2,000-point probe, two Mapper constructions); full scan of 9 runs 6 min.
Files: `results/run2/blockD12.json`, `analysis/run2_blockD12.py`, `logs/run2_reencoded_s*.log`, `runs/r2_reencoded_s*/`.
D3/D4 proceed under AMENDMENTS A-3 (post-hoc trigger = the direct statistic).
