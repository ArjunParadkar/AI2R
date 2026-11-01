# FINDINGS D — functional alignment, the evasion, and the interrupt

## The evasion result (stated as a finding, regardless of everything else)

> **A monitor that certifies a property of the representation's shape, used as a training signal, can be satisfied by a
> re-encoding that leaves the function unchanged.**

*Setting.* Z113 transformer with a planted shortcut (a hint token that leaks the answer for 40 of 113 residues). Monitor: the
run-1 H¹ detector — a persistent circle in resid_mid whose coordinate tracks the hint and to which the output is sensitive.
Training signal: the spec's L_c taken literally, a penalty on output change along the *discovered* circle at that layer, with
no knowledge of which input token the nuisance is; λ escalated 3 → 30 → 300 while the class persisted.

*Run 1 (2026-09-20 03:05, `logs/p3_structure_s0.log`).* **Three consecutive checkpoints (steps 3500, 4000, 4500) with zero
detected classes at ablation reliance 1.000**, decorrelated accuracy 0.014–0.016.

*Run 2 (regenerated under AMENDMENTS A-1 because run 1 never checkpointed the model; same seed, same code).* The trajectory
reproduces step for step. Seed 0: zero detected classes at steps 3500, 4000, 4500, 5500, 6000 — five of the last six checks —
with reliance 1.000 and decorrelated accuracy 0.014 throughout. Seed 1: zero classes at step 4000 (reliance 1.000), plus step
500, which is late first detection rather than re-encoding. On the evasion window the prediction equals the hint on 36% of
decorrelated probe rows, i.e. on every row whose hint is a leaked residue: the function is untouched.

*Sketch of why (an argument, not a proof).* The penalty's gradient is zero once the representation has no circle along which to move;
the loss that rewards the shortcut is unchanged by any invertible re-encoding of the hint inside the layer. The optimiser has
a free direction that satisfies the monitor and keeps the function, and it takes it.

## In plain English

The brief's replacement detector — mutual information between Mapper graphs built with the prediction and with the nuisance as
lenses — fails by the letter and, corrected, fails its null. What catches the re-encoded model is the plain dependence of the
prediction on the hint on a probe where hint and answer are independent. That is the ablation ground truth in another form, and
it needs z. The lesson is the asymmetry between monitoring shape and monitoring function, not a new instrument.

**Contradicting the brief's framing, first.**
1. The Mapper alignment does not match the floor it was meant to generalise (details in `PAUSE_2.md`): unreachable threshold as
   registered (1.139); chance-adjusted it false-alarms on 44/102 decoy checkpoints because it detects that z is *represented*.
2. "Generalising the KL-consistency loss that worked" made it worse. With z' drawn uniformly, a present hint is compared with
   NULL about once in 114 samples; run 1 had found that presence-invariance is the part that matters.
3. With z hidden (D5) nothing singles out the shortcut: best hint-tracking class r = 0.38–0.55, ranked 12th–29th of 30 in 7 of 8
   checkpoints.

| cell | pre-registered | measured | met? |
|---|---|---|---|
| A-1 evasion reproduces | ≥1 checkpoint, 0 H¹ classes at reliance > 0.9 | seed 0: 5 checkpoints; seed 1: 2 | reproduced |
| **D2** | raw Mapper NMI above clean mean + 3σ on every evasion checkpoint | threshold 1.139 > max(NMI); A in window 0.984–0.999 | **NOT MET** |
| floor: direct NMI(ŷ, z), threshold 0.537 | reported | evasion 7/7; shortcut 50/51 ×2; clean control + decoys 0/153 | — |
| secondary: Mapper AMI, threshold 0.126 | reported (A-2) | evasion 7/7; shortcut 50/51 ×2; clean control 0/51; decoys 19/51, 25/51 | fails its null |
| **D3** | decorrelated acc > 0.98 on 5/5 seeds, schedule R | by the letter the trigger never fires → no intervention (0.012–0.016) | **NOT MET** |
| **D4** | alignment below threshold throughout (6000, 12000] | vacuous by the letter | **NOT MET** (nothing passed D3) |
| D nulls | 0 firings on clean control and decoys | raw NMI 0 (vacuous); direct 0/153 | met for the direct statistic |
| D5 | none (exploratory) | shortcut class not surfaced | — |

## D3/D4 under the POST-HOC trigger (AMENDMENTS A-3, A-4, A-5) — cannot upgrade a verdict

| arm | trigger | penalty | decorrelated accuracy at step 6000, seeds 0–4 | max λ | D3 (>0.98) |
|---|---|---|---|---|---|
| by the letter | raw Mapper NMI (cannot fire) | — | 0.012–0.016 (= no intervention) | 0 | 0 / 5 |
| R | direct NMI (POST-HOC) | registered: uniform z ≠ z' | 0.011 / 0.010 / 0.009 / 1.000 / 0.010 | 300, 300, 300, 3, 300 | 1 / 5 |
| G (6000 steps, A-5) | direct NMI (POST-HOC) | registered | 0.012 / 0.014 / 0.013 / 1.000 / 0.012 | 81, 81, 81, 3, 81 | 1 / 5 |
| A-4 | direct NMI (POST-HOC) | run 1's: z vs z+1 AND z vs NULL | 0.999 / 1.000 / 1.000 / 1.000 / 1.000 | 3, 3, 3, 3, 3 | **5 / 5** |

The direct trigger fired at step 500 on every seed in every arm (run 1's H¹ detector: 500 / 1000 / 1500 / 500 / 1000).
D4 (the one R seed that passed, seed 3): 0 firings in 12 checks (direct NMI ≤ 0.409 < 0.537); reliance ≤ 0.09; decorrelated accuracy 1.000 except one transient dip to 0.769. Schedule-R seeds 0, 1, 2, 4 were stopped at step 6500 (A-6).
Every success in either run happened at λ = 3 in one round; every model that was escalated past it ended at accuracy 0.01 with the
shortcut gone and the task with it. The A-4 arm rescues seed 1, which failed under both schedules in run 1 when it was caught at
step 1000 rather than 500. All of this is post-hoc, one run per seed, and needs z.

**Claimable.** The evasion, as stated above, reproducible from a seed. That a decorrelated-probe functional test is not evaded
and has no false positives on "present but unused". NOT claimable: any Mapper/Reeb alignment detector; any unsupervised
discovery of the nuisance; anything from the post-hoc interrupt arms beyond "observed on five seeds, once".

Timings: D1 ~1 s per checkpoint; regenerating the re-encoded models ~25 min (GPU shared); D5 ~1 min per checkpoint.
Files: `results/run2/blockD12.json`, `blockD5.json`, `d34_summary.json`, `runs/r2_reencoded_s*`, `runs/r2_align_*`,
`figures/run2_alignment_vs_h1.png`.
