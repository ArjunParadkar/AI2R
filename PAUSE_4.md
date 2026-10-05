# PAUSE 4 — interrupt, quotients, vineyard  (D3–D5, E3, F) · 2026-09-20 13:10

**Plain English.** By the letter D3 and D4 are not met: the registered trigger cannot fire, so the registered interrupt is the
no-intervention run. Run with the floor statistic as trigger (post-hoc), the brief's "generalised" KL penalty is *worse* than
run 1's: it clears the shortcut on 1 of 5 seeds and flattens the other four. Run 1's own penalty with the same earlier trigger
clears **5 of 5**, including the seed that failed twice in run 1 — post-hoc, one run per seed. E3's Z₂ quotient replicates on
two unseen models; its rot-MNIST half fails. The vineyard fails, and it fails because an unseen model was included.

**Contradicting the brief's framing, first.**
1. "L_align generalises the KL-consistency loss that worked" — it removes the part that worked. Uniform z ≠ z' compares a present
   hint with NULL about once in 114 samples; presence-invariance is what breaks the shortcut.
2. Escalation does harm. Every model that reached λ = 300 (R) or λ = 81 (G) ended with decorrelated accuracy 0.01: the shortcut
   is gone (reliance → 0) and so is the task. Every success, in both runs, happened at λ = 3 in one round.
3. D5: with z hidden, sensitivity-ranked H¹ classes do not surface the shortcut (best tracking r 0.38–0.55, ranked 12–29 of 30).

| cell | pre-registered | measured | met? |
|---|---|---|---|
| D3 | decorrelated acc > 0.98 at step 6000 on 5/5 seeds, schedule R, trigger = D1 | by the letter D1 cannot fire → no intervention, 0.012–0.016 | **NOT MET** |
| D4 | no firing in (6000, 12000] on seeds that passed D3 | vacuous by the letter | **NOT MET** |
| D3, POST-HOC trigger (A-3), registered penalty, schedule R | — | 0.011 / 0.010 / 0.009 / 1.000 / 0.010 (seeds 0–4); max λ 300, 300, 300, 3, 300: four seeds escalated to 300 and were flattened (reliance → 0, accuracy 0.01), seed 3 cleared at λ = 3 | 1 / 5 |
| D4, POST-HOC trigger, the one seed that passed | — | seed 3, steps 6500–12000: 0 firings in 12 checks (direct NMI ≤ 0.409 < 0.537); reliance ≤ 0.09; decorrelated accuracy 1.000 except one transient dip to 0.769 | — |
| D3, POST-HOC trigger, registered penalty, schedule G (6000 steps, A-5) | — | 0.012 / 0.014 / 0.013 / 1.000 / 0.012; max λ 81, 81, 81, 3, 81: the same four seeds escalated to 81 and were flattened, seed 3 cleared at λ = 3 | 1 / 5 |
| **A-4 arm**: run-1 penalty + direct trigger, schedule R | — | 0.999 / 1.000 / 1.000 / 1.000 / 1.000, one round at λ = 3, first fired at step 500 on every seed | 5 / 5 (post-hoc) |
| E3 Z113 (UNSEEN decoy_s0, decoy_s1) | axis ratio < 0.1, slice > 0.9, swap < 0.02 | 0.019 / 0.997 / 0.0051 · 0.044 / 0.997 / 0.0081 | **MET** |
| E3 rot-MNIST | invariance depth exists | both nets lose the circle at conv3_gap | **NOT MET** |
| F | mode ±2 on every seed; W2 < null on every pair; spreads < 0.30 | 4/5 seeds; 6/10 pairs; count spread 0.71; lifetime spread 0.05 | **NOT MET** |

**Why seed 1 failed in run 1, now answerable.** In run 1 the H¹ detector first fired on seed 1 at step 1000; here the direct
statistic fires at step 500 on every seed, and with the same penalty seed 1 reaches 1.000. The shortcut is removable at λ = 3
if it is caught by step 500 on this seed and not at step 1000. (Seed 2 was caught at 1500 in run 1 and still recovered, so
lateness is seed-dependent, not a rule.) One run per seed; not a verdict.

**Kill tests.** Direct-trigger nulls: 0 firings in 153 clean-control and decoy checkpoints. Decoy models are the UNSEEN set for
E3 (passed) and F (one of them broke it).

**What is now claimable.** The evasion finding (FINDINGS_D.md). The Z₂ quotient on five models. That, post-hoc, a functional
trigger plus presence-and-value KL consistency removed the planted shortcut on five of five seeds at λ = 3 in one round, and
that every escalation beyond that destroyed the model. None of this is topological, and all of it needs z.

Timings: interrupt arms 1.5–2.5 h wall with 6–11 processes sharing one 8 GB GPU (two G seeds died of CUDA OOM and were rerun);
E3 40 s; F 3 min; D5 8 min.
