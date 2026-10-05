# PAUSE 5.1 — specimens and the sanity gate · 2026-09-21 13:52

**Plain English.** All eleven registered specimens exist and all eleven pass the gate. The augmented clean ResNets reach ≥ 99% on the
un-augmented training images (82 and 84 epochs) and their shift consistency is 0.988 and 0.984 — above the registered 0.75. Nothing else
had been read or scored when this was written (the Z113 sweeps were computing; their output was not opened). Not waiting here, as instructed.

| specimen | train | test | gate | note |
|---|---|---|---|---|
| Z113 clean s9 / s10 / s11 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | ok | test ≥ 0.99 first at step 7500 / 4500 / 6000 |
| Z113 memoriser (scrambled_decoy) s9 / s10 | 1.000 / 1.000 | 0.0087 / 0.0087 | ok (≤ 0.02) | max test over training 0.0105 / 0.0098 |
| Z113 decoy s9 / s10 | 1.000 / 1.000 | 1.000 / 1.000 | ok | test ≥ 0.99 first at step 20500 / 13000 |
| ResNet clean + augmentation s3 / s4 | 0.992 / 0.990 | 0.795 / 0.764 | ok | 82 / 84 epochs; Run 4's un-augmented clean nets: 13–14 epochs, test 0.58 |
| ResNet memoriser (random labels, no aug) s3 / s4 | 0.997 / 0.994 | 0.094 / 0.111 | ok (≤ 0.15) | 12 / 15 epochs |

**Shift consistency (P6.b instrument, reported here because the brief asks for it before anything else):**
clean + aug 0.988, 0.984 · memorisers 0.202, 0.180 · the same four networks at initialisation 0.989, 0.853, 0.989, 0.853.

**Two things to say before scoring.**
1. The registered clean-side clause will pass, and it was made to: random-crop augmentation *trains* shift consistency. With augmentation on one
   side of the pair only, P6.b no longer compares like with like — it cannot by itself say whether a memoriser's 0.19 comes from memorising or
   from the absence of augmentation. The like-for-like comparison is Run 4's (no augmentation on either side): 0.68 vs 0.18. Both will be
   reported: memorisation costs ≈ 0.5 of shift consistency at equal recipe; augmentation supplies the remaining ≈ 0.3 on the clean side.
2. As SPEC5 said in advance (point 3), an untrained network scores 0.85–0.99 here. A trained, augmented network at 0.988 is
   indistinguishable on this instrument from the same network before training (0.989). The instrument reads "symmetry preserved", not "symmetry learned".

Timings (GPU at full clock today, 2,350 MHz, unlike Run 4): seven Z113 runs concurrently 23–30 min; augmented ResNets 29 min and 6 min
(the second ran alone); memorisers under a minute each.
