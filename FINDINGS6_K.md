# FINDINGS (Run 6) K — P4 as orbit coherence

**Plain English.** Orbit coherence finds an out-of-vocabulary input on the transformer at AUROC 0.994–0.996 on three fresh seeds, beating max
softmax (0.79–0.89) and entropy. It cannot tell a p′ = 97 pair from a p = 113 pair because they are the same inputs (SPEC6 §2: AUROC 0.500,
as predicted), so K.2 fails by the letter and passes on the set that is actually unknown. On the CNN it **fails everywhere**: an augmented
ResNet is shift-invariant on SVHN, CIFAR-100 and noise just as it is on CIFAR-10 — more so — and κ_s is *below chance* on SVHN (0.35, 0.42).
Every softmax baseline beats it. Golan's 72 designer transformations do worse still (0.11–0.45). Memorisers know nothing by this definition
(κ = 0.00 / 0.23). Cited: Golan & El-Yaniv 2018; Hendrycks & Gimpel 2017; Liang et al. 2018 (ODIN); LieGG.

| cell | registered | measured | verdict |
|---|---|---|---|
| K.2 Z113 OOV (slot a = NULL/EQ token; κ on the valid slot's orbit) | AUROC(κ) ≥ 0.95, 3/3 | **0.994, 0.996, 0.996** (κ_s 0.985, 0.940, 0.987; max-softmax 0.793, 0.894, 0.830; entropy 0.79–0.90); unknown κ median 0.33 | passes |
| K.2 Z113 OOV-continuous (random vector at the embedding norm) | reported | κ 0.994, 0.995, 0.999; max-softmax 1.000 (a random vector is also *confidently* wrong) | — |
| K.2 Z113 p′ = 97 | AUROC(κ) ≥ 0.95, 3/3 | 0.500, 0.500, 0.500 (max-softmax 0.499, 0.490, 0.500); unknown κ = 1.0 | fails, as SPEC6 §2 said |
| **K.2** | both sets | | **FAIL** by the letter; PASS on OOV |
| K.3 ResNet SVHN | κ_s > msp, entropy, ODIN; ≥ Golan-72 | κ_s **0.351 / 0.418**; msp 0.809 / 0.738; entropy 0.821 / 0.742; ODIN 0.811 / 0.750; Golan 0.113 / 0.199 | fails |
| K.3 ResNet CIFAR-100 | same | κ_s 0.692 / 0.679; msp 0.710 / 0.710; entropy 0.717 / 0.715; ODIN 0.742 / 0.738; Golan 0.453 / 0.445 | fails |
| K.3 ResNet noise (reported) | | κ_s 0.602 / 0.431; msp 0.798 / 0.803; Golan 0.006 / 0.002 | — |
| **K.3** | | | **FAIL** (beats Golan 6/6 cells; beats no softmax baseline anywhere) |
| K.4 memoriser κ on own training set < 0.3; clean > 0.95 | all | Z113 mem 0.00, 0.00, 0.00; ResNet mem 0.23, 0.23, 0.23; clean 1.00 ×5 | **PASS** |

**Why the CNN fails, in one number.** Mean κ_s on CIFAR-10 test 0.983; on SVHN 0.989; on CIFAR-100 0.974; on Gaussian noise 0.983; on the
*untrained* network 0.996. Shift-and-flip coherence is a property of the architecture plus augmentation, not of what the network knows about
an image; it holds on anything you feed it. The same fact was written into SPEC5 §3 and SPEC6 for the P6 nulls ("an untrained network has its
architecture's symmetries") — it was not carried through to K.3, where it is fatal. The Z113 case works because the transformer's group is
*learned* (translation by the discovered generator, with the answer relabelled): an OOV token has no place on that orbit.

**Golan-72 is worse than κ_s everywhere** (6/6 cells), so the registered "matches or exceeds designer transformations" holds — but both lose
to max softmax, and the comparison is between two things that do not work on this network.

**K.5 knowledge map.** Inside CIFAR-10 the lowest-κ_s images are not random: 10/16 and 9/16 are misclassified (test error is 22%), and the
rest are ambiguous poses (a frog on black, a bird in a cage). Lowest-κ_s Z113 pairs on seed 12 — (25, 104), (37, 74), (31, 98), (25, 38) —
are all classified correctly with κ = 1; the soft score sees a graded margin the hard score does not.

Files: `results/run6/z113_finals.json`, `resnet.json`, `raw_*.json`, `figures/run6_roc.png`, `run6_knowledge_map.png`.
