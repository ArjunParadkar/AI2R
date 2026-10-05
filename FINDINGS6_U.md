# FINDINGS (Run 6) U — one object

**By the letter.** Spearman(κ_s, ε̄) over pooled in + unknown inputs: −0.84, −0.68, −0.66 (Z113 clean 12–14), −0.97, −0.97 (ResNet+aug 5, 6): all
< −0.6, **PASS**. Memorisers: κ interquartile range 0.00 (Z113) and 0.15–0.23 (ResNet) — vacuous on Z113 as the brief said, and reported.
Cited: LieGG; Golan & El-Yaniv 2018.

**What it means.** As SPEC6 §1 said before computing anything, κ_s and ε̄ are a cosine and a norm of the same orbit residuals; a strong
negative correlation between them is largely the construction. The registered test passes and is not read as "the boundary of knowledge and
the failure field coincide". The external evidence is in K and F: the same orbit statistic finds an unknown input on the transformer (AUROC
0.99) and does not find one on the CNN (0.35–0.69, below every softmax baseline), and it does not predict failure on either (F).

**The theory, stated for what held.** Six runs, on 30-odd fresh models:
- **P1** what it learned — the group and its generator, from outputs, on every generaliser and no memoriser (Runs 4–6, registered).
- **P2** why it decided — orbit coherence κ = 1.000 on generalisers, 0.000–0.23 on memorisers (Runs 4–6, registered as K.4).
- **P6** whether it understood — the four-instrument definition, registered on the transformer (Runs 5–6), three instruments on the CNN.
- **P7** how it formed — the generator arrives with the jump (leads +0, +0, +1, +4, +0, −1 on dense sweeps), never on a memoriser; persistence 0.87–1.00.
- **P3** spurious structure — the dominance rule and the functional interrupt (Run 4, registered).
- **P4** what it does not know — orbit coherence works where the group is *learned* (Z113 OOV, 0.99) and fails where the group is the
  architecture's (CNN, below softmax). Descriptive.
- **P5** where it will fail — no orbit quantity beats the softmax margin. Descriptive.
The unified sentence holds for five problems and for the sixth on one of two systems. The seventh is not answered by this object.
