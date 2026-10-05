# WHITEBOX v1.1 — Phase 0 exact edits (2026-10-01)

Authors on every document: Arjun Paradkar and Reva Paradkar. No university affiliation.

## Where v1 lives (no .tex/PDF exists on this machine)

The paper text exists only as private claude.ai artifacts plus `CLAIMS.md`. All quotes below were read from them (math decoded from the SVG glyph codes).

| Source | URL / path |
|---|---|
| Paper, Sections 1, 2, 4–9 ("Rest of the Paper") | https://claude.ai/artifact/8Gvv4iySjRUqFHKyhmrgq8 |
| Paper, Section 3 (math) | https://claude.ai/artifact/TsXoT3KLjSdtAeDzakg6UR |
| Board / deck "did the AI understand?" | https://claude.ai/artifact/WnLyq5uW1BEUwpasMBYMeS |
| Board / deck "reading what a network understands" | https://claude.ai/artifact/WfrxviJjqkg7AgDJzbSY8x |
| Registry of claims | `../../CLAIMS.md` (patched, backup `CLAIMS.md.v1_backup`) |

Artifacts are NOT republished. The next step is to rebuild them from the corrected text.

## Edits

**E1. Separation p-value** (Paper §7.12, "Separation (P6)")
- Before: `no overlap across 17 pairs; p < 10⁻⁵ under exchangeability`
- After: `no overlap across 17 models (10 generalizers, 7 memorizers); one-sided exact permutation p = 1/C(17,7) = 1/19,448 ≈ 5.1×10⁻⁵ for complete separation on a single instrument. The instruments are correlated; p-values are not combined.`
- Check: C(17,7) = 19,448.

**E2. Formation p-value** (Paper §7.12, "Formation (P7)")
- Before: `6/6 vs 0/102 memorizer checkpoints, binomial p < 10⁻³⁰`
- After: `Retracted: the 102 memorizer checkpoints come from a few autocorrelated runs (pseudoreplication). Unit = training run: generator accepted at or before the jump on 5/6 generalizer seeds, one checkpoint after on 1/6; on 0 of the memorizer runs (checkpoint counts reported descriptively, no p-value).`

**E3. "17 matched pairs / pairs / fresh matched models"**
- §1.4 row 3: `no overlap on 17 fresh matched models` → `no overlap on 17 models (10 generalizers + 7 memorizers)`
- §7.3 plane caption: `for all 17 models` (already right; keep)
- §7.12: fixed in E1
- Board "did the AI understand?": `17 AIs, zero mistakes`, `17 / 17 right`; board "reading what a network understands": `P6: fresh pairs, zero overlap` and `17 fresh models`. Reword to `17 models (10 + 7), no overlap` and drop "fresh"/"pairs" unless the pairing and freshness are documented in `results/run{4,5}/`.
- Open check: Run 5 trained 11 fresh models, so "fresh" cannot be true of all 17. Drop it.

**E4. P7 timing** (Paper §7.4 bullet 1; §1.4 row 4; Board; `CLAIMS.md` row P7.a — already patched)
- Before: `first accepted, at the jump or one checkpoint before (leads 0, 0, +1, +4, 0, −1)` and row 4 `three structural events coincide at the accuracy jump`
- After: `accepted at or before the jump on 5/6 seeds (leads 0, 0, +1, +4, 0; up to 4 checkpoints early) and one checkpoint after on 1/6 (−1). Coincident marker, not a predictor.`
- Reason: a lead of −1 is "after", so "at the jump or one checkpoint before" is false for the sixth seed. §7.4's last paragraph also withdraws "leads 1–4" while the bullet reports a +4. Reconcile both: the +4 is the sparse-sweep seed.
- Also fix `CLAIMS.md` lines 346–348: "Never after the jump holds 5/5" scopes to the dense Run 5 seeds only. The six-seed statement is the one above.

**E5. "From outputs alone"** (Paper §1.2 claim, §7.1 heading, §1.4 row 1 "Group discovery from outputs"; `CLAIMS.md` 297 and 458 — already patched; report HTMLs run3:82, run4:81 and 148, run7:123)
- After: `from the input–output map with Jacobian access (level B)`. Methods that use no gradients are level A. Methods that use internals are level C.

**E6. "Evasion theorem"** (Paper §1.4 row 6, §7.8 heading and "Theorem." label; `CLAIMS.md` 171 "theorem-shaped")
- After: `Proposition (evasion), with a demonstration.` State what was shown: one penalty, one architecture, one task. Cite Bailey et al. (ICLR 2026) for the general phenomenon. Keep "regenerated step for step".
- No proof is in `proofs/`; it is not a theorem.

**E7. "First"** (`CLAIMS.md` 297–298 "First pre-registered pass of a discovery cell in four runs")
- After: `Pre-registered pass of a discovery cell (Run 4, on four unseen generalizers).`

**E8. Related work to add** (new §2 or §9): LieSD, LieGAN, LieGG, Augerino, Lenc & Vedaldi, Goodfellow 2009, Bouchacourt 2021, Kvinge 2022, Deng 2022 (EI), Bahat et al. (TTA error detection), Stander 2024, Chughtai 2023, Morwani, Zhong 2023 (clock/pizza), Nanda 2023, counterfactual logit pairing, Hauser & Ray 2017, Benfenati & Marta 2023, Kunin et al. 2021, Hein & Andriushchenko 2017, "Intrinsic Task Symmetry Drives Generalization" (2026).
- Each needs a sentence on how it differs. Verify every bibliographic entry before it goes into the paper; none were looked up in this pass.

**E9. Abstract / title / access levels**
- Rewrite the abstract around three defended claims: (1) the methodology, (2) the built-in-symmetry null (Proposition 1), (3) mid-grokking failure prediction (AUROC 0.983–1.000 vs 0.56–0.90; random-transformation null at chance).
- Narrow the title to the modular-addition scope. Tag every method A/B/C/D.
- "Knowledge is a group" (§1.2) becomes "on the tasks tested, generalizers had a recoverable group and memorizers did not".

## Found while reading (not in your list)

- §7.12 "Slot collapse: Spearman ρ = −0.99, −1.00" is quoted next to the decoy-share claim in §7.4. The same ρ appears for two different quantities in the extracted text. Confirm against the source numbers.
- The §7.4 "memorizer checkpoints: 102" and `CLAIMS.md` "0/204" are different denominators (Run 5 vs Runs 5–6). Reconcile them when rebuilding.
- `CLAIMS.md` line 15 (P1 rot-MNIST) records a FAIL, so any "learned" statement about rotation circles must carry that.

## Status (updated 2026-10-04, Phase 0 complete)

- Done: E1-E9 applied in `paper.tex` + `related.tex` + `refs.bib` (compiles with pdflatex+bibtex to `paper.pdf`, 13 pages). E8: all 23 references web-verified (verification URL in `refs.bib`); `intrinsic2026` cited as arXiv preprint (arXiv 2603.01968, Hwang & Park; real title ends "in Algorithmic Tasks"), venue not confirmed by the author of this pass.
- Corrected beyond the E-list: Deng 2022 first author is Weijian Deng; "LieSD" is Hu, Li, Lin 2024 (arXiv); SPARE is Yang et al. AISTATS 2024; Bailey et al. = "Obfuscated Activations Bypass LLM Latent-Space Defenses" (ICLR 2026 proceedings page per the verifier; arXiv Dec 2024).
- Reconciled: "102" = Run 5 only; "0/204" not reproducible (138 on disk, Runs 4+5, 0 accepted). The duplicated rho = -0.99/-1.00 belongs to the decoy nuisance-share decline (P7.c), NOT slot collapse (no rho exists for slot collapse); relabelled. New finding: P5 natural-failure test accuracies are 0.09/0.785/0.880, not 0.51-0.88. See `reports/phase0.md`.
- Not done: board/deck artifacts and report HTMLs (run3:82, run4:81/148, run7:123) still carry old wording; CLAIMS.md line 463 still says 0/204 (not edited: shared registry, flagged); no Artifact summary page published.
