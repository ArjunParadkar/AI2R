# Phase 0 report (2026-10-04)
Authors: Arjun Paradkar and Reva Paradkar.

## Deliverables
- `paper/v1.1/paper.tex`, `related.tex`, `refs.bib`, `paper.pdf` (pdflatex + bibtex available; builds clean, 13 pp). Rebuilt from the two private artifacts (math decoded from SVG glyphs) plus CLAIMS.md.
- E1-E9 applied: corrected permutation p = 1/19,448; 17 models (10+7), "fresh/pairs" dropped; P7 5/6 + 1/6 wording; "level B" not "outputs alone"; Proposition (evasion) with one-penalty/one-architecture/one-task demonstration; no "first"/"theorem"/"breakthrough"; abstract around three claims; title narrowed to modular addition; A/B/C/D tags on every method (D unused); related work for all E8 items.
- docs/record.md and EDITS.md updated.

## Reconciliation (from results/)
1. 102 vs 0/204: 102 = Run 5 memorizer sweeps (2 seeds x 51). Spec-gated swept memorizers on disk: Run 4 (2x18=36) + Run 5 (2x51=102) = 138, generator accepted on 0. Run 6 memorizers have final-model checks only (3/3 vs 0/3), no sweeps; Run 3 sweeps are pre-gate. 204 cannot be reproduced; paper uses run-level counts and 138 descriptively.
2. Duplicate rho: -0.99/-1.00 is the Run 5 decoy nuisance-share decline (P7.c: -0.991 over 39 ckpts, -1.000 over 24; two decoys, autocorrelated, descriptive). "Slot collapse" has no rho; it was a mislabel in 7.12. (My recomputation up to the jump gave -0.87/-0.81; score.json's window differs; paper quotes score.json.)
3. Unrequested catch: Run 7 P5 test accuracies are 0.09, 0.785, 0.880 (v1: "0.51-0.88"). Seed 18 is pre-grokking (90% errors); generator found at the scored checkpoint on 1/3 seeds. Claim (3) is stated as three seeds, no CI.
Leads verified from sweep files: 0,0,+1 (Run 5), +4,0,-1 (Run 6); decoys 0,0.

## Bibliography
All 23 entries verified by web search (3 parallel verifier agents; I re-fetched arXiv for Bailey and Hwang & Park). Not confirmed on a fetched page: Lenc pages, Hauser pages (search snippets); venues of Bailey (ICLR 2026) and Hwang & Park (ICML 2026, cited as arXiv) rest on the verifier agents. Corrections to the E8 list are in EDITS.md.

## Open
CLAIMS.md 463 still says 0/204; board/deck artifacts and report HTMLs unrevised; bootstrap CIs over seeds and Holm correction not yet computed (planned v2); no Artifact page published (LaTeX is the deliverable).
