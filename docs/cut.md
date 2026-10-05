# cut.md — formalisms cut or downgraded, and why (rule 1: no ornamental mathematics)
Written 2026-10-05; entries marked PENDING are decided when the Phase 2 analysis runs.

| item | decision | reason / evidence |
|---|---|---|
| M4.3 Marsden-Weinstein reduction ("every certified G-invariant quantity is a function on the reduced space") | CUT as a claim (kept only as a one-line remark) | true by construction (invariance is descent to the quotient); yields no measurable prediction beyond M0.1 (theory/core.tex, proof-status table) |
| M4 momentum-map / H-symp | PENDING | kill rule in prereg/PHASE_2.md: cut if the J_attn signal does not precede or coincide with generator acceptance on >= 80% of seeds, or adds nothing beyond weight norm |
| Killing-algebra dimension for curved, non-constant-coefficient fields (M1.2 solver) | CUT from every empirical claim | failed its Phase-1 gate twice (T-N4-02 sphere band: dim 1 of 3, subspace error 3.7e-3 then 8.5e-3 after post-hoc tuning) ; the image-manifold case (T-N4-03b) passes only after a post-hoc gap rule |
| Certified Betti numbers (M3.1 certificate) at the Phase-1 sample sizes | CUT: not used on real models | T-N5-02: certificate (bar life > 4 d_H) cannot certify S^2 (n=300) or the Clifford torus (n=500); only circle cases pass |
| M3.2-M3.4 (quotient homology, circular coordinates, Mapper) | NOT FORMALISED | no testable prediction derived in this programme; circular structure was tested directly (Phase 4 days/months, Phase 6 tori) with shuffle / permutation nulls, not through these theorems |
| Connectivity-route "torus recovered from wiring" as a general claim | DOWNGRADED | planted-wiring validation: 87% of modules (two fresh-seed sets); no trained-network evidence (our grid RNN produced no grid cells) |
| Foliation-aligned-with-rotation claim (M1.1 on rotated MNIST) | NOT CLAIMED | H3.1b failed: a rotation-augmented memoriser has the same kernel alignment as the generaliser; rotated test accuracy separates every condition the instrument does |
| 'induction' task as a zoo member | DROPPED before the grid | not learned by any architecture in the pilots (test acc <= 0.15) |
| H2.1 as evidence of a mechanism-level clock/pizza distinction | DOWNGRADED in the registration itself | the registered descriptor swap_asym is nearly forced by the architecture that also defines the label (uniform attention is permutation-symmetric) |
