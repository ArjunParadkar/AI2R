# Block F — novelty log

Searches run 2026-09-07 (web search, top hits recorded; re-search required within 14 days of release, L10).

**Search 1: "parameter symmetry explanation identifiability neural network circuits non-identifiability"**
* Symmetry in Neural Network Parameter Spaces (survey, arXiv 2506.13018) — the parameter-symmetry survey the handbook cites; no interpretability audit.
* Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability (arXiv 2606.04754, June 2026) — identifiability of neurons beyond permutation/scaling, applied to mode connectivity; adjacent to F3 (hidden symmetries) and Z4 (path connectivity), not to explanation orbits.
* Parameter identifiability of a deep feedforward ReLU network (arXiv 2112.12982) — G is complete (perm × scaling) outside degenerate cases; the theoretical basis for treating excess fiber dimension (F3) as degeneracy.
* Méloux et al., Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable? (ICLR 2025, arXiv 2502.20914) — the enumeration protocol; multiplicity established, gauge not considered.
* Explanation Multiplicity: Circuit-Level Interpretability Evidence Does Not Survive Defensible Analytic Variation (arXiv 2608.13754, Aug 2026); Many Circuits, One Mechanism (arXiv 2606.06267, June 2026) — multiplicity under analytic variation / input variation; neither partitions explanations by the network's gauge group.
**Status: OPEN** for "explanation multiplicity partitioned by gauge orbits" — no hit computes orbits of explanations. Closest: 2606.04754 (neuron identifiability).

**Search 2: "implementation invariance mechanistic interpretability audit permutation symmetry interpretability methods"**
* Symmetry Defeats Auditing (arXiv 2605.27836, May 2026) — LoRA-based *model auditors* are not invariant under the model's symmetries and this is exploitable. Same principle (auditing must be gauge-invariant), different object (auditor of behaviour, not interpretability metrics). **Must be cited**; our audit table is the metric-level counterpart.
* Mechanistic Interpretability in the Presence of Architectural Obfuscation (arXiv 2506.18053) — activations permuted / linearly transformed; layer-wise circuit structure survives, head-level attributions scatter. This is an *empirical* invariance test of MI under a symmetry-like transformation. **Closest neighbour to F10/F11**; our contribution beyond it: exact symmetry group (not obfuscation), separate permutation/scaling/rotation samples, per-metric CV classification, covariance laws, and the alignment ceilings. Novelty of the audit **downgraded from OPEN to FLAGGED-adjacent**: cite and position as the systematic, exact-group version.
* Evaluating the Robustness of Interpretability Methods (arXiv 2304.06715); Polymorphism Is Rotation (arXiv 2605.24577) — rotation of representations as a lens on MI; cite.
* Sundararajan et al. 2017 (Implementation Invariance) — TAKEN at input level, cited.

**Consequence for claims.** F5's orbit result and Z4 remain OPEN. The audit (F10/F11) is claimable as "the mechanistic extension of Implementation Invariance carried out against the exact gauge group with separate symmetry components", citing 2605.27836 and 2506.18053 as the two nearest prior results.

**Search 3 (2026-09-07, for F4a after AMENDMENTS #8): "convergent learning ... permutation matching git re-basin ... circuits across seeds"**
* Li, Yosinski, Clune, Lipson, Hopcroft, Convergent Learning (ICLR 2016 workshop, arXiv 1511.07543) — bipartite neuron matching across seeds; asks whether *representations* converge. Our F4a asks the same question of *circuits' weights* under the exact gauge group with a calibrated residual; cite as the origin of the question.
* Ainsworth, Hayase, Srinivasa, Git Re-Basin (ICLR 2023, arXiv 2209.04836); Analysis of LMC via permutation weight matching (arXiv 2402.04051); Equivariant weight-space alignment (arXiv 2310.13397); Re-basin via Sinkhorn (2212.12042) — permutation matching of whole networks for mode connectivity/merging. F4a's Hungarian step is the same tool applied to *circuit sub-implementations* (not whole nets), with the ReLU scaling gauge fixed and the functional-dimension collapse; novelty lies in the object (minimal circuits) and the question (orbit count of explanations), not the matching.
* Transformers Converge to Invariant Algorithmic Cores (arXiv 2602.22600, Feb 2026) and Certified Circuits: Stability Guarantees for Mechanistic Circuits (arXiv 2602.22968, Feb 2026) — **closest neighbours**: algorithmic invariance across seeds in transformers; circuit stability across seeds. Must be cited; F4a's separation of *algorithmic* identity (K_alg) from *weight-orbit* identity (K(τ)) is the specific contribution to position against them.
* Contravariance Theory (arXiv 2607.08561, July 2026): alignment of minimal solutions to hard tasks — relevant to why OR4/AND4 cells merge more than parity cells; cite.
**Status: FLAGGED → position as "gauge-orbit counting of enumerated circuits across seeds, calibrated by planted controls"; the re-basin and convergent-learning lines are the tools, 2602.22600 / 2602.22968 the closest results.**
