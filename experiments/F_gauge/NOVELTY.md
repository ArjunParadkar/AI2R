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
