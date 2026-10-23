# Novelty check (Phase 0)

WhiteBox E&E, Arjun Paradkar. Written 2026-10-22. Master prompt Section 4.

## Verdict (read this first)

**The integrated definition was not found in the published literature.** No work located in this search defines understanding in a way that is all four of the following at once: derivational (concepts computed from concepts, forming a graph), measured from inside the system, predictive of generalization and of creation, and validated in both artificial networks and neural populations. The stop condition of the Phase 0 gate is therefore **not** triggered.

What does exist covers each clause separately, and two pieces of work are close enough that the paper must cite them and say how it differs:

1. **Beckmann and Queloz (2025, arXiv:2507.08017)** propose three tiers of understanding in LLMs (conceptual, state-of-the-world, principled). Their "principled understanding", in which a compact circuit replaces memorized facts, is the same idea as this programme's split between grounded and memorized behaviours. Their account is qualitative and philosophical: it has no explicit graph, no parent sets, no generality or depth metric, no creation clause and no neural data. This is the main risk to the novelty claim.
2. **Geiger et al. (JMLR 2025)** and the causal-abstraction line supply the causal-use condition and a high-level causal graph. That graph is supplied by the experimenter and tested one algorithm at a time. The line does not define understanding, generality, principles or memorization.

Two cautions follow for the paper's wording:

- **Separate the claim from Beckmann and Queloz explicitly.** The approved clause (rule 0.1.7) should be stated together with the words *quantitative* and *derivational*.
- **Do not say "validated" yet.** The "validated in both artificial networks and neural populations" clause is a claim about this programme's future results. Nothing in the prior work supplies it, and it may be stated only after K2 and B1–B5 pass.

## Method

- **Search.** Nine areas (master prompt Section 4) were searched by five parallel search passes on 2026-10-22, plus a dedicated stop-condition search for any published integrated definition. The stop-condition search phrases included "definition of understanding" mechanistic, "concept graph" interpretability generalization, "degrees of understanding" neural network, "measure of understanding" internal representations, and "natural abstraction".
- **Verification.** Every citation was checked against a primary record: the arXiv abstract page or API, Crossref DOI records, PMLR, NeurIPS, ICLR, JMLR, ACL Anthology, CVF or PMC pages, or the publisher's page. The four most load-bearing 2025–2026 preprints (Beckmann & Queloz; Queloz & Beckmann; Taşkesen; Guo et al.) were re-checked directly on arXiv.
- **Labels.**
  - **[V]** means title, authors, venue and year were confirmed.
  - **[V\*]** means the work is confirmed but one named field is UNVERIFIED.
  - **[UNVERIFIED]** means the work itself could not be confirmed. It is listed only so a reader can check it, and it is not relied on.
- **Corrections.** These were found during verification and are applied below:
  - The Khona, Chandra and Fiete Nature paper is titled "Global modules robustly emerge from local interactions and smooth gradients". "From smooth cortical gradients to discrete modules..." is only a seminar title.
  - Vector-HaSH is Nature 2025, volume 638.
  - The AlphaZero Hex paper is Lovering et al., not Forde et al.
  - Fascianelli et al. is "...reflect...", not "...correlate...".
  - ACE's arXiv ID is 1902.03129.
  - Vig et al.'s NeurIPS title differs from its arXiv title.
  - Sutter et al. is NeurIPS 2025.
  - MIB is ICML 2025.

**Counts.**

| Label | Count |
|---|---|
| [V] | 147 |
| [V\*] | 15 |
| [UNVERIFIED] | 2 |

The two UNVERIFIED entries are Chorna et al. 2025 and ConceptEvo; neither is relied on.

---

## Area 1. Concept definitions

| Work | Defines | Does not | This programme differs by |
|---|---|---|---|
| R. Wille. Restructuring lattice theory: an approach based on hierarchies of concepts. In I. Rival (ed.), *Ordered Sets*, NATO ASI Series C 83, pp. 445–470. Reidel, 1982. doi:10.1007/978-94-009-7798-3_15 [V] | A formal concept is a pair (extent, intent) closed under the Galois connection of an object–attribute relation; concepts form a complete lattice. | Has no learned state, no derivation (the lattice order is subsumption), and no causal use. | Proposition 2.17 (`theory/core.tex`, proof in `proofs/P5_fca.tex`): under nominal scaling of downstream readouts, the coarsest exact quotient is FCA's object-concept partition. Derivation and causal use are added, and they do not follow from FCA. |
| B. Ganter, R. Wille. *Formal Concept Analysis: Mathematical Foundations*. Springer, 1999. doi:10.1007/978-3-642-59830-2 [V] | The standard text: concept lattices, implications, attribute exploration. | As above. | As above. |
| P. Gärdenfors. *Conceptual Spaces: The Geometry of Thought*. MIT Press, 2000. doi:10.7551/mitpress/2076.001.0001 [V] | Properties are convex regions of quality dimensions; concepts are sets of such regions with salience weights. | A theory of cognitive geometry, not a measurement inside trained systems; no invariance test, no causal use, no derivation graph. | Concepts are certified by interventions on the system. Geometry enters only where a value space has shape (Section 2.6.4). |
| P. Gärdenfors. *The Geometry of Meaning: Semantics Based on Conceptual Spaces*. MIT Press, 2014. ISBN 978-0-262-02678-9 [V\*: DOI not found] | Extends conceptual spaces to word classes and to composition of meanings. | Same gap. | Same. |
| B. M. Lake, R. Salakhutdinov, J. B. Tenenbaum. Human-level concept learning through probabilistic program induction. *Science* 350(6266):1332–1338, 2015. doi:10.1126/science.aab3050 [V] | A concept is a generative program composed from reusable parts (Bayesian Program Learning, tested on Omniglot). | The concepts are explicit programs in a designed model, not variables found inside a trained network; there is no internal causal-use test. | Concepts are recovered from an arbitrary trained system; reuse becomes a measured generality. |
| B. M. Lake, T. D. Ullman, J. B. Tenenbaum, S. J. Gershman. Building machines that learn and think like people. *Behavioral and Brain Sciences* 40:e253, 2017. doi:10.1017/S0140525X16001837; arXiv:1604.00289 [V] | A position paper: human-like learning needs causal models, compositionality and learning-to-learn. | No formal definition and no measurement. | Formal and measured. |
| D. Endres, P. Földiák. Interpreting the neural code with Formal Concept Analysis. *NIPS 21*, 2008 [V] | FCA lattices with stimuli as objects and neurons as attributes, applied to macaque STSa recordings. | Observational only; no downstream invariance or intervention. | Interventional quotient. |
| D. M. Endres, P. Földiák, U. Priss. An application of formal concept analysis to semantic neural decoding. *Annals of Mathematics and Artificial Intelligence* 57:233–248, 2010. doi:10.1007/s10472-010-9196-8 [V] | Journal version of the above. | As above. | As above. |
| J. Hirth, T. Hanika. Conceptual views of neural networks: a framework for neuro-symbolic analysis. arXiv:2209.13517, 2022 (revised 2026) [V] | FCA lattices built from neuron activations of trained ImageNet models, used as global explanations. | Subsumption order only; no causal-use test, no minimal parents, no generality or depth. | All three conditions; generality and depth. |
| K. Park, Y. J. Choe, Y. Jiang, V. Veitch. The geometry of categorical and hierarchical concepts in large language models. *ICLR* 2025. arXiv:2406.01506 [V] | Categorical concepts are polytopes; hierarchically related concepts are orthogonal. | The hierarchy is imported from WordNet, not derived from the computation; no interventions, no generalization prediction. | The graph is derived from the computation and certified causally. |
| I. Higgins, D. Amos, D. Pfau, S. Racanière, L. Matthey, D. Rezende, A. Lerchner. Towards a definition of disentangled representations. arXiv:1812.02230, 2018 [V] | Disentanglement via symmetry subgroups acting on separate subspaces, which is close to an invariance (quotient) condition. | No causal use, no derivation, no generality. | As stated. |

## Area 2. Abstraction in brains and networks

| Work | Defines | Does not | This programme differs by |
|---|---|---|---|
| S. Bernardi, M. K. Benna, M. Rigotti, J. Munuera, S. Fusi, C. D. Salzman. The geometry of abstraction in the hippocampus and prefrontal cortex. *Cell* 183(4):954–967.e21, 2020. doi:10.1016/j.cell.2020.09.031 [V] | Cross-condition generalization performance (CCGP) and the parallelism score: a variable is abstract if a linear decoder trained on some conditions generalizes to held-out ones. Shown in monkey HPC/PFC and in networks. | Decodability by an external probe, not use by the system; one variable at a time; no derivation, no generality, no depth. | CCGP is the planned depth-one special case (Proposition 2.7.1, Phase 1). The expected statement: for a single-node graph with linear value map and observational data, depth reduces to a function of CCGP, and the causal-use term can only lower it. CCGP is a K2 baseline. |
| W. J. Johnston, S. Fusi. Abstract representations emerge naturally in neural networks trained to perform multiple tasks. *Nature Communications* 14:1040, 2023. doi:10.1038/s41467-023-36583-0 [V] | Multi-task training produces abstract, disentangled latents, scored by CCGP-style generalization. | Representational; one level; no causal use. | Multi-level graph with causal certificates. |
| L. M. Boyle, L. Posani, S. Irfan, S. A. Siegelbaum, S. Fusi. Tuned geometries of hippocampal representations meet the computational demands of social memory. *Neuron* 112(8):1358–1371.e9, 2024. doi:10.1016/j.neuron.2024.01.021 [V] | CA2 codes familiarity abstractly; the change in dimensionality predicts social-memory performance. | Geometry–behaviour correlation; no intervention on the variable. | Level-D causal-use certificate where perturbation data exist. |
| H. S. Courellis, J. Minxha, A. R. Cardenas, et al., U. Rutishauser. Abstract representations emerge in human hippocampal neurons during inference. *Nature* 632:841–849, 2024. doi:10.1038/s41586-024-07799-x [V] | CCGP-abstract coding of a latent context emerges in human hippocampal neurons when subjects can infer. | Correlational; no multi-level structure. | A natural test bed for B2 (learning is grounding), if data access is granted. |
| R. Nogueira, C. C. Rodgers, R. M. Bruno, S. Fusi. The geometry of cortical representations of touch in rodents. *Nature Neuroscience* 26:239–250, 2023. doi:10.1038/s41593-022-01237-9 [V] | Whisker contacts are coded in near-orthogonal subspaces. | Representational. | — |
| V. Fascianelli, A. Battista, F. Stefanini, S. Tsujimoto, A. Genovesio, S. Fusi. Neural representational geometries reflect behavioral differences in monkeys and recurrent neural networks. *Nature Communications* 15:6479, 2024. doi:10.1038/s41467-024-50503-w [V] | Differences in PFC geometry track strategy differences between monkeys and amount of training in RNNs. | No interventions. | — |
| N. Kriegeskorte, M. Mur, P. Bandettini. Representational similarity analysis – connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience* 2:4, 2008. doi:10.3389/neuro.06.004.2008 [V] | RSA: compare representational dissimilarity matrices across brains, models and behaviour. | Second-order similarity only; discards what is used downstream. | Use is the criterion. |
| N. Kriegeskorte, R. A. Kievit. Representational geometry: integrating cognition, computation, and the brain. *Trends in Cognitive Sciences* 17(8):401–412, 2013. doi:10.1016/j.tics.2013.06.007 [V] | Review: the distance matrix as a sufficient summary of a code. | Same. | Same. |
| N. Kriegeskorte, J. Diedrichsen. Peeling the onion of brain representations. *Annual Review of Neuroscience* 42:407–432, 2019. doi:10.1146/annurev-neuro-080317-061906 [V] | Content of a representation is what a linear readout can decode. | This decodability criterion is exactly what Condition 3 goes beyond. | "Represented, not used" is a recorded category. |
| S. Chung, L. F. Abbott. Neural population geometry: an approach for understanding biological and artificial neural networks. *Current Opinion in Neurobiology* 70:137–144, 2021. doi:10.1016/j.conb.2021.10.010; arXiv:2104.07059 [V] | Review of manifold capacity, untangling and abstraction. | Survey; no graph, no causal use. | — |
| B. Sorscher, S. Ganguli, H. Sompolinsky. Neural representational geometry underlies few-shot concept learning. *PNAS* 119(43):e2200800119, 2022. doi:10.1073/pnas.2200800119 [V\*: no arXiv ID found] | A concept is a manifold of patterns; few-shot error follows from manifold geometry, in deep nets and primate IT. | Predicts generalization from one readout layer; no derivation, no interventions. | Prediction from a multi-level derivation; a candidate K2 baseline. |
| S. Kornblith, M. Norouzi, H. Lee, G. Hinton. Similarity of neural network representations revisited. *ICML* 2019, PMLR 97:3519–3529. arXiv:1905.00414 [V] | CKA similarity between representations. | Whole-layer comparison; no variables. | — |
| T. Ito, T. Klinger, D. Schultz, J. Murray, M. Cole, M. Rigotti. Compositional generalization through abstract representations in human and artificial neural networks. *NeurIPS* 2022. arXiv:2209.07431 [V] | Abstract compositional representations shared by human fMRI and pretrained networks predict zero-shot generalization. | Representational; no derivation, no creation, no definition of understanding. | Covers networks + brains + generalization, which is why it is listed with the closest work. |

## Area 3. Implementation: causal abstraction, interchange interventions, circuits

| Work | Defines | Does not | This programme differs by |
|---|---|---|---|
| A. Geiger, H. Lu, T. Icard, C. Potts. Causal abstractions of neural networks. *NeurIPS 34*, 2021. arXiv:2106.02997 [V] | Interchange interventions test whether a network implements a given symbolic causal model. | One hypothesized algorithm with a hand-chosen alignment at a time; no global graph, generality or depth. | Source of Condition 3. Here the high-level graph is discovered (minimal parents), not supplied, and is global over a behaviour suite. |
| A. Geiger, Z. Wu, H. Lu, J. Rozner, E. Kreiss, T. Icard, N. Goodman, C. Potts. Inducing causal structure for interpretable neural networks. *ICML* 2022, PMLR 162:7324–7338. arXiv:2112.00826 [V] | Interchange intervention training imposes a causal model. | Imposes rather than discovers. | Discovery. |
| A. Geiger, Z. Wu, C. Potts, T. Icard, N. Goodman. Finding alignments between interpretable causal variables and distributed neural representations. *CLeaR* 2024, PMLR 236:160–187. arXiv:2303.02536 [V] | Distributed alignment search (DAS): a learned rotated subspace realizing a high-level variable. | The high-level model must be supplied. | DAS is one realiser proposer and a baseline (Section 6.4). |
| A. Geiger, D. Ibeling, A. Zur, M. Chaudhary, S. Chauhan, J. Huang, A. Arora, Z. Wu, N. Goodman, C. Potts, T. Icard. Causal abstraction: a theoretical foundation for mechanistic interpretability. *JMLR* 26(83):1–64, 2025. arXiv:2301.04709 [V] | General theory of when one causal model abstracts another; unifies patching, mediation, SAEs, DAS. | Not a definition of understanding; no generality, principles, depth, memorization or creation. | Builds those on top of its notion of implementation. |
| Z. Wu, A. Geiger, T. Icard, C. Potts, N. Goodman. Interpretability at scale: identifying causal mechanisms in Alpaca. *NeurIPS* 2023. arXiv:2305.08809 [V] | Boundless DAS scales DAS to 7B. | One hand-specified algorithm. | — |
| S. Beckers, J. Y. Halpern. Abstracting causal models. *AAAI* 2019. arXiv:1812.03789 [V] | Hierarchy of abstraction relations between SCMs. | No trained systems. | — |
| P. K. Rubenstein, S. Weichwald, S. Bongers, J. M. Mooij, D. Janzing, M. Grosse-Wentrup, B. Schölkopf. Causal consistency of structural equation models. *UAI* 2017, PMLR R15:151–160. arXiv:1707.00819 [V] | Exact transformations between micro and macro SEMs. | No network measurement. | — |
| K. Chalupka, P. Perona, F. Eberhardt. Visual causal feature learning. *UAI* 2015. arXiv:1412.2309 [V\*: page numbers conflict between PMLR (R13:328–337) and mlanthology (181–190)] | Causal coarsening: the coarsest partition of input space by the interventional distribution of a target. | Partitions inputs relative to one target; no internal states, no DAG, no generality. | **Closest precedent for Condition 1.** Here the partition is of internal realiser states, tested by within-fibre swaps, across a behaviour suite. |
| J. Vig, S. Gehrmann, Y. Belinkov, S. Qian, D. Nevo, Y. Singer, S. Shieber. Investigating gender bias in language models using causal mediation analysis. *NeurIPS 33*, 2020 (expanded as arXiv:2004.12265 under a different title) [V] | Neuron- and head-level mediation of one behaviour. | One behaviour; components, not concepts. | — |
| K. Meng, D. Bau, A. Andonian, Y. Belinkov. Locating and editing factual associations in GPT. *NeurIPS* 2022. arXiv:2202.05262 [V] | Causal tracing; ROME edits. | Localization for one behaviour class. | — |
| C. Olah, N. Cammarata, L. Schubert, G. Goh, M. Petrov, S. Carter. Zoom in: an introduction to circuits. *Distill*, 2020. doi:10.23915/distill.00024.001 [V] | Features and circuits as units; universality conjecture. | Qualitative, hand-traced. | — |
| K. Wang, A. Variengien, A. Conmy, B. Shlegeris, J. Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. *ICLR* 2023. arXiv:2211.00593 [V] | Path patching, faithfulness/completeness/minimality of one circuit. | One task. | — |
| A. Conmy, A. N. Mavor-Parker, A. Lynch, S. Heimersheim, A. Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. *NeurIPS* 2023. arXiv:2304.14997 [V] | ACDC prunes the computational graph per task. | Per task; nodes are components. | — |
| S. Marks, C. Rager, E. J. Michaud, Y. Belinkov, D. Bau, A. Mueller. Sparse feature circuits: discovering and editing interpretable causal graphs in language models. *ICLR* 2025. arXiv:2403.19647 [V] | Causal graphs over SAE features; SHIFT improves OOD generalization by ablating spurious features. | Task-local graphs; no quotient or minimal-parent criterion; no generality, depth or memorization verdict. | A working internal-graph → generalization link; a direct comparison point. |
| A. Mueller, A. Geiger, S. Wiegreffe, D. Arad, I. Arcuschin, A. Belfki, et al. MIB: a mechanistic interpretability benchmark. *ICML* 2025. arXiv:2504.13151 [V] | Benchmarks circuit and causal-variable localization. | Evaluates methods; no theory of understanding. | A source of estimator checks in Phase 1. |
| A. Makelov, G. Lange, N. Nanda. Is this the subspace you are looking for? An interpretability illusion for subspace activation patching. *ICLR* 2024. arXiv:2311.17030 [V\*: whether A. Geiger is a co-author of the ICLR version is unconfirmed] | Subspace patches can act through dormant parallel pathways. | A critique. | Requires the within-fibre (quotient) test and random-realiser nulls on every patch. |
| D. Sutter, J. Minder, T. Hofmann, T. Pimentel. The non-linear representation dilemma: is causal abstraction enough for mechanistic interpretability? *NeurIPS* 2025. arXiv:2507.08802 [V] | With unrestricted alignment maps, almost any algorithm can be aligned to any network. | A critique. | **Constrains this programme:** realisers must come from a restricted class (linear subspaces, dictionary features), fixed in pre-registration. Recorded as a risk in `reports/phase0.md`. |

## Area 4. Concept-based interpretability

| Work | Defines | Does not | This programme differs by |
|---|---|---|---|
| B. Kim, M. Wattenberg, J. Gilmer, C. Cai, J. Wexler, F. Viegas, R. Sayres. Interpretability beyond feature attribution: quantitative testing with concept activation vectors (TCAV). *ICML* 2018. arXiv:1711.11279 [V] | A concept is a linear direction separating user examples; class sensitivity by directional derivative. | Human-supplied, correlational. | Concepts are proposed without supervision and certified causally. |
| P. W. Koh, T. Nguyen, Y. S. Tang, S. Mussmann, E. Pierson, B. Kim, P. Liang. Concept bottleneck models. *ICML* 2020. arXiv:2007.04612 [V] | Predict annotated concepts, then the label; test-time concept interventions. | Concepts imposed; single layer. | Recovered, multi-level. |
| A. Ghorbani, J. Wexler, J. Zou, B. Kim. Towards automatic concept-based explanations (ACE). *NeurIPS 32*, 2019. arXiv:1902.03129 [V] | Segment, cluster, score with TCAV. | One layer and class at a time. | — |
| C.-K. Yeh, B. Kim, S. Arik, C.-L. Li, T. Pfister, P. Ravikumar. On completeness-aware concept-based explanations in deep neural networks. *NeurIPS 33*, 2020. arXiv:1910.07969 [V] | Completeness: concept scores as sufficient statistics for the prediction; ConceptSHAP. | One layer; no interventions or hierarchy. | Close to Condition 1 in spirit (sufficiency); differs by interventions and the graph. |
| K. Park, Y. J. Choe, V. Veitch. The linear representation hypothesis and the geometry of large language models. *ICML* 2024. arXiv:2311.03658 [V] | Formalizes linear representation via counterfactual pairs and a causal inner product. | Individual concepts; no derivation or use mass. | — |
| T. Mikolov, W. Yih, G. Zweig. Linguistic regularities in continuous space word representations. *NAACL-HLT* 2013, pp. 746–751 [V] | Relations as constant vector offsets. | Observation only. | — |
| N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, et al. Toy models of superposition. Transformer Circuits Thread, 2022. arXiv:2209.10652 [V] | Superposition of sparse features. | Representational. | — |
| T. Bricken, A. Templeton, J. Batson, B. Chen, A. Jermyn, T. Conerly, et al. Towards monosemanticity: decomposing language models with dictionary learning. Transformer Circuits Thread, 2023 [V] | SAEs recover monosemantic features. | A flat dictionary. | Dictionaries are proposers only. |
| H. Cunningham, A. Ewart, L. Riggs, R. Huben, L. Sharkey. Sparse autoencoders find highly interpretable features in language models. *ICLR* 2024. arXiv:2309.08600 [V] | SAE features beat PCA/ICA on interpretability. | Same. | Same. |
| A. Templeton, T. Conerly, J. Marcus, J. Lindsey, T. Bricken, et al. Scaling monosemanticity: extracting interpretable features from Claude 3 Sonnet. Transformer Circuits Thread, 2024 [V] | SAEs at production scale. | Flat dictionary. | — |
| T. Lieberum, S. Rajamanoharan, A. Conmy, L. Smith, N. Sonnerat, V. Varma, et al., N. Nanda. Gemma Scope: open sparse autoencoders everywhere all at once on Gemma 2. *BlackboxNLP* 2024. arXiv:2408.05147 [V] | Open JumpReLU SAEs for every layer of Gemma 2 2B and 9B. | Infrastructure. | Used as the K1 proposer on Gemma 2. |
| E. Ameisen, J. Lindsey, A. Pearce, W. Gurnee, N. L. Turner, B. Chen, C. Citro, D. Abrahams, et al. Circuit tracing: revealing computational graphs in language models. Transformer Circuits Thread, 2025 [V] | Attribution graphs over cross-layer-transcoder features, validated by perturbation. | **Per prompt**: the authors state that global circuits are left as hard. No generality, depth or memorization. | A global graph over a behaviour suite with generality and depth. Attribution graphs are a proposer and a baseline. |
| J. Lindsey, W. Gurnee, E. Ameisen, B. Chen, A. Pearce, N. L. Turner, C. Citro, D. Abrahams, et al. On the biology of a large language model. Transformer Circuits Thread, 2025 [V] | Case studies with attribution graphs (multi-hop, planning, arithmetic). | Qualitative, one prompt at a time. | — |
| J. Dunefsky, P. Chlenski, N. Nanda. Transcoders find interpretable LLM feature circuits. *NeurIPS* 2024. arXiv:2406.11944 [V] | Input-invariant (weight-based) feature attribution. | The nearest thing to a global feature graph, with no concept criteria. | — |
| T. Fel, A. Picard, L. Béthune, T. Boissin, D. Vigouroux, J. Colin, R. Cadène, T. Serre. CRAFT: concept recursive activation factorization for explainability. *CVPR* 2023. arXiv:2211.10154 [V] | Recursive NMF concepts across layers with Sobol importance. | No causal-use or quotient test. | — |
| J. Engels, E. J. Michaud, I. Liao, W. Gurnee, M. Tegmark. Not all language model features are one-dimensionally linear. *ICLR* 2025. arXiv:2405.14860 [V] | Irreducible multi-dimensional (e.g. circular) features, causally used. | Nothing about graphs. | Supports non-linear value maps φ on a linear realiser. |
| D. Bau, B. Zhou, A. Khosla, A. Oliva, A. Torralba. Network dissection: quantifying interpretability of deep visual representations. *CVPR* 2017, pp. 6541–6549. doi:10.1109/CVPR.2017.354 [V] | Per-unit alignment with labelled concepts. | Correlational. | — |

## Area 5. Compositional generalization

| Work | Defines | Does not | This programme differs by |
|---|---|---|---|
| D. Hupkes, V. Dankers, M. Mul, E. Bruni. Compositionality decomposed: how do neural networks generalise? *JAIR* 67:757–795, 2020. doi:10.1613/jair.1.11674; arXiv:1908.08351 [V] | Five tests: systematicity, productivity, substitutivity, localism, overgeneralisation. | Behavioural, against a human grammar. | Moves are defined relative to the system's own graph. |
| B. Lake, M. Baroni. Generalization without systematicity: on the compositional skills of sequence-to-sequence recurrent networks (SCAN). *ICML* 2018, PMLR 80:2873–2882. arXiv:1711.00350 [V] | SCAN splits. | Behavioural. | Its splits are a natural source of sideways inputs for K3. |
| N. Kim, T. Linzen. COGS: a compositional generalization challenge based on semantic interpretation. *EMNLP* 2020, pp. 9087–9105. arXiv:2010.05465 [V] | Structural gaps in semantic parsing. | Behavioural. | Same. |
| D. Keysers, N. Schärli, N. Scales, et al. Measuring compositional generalization: a comprehensive method on realistic data (CFQ). *ICLR* 2020. arXiv:1912.09713 [V\*: co-authors beyond the first three not read off a page] | Atom vs compound divergence; accuracy falls with compound divergence. | Defined on dataset labels. | The **data-side analogue of lower vs sideways**; here the split is computed from the system's own concept values and parent tuples. There is no diagonal case in CFQ. |
| B. M. Lake, M. Baroni. Human-like systematic generalization through a meta-learning neural network. *Nature* 623:115–121, 2023. doi:10.1038/s41586-023-06668-3 [V] | Meta-learning for compositionality. | Behavioural. | — |
| D. Hupkes, M. Giulianelli, V. Dankers, et al. A taxonomy and review of generalization research in NLP. *Nature Machine Intelligence* 5:1161–1174, 2023. doi:10.1038/s42256-023-00729-y [V\*: second and third authors from memory] | Five-axis taxonomy of generalization shifts. | Classifies shifts in the data. | Classifies inputs relative to the system. |
| J. A. Fodor, Z. W. Pylyshyn. Connectionism and cognitive architecture: a critical analysis. *Cognition* 28(1–2):3–71, 1988. doi:10.1016/0010-0277(88)90031-5 [V] | The systematicity argument. | No measurement. | — |
| T. Wiedemer, P. Mayilvahanan, M. Bethge, W. Brendel. Compositional generalization from first principles. *NeurIPS* 2023. arXiv:2307.05596 [V] | Sufficient conditions on support and architecture for generalizing to unseen combinations. | Latents are fixed in advance; sideways only. | Includes diagonal and creation; the latents are recovered. |
| M. Okawa, E. S. Lubana, R. P. Dick, H. Tanaka. Compositional abilities emerge multiplicatively: exploring diffusion models on a synthetic task. *NeurIPS* 2023. arXiv:2310.09336 [V] | Composition appears only after every atomic capability is acquired. | Behavioural composition. | Relevant to K5 and Proposition 2.5.3. |
| J. Andreas. Measuring compositionality in representation learning (TRE). *ICLR* 2019. arXiv:1902.07181 [V] | Tree reconstruction error: an internal metric of composition. | Human-given primitives and tree; no interventions. | Primitives and tree are discovered and certified. |

## Area 6. Analogy and creation

| Work | Defines | Does not | This programme differs by |
|---|---|---|---|
| D. Gentner. Structure-mapping: a theoretical framework for analogy. *Cognitive Science* 7(2):155–170, 1983. doi:10.1207/s15516709cog0702_3 [V] | Analogy maps relational structure, preferring connected systems of relations. | Symbolic representations. | — |
| B. Falkenhainer, K. D. Forbus, D. Gentner. The structure-mapping engine: algorithm and examples. *Artificial Intelligence* 41(1):1–63, 1989. doi:10.1016/0004-3702(89)90077-5 [V] | SME algorithm. | Symbolic. | — |
| D. R. Hofstadter, M. Mitchell. The Copycat project: a model of mental fluidity and analogy-making. Ch. 5 in D. R. Hofstadter & FARG, *Fluid Concepts and Creative Analogies*. Basic Books, 1995 [V] | A hand-built concept network (the Slipnet) whose concepts slip under pressure. | Concepts are not created. | — |
| M. A. Boden. *The Creative Mind: Myths and Mechanisms*, 2nd ed. Routledge, 2004. ISBN 978-0-415-31453-4 [V] | Combinational, exploratory and transformational creativity; P- vs H-creativity. | Informal. | Transformational creativity is the informal analogue of the diagonal move plus creation; here it is an operational test. |
| G. A. Wiggins. A preliminary framework for description, analysis and comparison of creative systems. *Knowledge-Based Systems* 19(7):449–458, 2006. doi:10.1016/j.knosys.2006.04.009 [V] | Formalizes Boden: conceptual space plus rules of validity, traversal and evaluation. | Abstract; not mapped to a system's variables. | Structurally the nearest formal precedent for creation. |
| G. Ritchie. Some empirical criteria for attributing creativity to a computer program. *Minds and Machines* 17(1):67–99, 2007. doi:10.1007/s11023-007-9066-2 [V] | Criteria on outputs relative to an inspiring set. | Outputs only. | Internal criterion. |
| A. Jordanous. A standardised procedure for evaluating creative systems (SPECS). *Cognitive Computation* 4(3):246–279, 2012. doi:10.1007/s12559-012-9156-1 [V] | Domain-specific creativity standards. | Human-run protocol. | — |
| S. Colton. Creativity versus the perception of creativity in computational systems. AAAI Spring Symposium, Tech. Report SS-08-03, pp. 14–20, 2008 [V] | The creative tripod. | Perception of behaviour. | — |
| G. Franceschelli, M. Musolesi. On the creativity of large language models. *AI & Society*, 2024. doi:10.1007/s00146-024-02127-3; arXiv:2304.00008 [V] | Conceptual analysis using Boden's criteria. | No internals. | — |
| T. Webb, K. J. Holyoak, H. Lu. Emergent analogical reasoning in large language models. *Nature Human Behaviour* 7(9):1526–1541, 2023. doi:10.1038/s41562-023-01659-w [V] | GPT-3 matches humans on zero-shot analogies. | Behavioural; contested. | — |
| F. Chollet. On the measure of intelligence. arXiv:1911.01547, 2019 [V] | Intelligence as skill-acquisition efficiency; ARC. | Behavioural. | "Extreme generalization" is a behavioural analogue of the diagonal move. |
| X. Lu, M. Sclar, S. Hallinan, N. Mireshghallah, J. Liu, S. Han, et al. AI as humanity's Salieri: quantifying linguistic creativity of language models via systematic attribution of machine text against web text. *ICLR* 2025. arXiv:2410.04265 [V] | The Creativity Index (text not reconstructible from web snippets). | Output vs corpus. | Internal. |
| G. Franceschelli, M. Musolesi. DeepCreativity. arXiv:2201.06118 [V] | Latent-space novelty scores. | Novelty distance, not a new causally used derived variable. | — |

**Search result.** No work was found that defines creation operationally inside a system as the appearance of a new variable that is causally used and derived from existing variables.

## Area 7. Concepts in superhuman engines (and protein models)

| Work | Defines | Does not | This programme differs by |
|---|---|---|---|
| T. McGrath, A. Kapishnikov, N. Tomašev, et al., V. Kramnik. Acquisition of chess knowledge in AlphaZero. *PNAS* 119(47):e2206625119, 2022. doi:10.1073/pnas.2206625119; arXiv:2111.09259 [V\*: middle co-authors from memory] | Linear probes for about 100 human chess concepts across layers and training. | Supervised, correlational; no derivation order. | K12's baseline: unsupervised recovery with derivation order and causal certificates. |
| L. Schut, N. Tomašev, T. McGrath, D. Hassabis, U. Paquet, B. Kim. Bridging the human–AI knowledge gap through concept discovery and transfer in AlphaZero. *PNAS* 122, 2025. doi:10.1073/pnas.2406675122; arXiv:2310.16410 [V\*: volume/issue from memory] | Discovers latent concepts absent from human play, filtered for teachability and novelty; grandmasters learn them. | Novelty is judged against human data, not the model's earlier graph; no derivation graph; the causal test is human learning. | Closest internal precedent for "concepts no human named"; K12's discovery list is certified by Conditions 1–3. |
| C. Lovering, J. Z. Forde, G. Konidaris, E. Pavlick, M. L. Littman. Evaluation beyond task performance: analyzing concepts in AlphaZero in Hex. *NeurIPS* 2022. arXiv:2211.14673 [V] | Probes and behavioural tests for Hex concepts. | Supervised. | — |
| E. Jenner, S. Kapur, V. Georgiev, C. Allen, S. Emmons, S. Russell. Evidence of learned look-ahead in a chess-playing neural network. *NeurIPS* 2024. arXiv:2406.00877 [V] | Leela represents future moves (patching, probes). | One hypothesized mechanism. | — |
| A. Karvonen. Emergent world models and latent variable estimation in chess-playing language models. arXiv:2403.15498, 2024 [V\*: COLM 2024 venue from the arXiv comment only] | Board state and skill probed in a chess GPT. | Supervised probes. | — |
| K. Li, A. K. Hopkins, D. Bau, F. Viégas, H. Pfister, M. Wattenberg. Emergent world representations: exploring a sequence model trained on a synthetic task. *ICLR* 2023. arXiv:2210.13382 [V] | Othello-GPT board state, with interventions. | The variable is chosen by humans. | — |
| N. Nanda, A. Lee, M. Wattenberg. Emergent linear representations in world models of self-supervised sequence models. *BlackboxNLP* 2023, pp. 16–30. doi:10.18653/v1/2023.blackboxnlp-1.2 [V] | Linear "mine vs theirs" board representation. | Same. | — |
| E. Simon, J. Zou. InterPLM: discovering interpretable features in protein language models via sparse autoencoders. arXiv:2412.12101, 2024 [V\*: no journal version found; cite the preprint] | SAE features on ESM-2 matched to biological annotations. | No derivation or creation criterion. | K12 proposer for protein models. |
| E. Adams, L. Bai, M. Lee, Y. Yu, M. AlQuraishi. From mechanistic interpretability to mechanistic biology: training, evaluating, and interpreting sparse autoencoders on protein language models. *ICML* 2025, PMLR 267:460–476 [V] | Generic and family-specific SAE features. | Same. | Same. |

## Area 8. Brain structure and reuse

| Work | Defines | Does not | This programme differs by |
|---|---|---|---|
| J. C. R. Whittington, T. H. Muller, S. Mark, G. Chen, C. Barry, N. Burgess, T. E. J. Behrens. The Tolman-Eichenbaum machine: unifying space and relational memory through generalization in the hippocampal formation. *Cell* 183:1249–1263.e23, 2020. doi:10.1016/j.cell.2020.10.024 [V] | Factorizes structure (entorhinal) from content (hippocampal); reproduces grid and place phenomena. | Reuse is built in, not scored as generality; no general concept definition. | Generality is measured, and the scaffold should come out as the top-generality layer (Section 6.5 anchor). |
| J. C. R. Whittington, J. Warren, T. E. J. Behrens. Relating transformers to models and neural representations of the hippocampal formation. *ICLR* 2022. arXiv:2112.04035 [V] | TEM-t. | One model class. | — |
| T. E. J. Behrens, T. H. Muller, J. C. R. Whittington, S. Mark, A. B. Baram, K. L. Stachenfeld, Z. Kurth-Nelson. What is a cognitive map? Organizing knowledge for flexible behavior. *Neuron* 100:490–509, 2018. doi:10.1016/j.neuron.2018.10.002 [V] | Review: structure factored from content for reuse. | Conceptual. | — |
| S. Chandra, S. Sharma, R. Chaudhuri, I. Fiete. Episodic and associative memory from spatial scaffolds in the hippocampus. *Nature* 638:739–751, 2025. doi:10.1038/s41586-024-08392-y [V] | Vector-HaSH: grid modules as a fixed scaffold with heteroassociated content. | The scaffold is designed in, not shown to be the most-reused node. | Brain anchor: the pipeline run blind must recover the scaffold as the top-generality layer. |
| S. Sharma, S. Chandra, I. Fiete. Content addressable memory without catastrophic forgetting by heteroassociation with a fixed scaffold (MESH). *ICML* 2022, PMLR 162:19658–19682. arXiv:2202.00159 [V] | A fixed scaffold separated from content storage. | A memory architecture. | Planted system V4 (Phase 1). |
| M. Khona, S. Chandra, I. Fiete. Global modules robustly emerge from local interactions and smooth gradients. *Nature* 640:155–164, 2025. doi:10.1038/s41586-024-08541-3 [V] | Discrete grid modules from smooth gradients plus local interactions ("peak selection"). | Modules from developmental dynamics, not reuse under wiring cost; no generality measure. | Proposition 2.5.4 is a different mechanism (reuse under cost); B4 must compare the two predictions rather than assume ours. |
| M. Klukas, M. Lewis, I. Fiete. Efficient and flexible representation of higher-dimensional cognitive variables with grid cells. *PLoS Computational Biology* 16:e1007796, 2020. doi:10.1371/journal.pcbi.1007796 [V] | Grid modules can encode higher-dimensional variables. | Reuse is argued, not measured. | B1 measures it. |
| A. O. Constantinescu, J. X. O'Reilly, T. E. J. Behrens. Organizing conceptual knowledge in humans with a gridlike code. *Science* 352:1464–1468, 2016. doi:10.1126/science.aaf0941 [V] | A hexagonal fMRI signal in an abstract "bird" space. | Correlational. | — |
| R. J. Gardner, E. Hermansen, M. Pachitariu, Y. Burak, N. A. Baas, B. A. Dunn, M.-B. Moser, E. I. Moser. Toroidal topology of population activity in grid cells. *Nature* 602:123–128, 2022. doi:10.1038/s41586-021-04268-7 [V] | Single-module activity lies on a torus. | Structure, not use. | B1 data source; the torus is the value space of a module node. |
| H. Stensola, T. Stensola, T. Solstad, K. Frøland, M.-B. Moser, E. I. Moser. The entorhinal grid map is discretized. *Nature* 492:72–78, 2012. doi:10.1038/nature11649 [V] | Discrete grid modules. | Empirical only. | — |
| I. R. Fiete, Y. Burak, T. Brookings. What grid cells convey about rat location. *J. Neuroscience* 28:6858–6871, 2008. doi:10.1523/JNEUROSCI.5684-07.2008 [V] | Grid code as a residue number system. | Capacity argument. | — |
| S. Sreenivasan, I. Fiete. Grid cells generate an analog error-correcting code for singularly precise neural computation. *Nature Neuroscience* 14:1330–1337, 2011. doi:10.1038/nn.2901 [V] | Error-correcting grid code. | Information-theoretic. | — |
| I. Kanitscheider, I. Fiete. Training recurrent networks to generate hypotheses about how the brain solves hard navigation problems. *NeurIPS 30*, 2017. arXiv:1609.09059 [V\*: pages] | RNNs develop place-like remapping units. | One task family. | — |
| B. Sorscher, G. C. Mel, S. Ganguli, S. A. Ocko. A unified theory for the origin of grid cells through the lens of pattern formation. *NeurIPS 32*, 2019 [V]; and B. Sorscher, G. C. Mel, S. A. Ocko, L. M. Giocomo, S. Ganguli. A unified theory for the computational and mechanistic origins of grid cells. *Neuron* 111:121–137.e13, 2023. doi:10.1016/j.neuron.2022.10.003 [V] | Grids from pattern formation in path integrators. | One task. | — |
| R. Schaeffer, M. Khona, I. Fiete. No free lunch from deep learning in neuroscience: a case study through models of the entorhinal-hippocampal circuit. *NeurIPS 35*:16052–16067, 2022 [V] | Grid emergence depends on hand-picked choices. | A critique. | A warning for any "emergence" claim in K15/B4; nulls must include those choices. |
| J. Clune, J.-B. Mouret, H. Lipson. The evolutionary origins of modularity. *Proc. R. Soc. B* 280:20122863, 2013. doi:10.1098/rspb.2012.2863 [V] | Connection cost alone yields modularity. | Network-level, no node scores. | Proposition 2.5.4 adds reuse as the objective and scores nodes. |
| N. Kashtan, U. Alon. Spontaneous evolution of modularity and network motifs. *PNAS* 102:13773–13778, 2005. doi:10.1073/pnas.0503610102 [V] | Modularly varying goals yield modular networks. | Reuse of subgoals is not a measured quantity. | Closest precedent for "modularity from reuse". |
| J. Achterberg, D. Akarca, D. J. Strouse, J. Duncan, D. E. Astle. Spatially embedded recurrent neural networks reveal widespread links between structural and functional neuroscience findings. *Nature Machine Intelligence* 5:1369–1381, 2023. doi:10.1038/s42256-023-00748-9 [V] | A wiring cost in space gives brain-like modular organization. | One task. | — |
| J. L. McClelland, B. L. McNaughton, R. C. O'Reilly. Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review* 102:419–457, 1995. doi:10.1037/0033-295X.102.3.419 [V] | Complementary learning systems. | No variable-level account. | B5 states consolidation as depth migration, which is measurable. |
| D. Kumaran, D. Hassabis, J. L. McClelland. What learning systems do intelligent agents need? Complementary learning systems theory updated. *Trends in Cognitive Sciences* 20:512–534, 2016. doi:10.1016/j.tics.2016.05.004 [V] | Update with replay and schemas. | Review. | — |
| L. R. Squire, P. Alvarez. Retrograde amnesia and memory consolidation: a neurobiological perspective. *Current Opinion in Neurobiology* 5:169–177, 1995. doi:10.1016/0959-4388(95)80023-9 [V]; P. Alvarez, L. R. Squire. Memory consolidation and the medial temporal lobe: a simple network model. *PNAS* 91:7041–7045, 1994. doi:10.1073/pnas.91.15.7041 [V] | Standard consolidation account. | Systems level. | — |
| P. W. Frankland, B. Bontempi. The organization of recent and remote memories. *Nature Reviews Neuroscience* 6:119–130, 2005. doi:10.1038/nrn1607 [V] | Review of trace migration. | Descriptive. | — |
| D. Tse, R. F. Langston, M. Kakeyama, I. Bethus, P. A. Spooner, E. R. Wood, M. P. Witter, R. G. M. Morris. Schemas and memory consolidation. *Science* 316:76–82, 2007. doi:10.1126/science.1135935 [V] | Schemas speed consolidation of consistent associations. | Behavioural/lesion. | — |
| G. R. Yang, M. R. Joglekar, H. F. Song, W. T. Newsome, X.-J. Wang. Task representations in neural networks trained to perform many cognitive tasks. *Nature Neuroscience* 22:297–306, 2019. doi:10.1038/s41593-018-0310-2 [V] | Task-variance clusters shared across tasks; selective lesions. | Variance, not certified routing; no concept definition. | The cluster task-variance profile is an informal generality; this programme defines it with causal certificates. |
| L. N. Driscoll, K. Shenoy, D. Sussillo. Flexible multitask computation in recurrent networks utilizes shared dynamical motifs. *Nature Neuroscience* 27:1349–1363, 2024. doi:10.1038/s41593-024-01668-6 [V] | Dynamical motifs reused across tasks; lesioning a motif impairs every task using it. | Motifs are not variables passing quotient/derivation; no generality metric, no understanding. | **Nearest precedent for generality**: "a principle is a node many behaviours route through". The paper must cite it as such. |

## Area 9. Understanding as compression; generalization bounds

| Work | Defines | Does not | This programme differs by |
|---|---|---|---|
| J. Rissanen. Modeling by shortest data description. *Automatica* 14(5):465–471, 1978. doi:10.1016/0005-1098(78)90005-5 [V] | MDL. | Compression of data by a model, not of internal state. | — |
| P. D. Grünwald. *The Minimum Description Length Principle*. MIT Press, 2007. ISBN 978-0-262-07281-6 [V] | MDL textbook. | Model selection. | — |
| N. Tishby, F. C. Pereira, W. Bialek. The information bottleneck method. *Proc. 37th Allerton Conf.*, pp. 368–377, 1999. arXiv:physics/0004057 [V] | IB: minimize I(X;T) − βI(T;Y). | A single bottleneck variable; no graph, no causal test. | Condition 1 is an interventional, per-node analogue; H(C) small plays the role of I(X;T) small. |
| N. Tishby, N. Zaslavsky. Deep learning and the information bottleneck principle. *IEEE ITW* 2015. doi:10.1109/ITW.2015.7133169; arXiv:1503.02406 [V] | Layers in the information plane. | Layer-level. | — |
| R. Shwartz-Ziv, N. Tishby. Opening the black box of deep neural networks via information. arXiv:1703.00810, 2017 [V] | Fitting-then-compression phases. | Disputed; whole layers. | — |
| A. M. Saxe, Y. Bansal, J. Dapello, M. Advani, A. Kolchinsky, B. D. Tracey, D. D. Cox. On the information bottleneck theory of deep learning. *ICLR* 2018; *J. Stat. Mech.* 2019:124020. doi:10.1088/1742-5468/ab3985 [V] | Compression depends on the nonlinearity and is not causally tied to generalization. | — | A warning: compression alone is not claimed to predict anything here. |
| A. Achille, S. Soatto. Emergence of invariance and disentanglement in deep representations. *JMLR* 19(50):1–34, 2018. arXiv:1706.01350 [V] | Weight information bounds invariance and disentanglement. | Invariance without derivation or interventions. | — |
| D. A. McAllester. Some PAC-Bayesian theorems. *Machine Learning* 37:355–363, 1999. doi:10.1023/A:1007618624809 [V]; PAC-Bayesian model averaging. *COLT* 1999, pp. 164–170. doi:10.1145/307400.307435 [V] | PAC-Bayes bounds. | Whole hypotheses. | Section 2.6.5 seeks a bound in depth with a measurable constant (not attempted in Phase 0). |
| G. K. Dziugaite, D. M. Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. *UAI* 2017. arXiv:1703.11008 [V] | Non-vacuous PAC-Bayes bounds. | Flatness, not structure. | — |
| Y. Jiang, B. Neyshabur, H. Mobahi, D. Krishnan, S. Bengio. Fantastic generalization measures and where to find them. *ICLR* 2020. arXiv:1912.02178 [V] | Large-scale test of 40+ complexity measures. | Weight and sharpness measures. | Defines the comparison protocol for K2's sharpness baselines. |
| L. Blier, Y. Ollivier. The description length of deep learning models. *NeurIPS* 2018, pp. 2216–2226. arXiv:1802.07044 [V] | Prequential description length of networks. | Whole model. | — |
| G. Delétang, A. Ruoss, P.-A. Duquenne, E. Catt, T. Genewein, C. Mattern, J. Grau-Moya, L. K. Wenliang, M. Aitchison, L. Orseau, M. Hutter, J. Veness. Language modeling is compression. *ICLR* 2024. arXiv:2309.10668 [V] | Prediction is equivalent to compression. | Output-level. | — |
| S. Lotfi, M. Finzi, S. Kapoor, A. Potapczynski, M. Goldblum, A. G. Wilson. PAC-Bayes compression bounds so tight that they can explain generalization. *NeurIPS* 2022. arXiv:2211.13609 [V] | Tight Occam-style bounds by weight compression. | Parameters, not concepts. | — |
| M. Hutter. *Universal Artificial Intelligence*. Springer, 2005. doi:10.1007/b138233 [V]; the Hutter Prize (hutter1.net/prize) [V\*: founding year unconfirmed] | AIXI; intelligence as compression. | Behavioural, uncomputable. | — |
| D. A. Wilkenfeld. Understanding as compression. *Philosophical Studies* 176:2807–2831, 2019. doi:10.1007/s11098-018-1152-1 [V] | Understanding as a representation/process pair that generates useful information at short description length. | Philosophical; nothing inside a system; no test. | Closest philosophical match for the compression reading. |
| M. Queloz, P. Beckmann. Why we care about understanding: competence through predictive compression. arXiv:2609.04962, 2026 [V] | Understanding as a model of relational structure supporting prediction; compression as its by-product. | Theoretical; no measurement. | — |
| N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, P. T. P. Tang. On large-batch training for deep learning: generalization gap and sharp minima. *ICLR* 2017. arXiv:1609.04836 [V] | Sharp minima and the generalization gap. | Loss geometry. | K2 baseline (sharpness). |
| P. L. Bartlett, D. J. Foster, M. Telgarsky. Spectrally-normalized margin bounds for neural networks. *NeurIPS* 2017, pp. 6240–6249. arXiv:1706.08498 [V] | Margin bound with spectral norms. | Norm-based. | K2 baseline (weight norm). |

## Stop-condition search: definitions of understanding

| Work | Clauses covered (D derivational / I inside / G,C predicts generalization, creation / N+B networks and brains) |
|---|---|
| P. Beckmann, M. Queloz. Mechanistic indicators of understanding in large language models. arXiv:2507.08017, v1 7 July 2025, v5 25 Feb 2026 (preprint) [V, checked directly] | D partial (tiers built on connected features; no explicit graph or parent sets). I qualitative (circuit evidence, no metric). G argued, not measured. C no. N+B no. **Closest conceptual precedent.** |
| A. Geiger, D. Ibeling, A. Zur, et al. Causal abstraction (JMLR 2025; above) [V] | I yes. D partial (a supplied high-level graph). G no. C no. Networks only. Not a definition of understanding. |
| A. Geiger, J. Harding, T. Icard. How causal abstraction underpins computational explanation. arXiv:2508.11214, 2025 (revised 2026) [V] | I yes. G framed, not tested. D, C and empirical N+B no. |
| E. Ameisen et al.; J. Lindsey et al. (2025; above) [V] | D yes in mechanics (per-prompt graphs). I yes. No definition of understanding, no generality, no G/C test; brains by analogy only. |
| B. Taşkesen. A mathematical theory of understanding. arXiv:2603.19349, 19 Mar 2026 [V, checked directly] | D yes, abstractly (a prerequisite structure over concepts in a model learner). I no (no measurement in any system). G/C no. N+B no. |
| L. Guo, H. Chen, Y. Chen, Y. Bi, S. Yu. A neural network for modeling human concept formation, understanding and communication. arXiv:2601.02010, 5 Jan 2026 [V\*: the arXiv page lists no journal reference; a *Nature Computational Science* 2026 version reported in press coverage is UNVERIFIED] | N+B partial (a designed architecture aligned to ventral occipitotemporal fMRI). I only within its own concept module. G as cross-network transfer. D no. C no. |
| T. Ito et al. 2022 (Area 2) [V] | N+B yes, G yes, I representational, D no, C no. |
| S. Bernardi et al. 2020; W. J. Johnston, S. Fusi 2023 (Area 2) [V] | I and G yes (decoders); N+B yes for Bernardi; D and C no. |
| K. Leyton-Brown, Y. Shoham. Understanding understanding: a pragmatic framework motivated by large language models. arXiv:2406.10937, 2024 [V] | Behavioural (question sampling with confidence bounds). Fails I, D, N+B. |
| M. Mitchell, D. C. Krakauer. The debate over understanding in AI's large language models. *PNAS* 120(13):e2215907120, 2023. arXiv:2210.13966 [V\*: DOI 10.1073/pnas.2215907120 inferred from the article number] | Survey; no definition. |
| N. Nanda, L. Chan, T. Lieberum, J. Smith, J. Steinhardt. Progress measures for grokking via mechanistic interpretability. *ICLR* 2023. arXiv:2301.05217 [V] | I and G yes in one task (memorization → circuit → cleanup); no general definition, D, C or brains. A K14 reference point. |
| A. Power, Y. Burda, H. Edwards, I. Babuschkin, V. Misra. Grokking: generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177, 2022 [V] | The phenomenon. |
| Z. Liu, O. Kitouni, N. Nolte, E. J. Michaud, M. Tegmark, M. Williams. Towards understanding grokking: an effective theory of representation learning. *NeurIPS* 2022. arXiv:2205.10343 [V] | I and G yes (structured representations); D, C, brains no. |
| F. E. Rosas, P. A. M. Mediano, H. J. Jensen, A. K. Seth, A. B. Barrett, R. L. Carhart-Harris, D. Bor. Reconciling emergences: an information-theoretic approach to identify causal emergence in multivariate data. *PLoS Computational Biology* 16(12):e1008289, 2020. doi:10.1371/journal.pcbi.1008289 [V] | Macro-variables with causal emergence, applied to ECoG; no concept graph, no understanding. |
| Y. Bengio. The consciousness prior. arXiv:1709.08568, 2017 [V]; A. Goyal, Y. Bengio. Inductive biases for deep learning of higher-level cognition. *Proc. R. Soc. A* 478:20210068, 2022. doi:10.1098/rspa.2021.0068 [V] | Proposals for sparse causally related high-level variables as inductive biases; not a definition or a measurement. |
| M. Huh, B. Cheung, T. Wang, P. Isola. The platonic representation hypothesis. *ICML* 2024. arXiv:2405.07987 [V] | Representational convergence; nothing on understanding. |
| T. J. Sejnowski. Large language models and the reverse Turing test. *Neural Computation* 35(3):309–342, 2023. doi:10.1162/neco_a_01563 [V] | Behavioural essay. |
| J. Wentworth, natural abstraction hypothesis (LessWrong, 2021) [V as blog posts, not peer reviewed]; L. Chan, L. Lang, E. Jenner, Natural abstractions: key claims, theorems, and critiques (LessWrong/AF, 2023) [V as a post] | Low-dimensional abstractions carrying redundant information; close in spirit to Condition 1; no derivation, measure, or brains. |
| S. R. Grimm. Is understanding a species of knowledge? *BJPS* 57:515–535, 2006. doi:10.1093/bjps/axl015 [V]; J. L. Kvanvig. *The Value of Knowledge and the Pursuit of Understanding*. CUP, 2003 [V]; H. W. de Regt. *Understanding Scientific Understanding*. OUP, 2017 [V\*: DOI]; K. Khalifa. *Understanding, Explanation, and Scientific Knowledge*. CUP, 2017 [V] | Philosophy of understanding (including understanding as coming in degrees); no internal measurement. |
| Chorna et al. Concept-based mechanistic interpretability using structured knowledge graphs. arXiv:2507.05810, 2025 [UNVERIFIED: authors and content not opened] | Not relied on. |
| ConceptEvo, arXiv:2203.16475 [UNVERIFIED: a lead only] | Reported to track concept emergence over training; not relied on. |
| (Also checked) Lake et al. 2017 and Chollet 2019 (above) [V] | Behavioural or programmatic; no internal definition. |

---

## Statement of exactly which clauses of the definition are new

The table below decomposes the definition (master prompt Sections 2.2–2.4; `theory/core.tex`) into clauses. Each clause is marked **prior** (published, with the source) or **new**, where new means not found in this search. A clause marked new is a claim about the literature, not about the world. It becomes a claim in the paper only after the pre-registered tests that support it pass.

| # | Clause | Status | Basis |
|---|---|---|---|
| 1 | Causal use certified by interchange interventions on an aligned realiser | **Prior** | Geiger, Lu, Icard, Potts 2021; Geiger et al. 2024 (DAS); Geiger et al. JMLR 2025 |
| 2 | A quotient (coarsening) defined by interventional equivalence | **Prior** for input spaces relative to one target | Chalupka, Perona, Eberhardt 2015; Rubenstein et al. 2017; Beckers & Halpern 2019 |
| 2′ | The quotient condition on *internal realiser states*, tested by within-fibre swaps across a behaviour suite, with compression H(C)/H(X) and random-partition nulls; its exact relation to FCA (Proposition 2.17) | **New** | Not found. Yeh et al. 2020 (completeness) and Hirth & Hanika 2022 (FCA on activations) are the nearest; neither is interventional |
| 3 | Internal causal graphs over features for a task or prompt | **Prior** | Marks et al. 2025; Ameisen et al. 2025; Conmy et al. 2023 |
| 3′ | A single *global* concept DAG over a behaviour suite whose nodes must pass all three conditions; edges from minimal sufficient parent sets; well-definedness up to a stated equivalence (partition identity, reparametrisation, merging) with uniqueness under positivity (Propositions 2.5–2.14) | **New** | Not found. Attribution graphs are explicitly per prompt; causal abstraction takes the graph as given |
| 4 | Reuse of internal structure across tasks (motifs, clusters) | **Prior** as an observation | Driscoll, Shenoy, Sussillo 2024; Yang et al. 2019; TEM; Vector-HaSH |
| 4′ | *Generality* as the behaviour-weighted mass of inputs on whose active derivation a certified node lies, null-corrected; *principles* as thresholded generality | **New** | Not found as a defined quantity |
| 5 | Understanding as "principled" circuits replacing memorized facts | **Prior** qualitatively | Beckmann & Queloz 2025; Nanda et al. 2023 (grokking progress measures) |
| 5′ | *Depth of understanding* as grounding of a behaviour's derivation in principles: thresholded (vertex cut), structural bottleneck λ, interventional mediation share γ_τ, behaviour-level scores; *structural memorization* computed without held-out data (Definitions 3.4–3.7, Propositions 3.5 and 3.8) | **New** | Not found |
| 6 | Lower vs sideways inputs as atom vs compound novelty | **Prior** on the data side | Keysers et al. 2020; Lake & Baroni 2018; Kim & Linzen 2020 |
| 6′ | Lower/sideways/diagonal defined *relative to the system's own concept graph* (value and parent-tuple ranges, coverage), with the sideways prediction tied to interchange on unseen combinations (Proposition 2.8) | **New** | Not found |
| 7 | Transformational creativity; output-based creativity metrics | **Prior** informally or behaviourally | Boden 2004; Wiggins 2006; Ritchie 2007; Lu et al. 2025 |
| 7′ | *Creation* as the appearance under exposure of a new node that passes all three conditions and is derived from existing nodes; creation rate and reach; four explicit false-creation categories | **New** | Not found; Schut et al. 2025 discover concepts new relative to humans, but not new relative to the model's own prior graph, and without derivation |
| 8 | Abstraction measured by the same instrument in brains and networks | **Prior** for representational abstraction | Bernardi et al. 2020; Ito et al. 2022 |
| 8′ | The same derivational definition and pipeline applied to neural populations at a stated access level (D) | **New as a definition**; **untested** | The "validated" half requires B1–B5 |
| 9 | The integration of 2′, 3′, 4′, 5′, 6′, 7′ into one definition of understanding | **New** | The stop-condition search found no published integration |

**What may be said, and when.** The approved clause of rule 0.1.7 reads: "the first definition of understanding that is derivational, measured from inside the system, predictive of generalization and creation, and validated in both artificial networks and neural populations". It decomposes as follows.

- *Derivational* rests on clause 3′.
- *Measured from inside the system* rests on 2′, 3′, 4′ and 5′.
- *Predictive of generalization* needs K2 to pass.
- *Predictive of creation* needs K13 to pass.
- *Validated in networks and neural populations* needs K-tests and at least one of B1–B5 to pass.

Today only the first two words are supported, and only as a novelty statement about the literature. In the paper, the claim must be accompanied by citations to Beckmann & Queloz (2025), Geiger et al. (2025), Chalupka et al. (2015), Driscoll et al. (2024) and Bernardi et al. (2020), each named for the clause it already covers.

**What is not new and must not be presented as new:**

- interchange interventions;
- sparse dictionaries and attribution graphs;
- causal coarsening of inputs;
- CCGP;
- the idea that understanding is compression;
- the qualitative split between principled and memorized understanding;
- the observation that brains and multitask networks reuse structure;
- Boden's taxonomy of creativity.
