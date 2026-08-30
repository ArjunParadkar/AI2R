## Novelty check (run 2026-08-30, before write-up; WebSearch, six phrasings)

Queries: "random matrix theory Marchenko-Pastur null sparse autoencoder error residual";
"Marchenko-Pastur eigenvalue spectrum SAE reconstruction error dark matter language model";
"sparse autoencoder reconstruction error covariance eigenvalues Tracy-Widom outliers interpretability";
"\"Marchenko-Pastur\" \"sparse autoencoder\" residual stream LLM";
"SAE reconstruction residual random matrix / Marchenko spectrum structured vs noise 2025";
"sparse autoencoder error term persistent homology / random matrix null shuffle \"SAE error\"".

Closest hits:
* Engels, Smith & Tegmark 2024, *Decomposing the dark matter of sparse autoencoders*, https://arxiv.org/abs/2410.14670 — the target paper: dark matter is largely linearly predictable from x; splits it into unlearned linear / dense / nonlinear error; per-token error-norm prediction. **No RMT / MP null, no topology.**
* Li, Michaud, Baek, Engels, Sun & Tegmark 2024/2025, *The Geometry of Concepts: SAE feature structure*, https://pmc.ncbi.nlm.nih.gov/articles/PMC12025678/ (Entropy 27(4):344) — compares the eigenvalue spectrum of the **SAE feature (decoder-direction) covariance** and raw activations to the Marchenko–Pastur expectation and finds a power law instead. This is the nearest prior use of MP in the SAE literature, but it is on the *features*, not the *reconstruction error*, and uses no TW edge count or outlier fraction.
* Staats et al. 2026 (?), *Decoding Transformers Spectra: an RMT framework beyond Marchenko–Pastur*, https://www.researchgate.net/publication/397815057 ; and *Structure and Redundancy in LLMs: a spectral study via RMT*, https://arxiv.org/pdf/2602.22345 — MP baselines + bootstrap calibration on **weight matrices**, not activations or SAE residuals.
* *Pruning Deep Neural Networks via the Marchenko–Pastur distribution*, https://arxiv.org/html/2606.02608 — weights.
* *Matrix completion via residual spectral matching*, https://arxiv.org/abs/2412.10005 — residual-vs-random-matrix spectral criterion in matrix completion (same idea in a different field; cite as precedent for "residual should look random").
* *Do Sparse Autoencoders Capture Concept Manifolds?*, https://arxiv.org/pdf/2604.28119 — topology of SAE features/manifolds; not the error term.
* *How much of persistent homology is topology?* https://arxiv.org/pdf/2603.29072 — shuffled-null total persistence is a deterministic function of density; supports using a shuffle null for TC (as v1 does).
* Generic RMT denoising precedents: Aparicio et al. 2020 single-cell RMT denoising https://www.cell.com/patterns/pdfExtended/S2666-3899(20)30040-4 ; Bun/Bouchaud/Potters-style correlation-matrix cleaning https://portfoliooptimizer.io/blog/correlation-matrices-denoising-results-from-random-matrix-theory/ .

Verdict: **the RMT (MP bulk + TW edge) null applied to the SAE reconstruction error is not taken** in what we could find; MP has been applied to SAE *features* (Geometry of Concepts) and to transformer *weights*, and "does the residual look like a random matrix" is a known idea in matrix completion. So the honest framing is "a standard RMT instrument, previously applied to SAE features and to weights, applied here to the dark-matter term with kill controls" — a measurement contribution, not a new theory. Caveat: search was web-search only, no arXiv full-text search; a 2025-26 workshop paper could have been missed.
