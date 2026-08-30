# whitebox-topo

Topological interpretability of neural-network training: persistent-homology
instruments (structural velocity, topological complexity, Betti numbers,
feature persistence, BiasTop) and two **calibrated** signals built on them —
a memorisation score and a data-manifold correspondence measure — each
shipped with its ROC / error characteristics and an explicit list of where it
fails (`VALIDATION.md`).

The library grew out of the WHITEBOX paper (Paradkar 2026). Several of the
paper's claims did **not** survive the validation battery built here; see
`AUDIT.md` and `benchmarks/PHASE*.md`. What ships is what replicated.

## Install (10 minutes to a first signal)

```bash
git clone https://github.com/ArjunParadkar/AI2R whitebox-lib && cd whitebox-lib
python -m venv .venv && . .venv/bin/activate          # or: uv venv --python 3.12 .venv
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or cpu
pip install -e ".[dev]"
pytest -q                                             # ~30 s
```

Dependencies: numpy, scipy, scikit-learn, ripser, persim; giotto-ph
(recommended, multi-threaded exact PH); torch only for model hooks;
`[legibility]` extra (networkx, python-igraph, leidenalg) for `whitebox.legibility`.

## Quickstart 1 — is my classifier memorising?

```python
import torch, whitebox as wb

model = ...                      # a trained conv classifier (eval mode)
probe = [(x, None) for x, _ in probe_loader]   # ~500 *training* images, no augmentation

sig = wb.memorization_signal(model, probe, layer="avgpool",   # penultimate pooled features
                             family="cnn_classifier")
print(sig)
# MemorizationSignal(score=0.099 CI95=[0.000,0.375] threshold=0.63 -> generalizing;
#   within the battery's generalising range [0.05, 0.60]; tc_ratio=0.49 pe1=5.42 n=500)
# (real output for a ResNet-18 trained on full CIFAR-10 with augmentation; an 80 %-label-noise
#  ResNet gives score=0.760 CI95=[0.615,0.899], "within the battery's memorising range")
```

`score` is a calibrated ranking on [0, 1] (memorising ↔ generalising) fitted
on a label-noise battery, with a probe-subsampling interval; `verdict`
compares the interval to the battery's separating threshold — CNN intervals are
wide (±0.15–0.25), so expect `uncertain` unless the score is clearly on one side;
`battery_position` tells you which range the point estimate falls in. **`family` is
required**: the sign of the signature flips between CNN classifiers and
transformer LMs (`benchmarks/PHASE2.md`), so there is no auto-detect and no
support for other families until you calibrate them
(`benchmarks/phase2_battery_*.py` + `phase2_calibrate.py`).

For a causal LM:

```python
from whitebox.activations import hf_hidden_states
acts = hf_hidden_states(model, input_ids, attention_mask, layers=[12])   # mask-mean final hidden state
sig = wb.memorization_signal(activations=acts[12], family="lm_transformer")
```

## Quickstart 2 — does the representation preserve the data's topology?

```python
res = wb.representation_correspondence(model, probe, layers=["layer3", "avgpool"])
for layer, c in res.items():
    print(c)
# Correspondence(layer=avgpool, score=0.08 CI95=[0.05,0.11], RTD=44.8 null=47.1, W2(H1)=2.9, n=500)
```

`score = 1 − RTD/RTD_null`: 1 means the layer is topologically identical to
the input cloud (RTD, Barannikov et al. ICML 2022, reimplemented exactly);
0 means no better than a random re-pairing of the same activations. On
synthetic manifolds of known topology this works as advertised; on real
images the input cloud is pixel space and **good classifiers score near 0**
— they discard input topology on purpose. Read `benchmarks/PHASE3.md`
before using this to judge a model.

## v2 modules: gauge invariance, legibility, dark matter, and gluing (Phase 6 / Stage 2B)

**Is my metric a statement about the function, or about one arbitrary
representative of it?** For a ReLU MLP the network's function is unchanged by
hidden-unit permutations and per-unit positive rescalings. `check_invariance`
samples that orbit and reports how much any metric you write moves:

```python
from whitebox.invariants import check_invariance, standardized_tc, sorted_ablation
rep = check_invariance(lambda m: my_metric(m, X), model, n_reps=100)   # numpy .Ws/.bs MLP or nn.Sequential(Linear/ReLU)
print(rep)
# InvarianceReport(my_metric, group=combined, n_reps=100: CV=1.4, agreement=0.01, verdict=not_invariant,
#   broken_by=scale, param-distance median=3.1)
```

`whitebox.invariants` also ships the quantities E2 measured to be exactly
invariant (logits, input Jacobian, W-product, **TC of the per-unit standardised
cloud**, sorted grad×act / ablation / gate multisets), each re-checked by its
own orbit unit test. What is *not* there matters more: TC, β and persistent
entropy of a **raw** ReLU activation cloud are broken by per-unit scaling (orbit
CV 1.4–3.0, larger than their variation between independently trained nets),
and so is every "neuron i does X" reading. Scope: ReLU MLPs; other
architectures have other groups (`VALIDATION.md`, `benchmarks/PHASE6.md`).

**How legible is this model's attribution graph?** `legibility_profile`
measures modularity (Leiden Q with a density sweep and a shuffled-weight null),
a min-fill treewidth bound, effective sparsity, and the outcome of a fixed
attribution-patching + greedy-prune discovery pipeline (guarded by an accuracy
floor):

```python
from whitebox.legibility import legibility_profile, legibility_score
prof = legibility_profile(model, (x, targets), discovery_batches=[(x1, t1), (x2, t2)])
prof["Q10"], prof["Q10_excess"], prof["tw10_norm"], prof["sparsity"], prof["S"], prof["jaccard"]
legibility_score(model, (x, targets))   # raises NotCalibratedError — deliberately
```

There is **no** `legibility_score` that returns a number: on E4's 87-model
ensemble no structural measure predicted circuit-discovery success within an
architecture (ρ(Q₁₀, S) = −0.37 pooled is an architecture offset with the wrong
sign; within-architecture CIs all cross 0). The profile is a measurement.
Needs the `legibility` extra (`networkx`, `python-igraph`, `leidenalg`, `torch`).

**Is an SAE's reconstruction error ("dark matter") structured, or isotropic
noise?** `residual_structure` reads the error cloud e = x − recon(x) with two
independent nulls — a Marchenko–Pastur/Tracy–Widom spectral null (with a
matched-diagonal-Gaussian kill and a feature-shuffle validity check) and the
v1 topology-with-shuffle-null instrument (with a covariance-matched Gaussian
control):

```python
from whitebox.darkmatter import residual_structure
rep = residual_structure(errors)                # E = x - SAE(x), (n_tokens, d)
rep = residual_structure(acts=x, recon=sae(x))  # or pass the pair directly
rep.n_outliers, rep.spectral_test_valid          # Test 1: MP/TW spectral null
rep.tc_ratio, rep.tc_ratio_ci, rep.gaussian_control_tc_ratio   # Test 2: topology null
rep.verdict   # "neither" / "spectrally_structured" / "topologically_structured" / "both"
```

`verdict` is **not** a claim that dark matter is structured — it names which
instrument(s) cleared their own pre-registered bar. On the one battery run so
far (E6, GPT-2 small, `gpt2-small-res-jb`, 3 layers) every layer read
`"spectrally_structured"` (real, robust MP outliers — 179–198 of 768 dims,
surviving a matched-diagonal-Gaussian kill and a top-50-PC-of-x projection
kill) and **none** read `"topologically_structured"` or `"both"`: `tc_ratio`
was **below 1.0** at every layer (0.11–0.46) — the error cloud is *less*
topologically complex than its own feature-shuffled null, and below a
covariance-matched Gaussian control that correctly calibrates to ~1.0. Takes
plain numpy arrays; no `transformer_lens` / SAELens / torch dependency.

**Does my explanation glue into one global explanation, or does it fall apart
across the input space?** `explanation_obstruction` covers your examples with
overlapping clusters (from an embedding, or a cover you supply), fits a local
"explanation" per cluster (its top-k highest-|score| columns), and measures
the frustrated-cycle obstruction Ω to gluing those local explanations into
one global one — the discrete Čech-style construction of the sheaf-of-
explanations framework (arXiv 2603.17141):

```python
from whitebox.gluing import explanation_obstruction
rep = explanation_obstruction(scores, embedding=emb, k=10, k_e=200)   # scores: (n_examples, n_edges)
rep.omega, rep.null_lo, rep.null_hi, rep.p_null      # Ω and its example-label-shuffle null
rep.significant_vs_null                              # p_null <= 0.025 -- NOT a "distinct mechanisms" claim
rep.blocking_pairs                                   # top frustrated cluster pairs, bootstrap CIs
rep.cyclomatic                                        # the nerve's cyclomatic number
print(rep.caveat)                                     # read this before using significant_vs_null for anything
```

**There is no `verdict` and no "distinct mechanisms" claim generator.** On
the one battery run so far (E3, GPT-2 small, IOI vs an induction control) the
pre-registered matched single-mechanism control **failed its own null-band
test** (Ω above the shuffle null in 34/40 grid cells, including the summary
config) and IOI showed **smaller** raw Ω than that failed control in **every**
one of 40 grid cells — the opposite of the pre-registered direction. The one
IOI blocking pair that did clear its own null did not survive a second-
embedding stability check. `rep.caveat` states this in the returned object,
not only in documentation someone might skip: a positive `significant_vs_null`
supports "distinct mechanisms" only if a caller independently confirms a
matched single-mechanism control lands in its own null band AND the blocking
pair survives a second embedding — neither held in E3's battery. Takes plain
numpy arrays; no `transformer_lens` / EAP dependency.

## Instruments (no claims attached)

```python
film = wb.topological_film(model, probe, layers=["layer3", "avgpool"], checkpoints=ckpts)
film.tc("avgpool")      # per-epoch TC with 95 % subsampling intervals
film.sv("avgpool")      # structural velocity, paired-subsample intervals
film.betti("avgpool"); film.fpl("avgpool"); film.summary()
```

Or live during training with `wb.FilmRecorder(model, probe, layers).record(epoch)`.
Exact PH on a 500-point probe (PCA→30) takes 0.1 s; a 5-layer frame with 20
bootstrap subsamples ~7 s on a laptop GPU; the forward pass dominates for
large models (`benchmarks/PHASE1.md`, verified to GPT-2 XL).

## Layout

```
whitebox/core        reduce (fixed PCA), persistence (ripser / giotto-ph / landmark), film, bootstrap
whitebox/metrics     sv, tc (+ persistent entropy), betti, fpl, biastop
whitebox/signals     memorization (calibrated per family; standardize= option, PHASE6.md), correspondence (RTD)
whitebox/invariants  (v2) ReLU-MLP gauge group, orbit sampler, check_invariance, the invariant metrics
whitebox/legibility  (v2) attribution-graph measures, discovery pipeline, legibility_profile (no score)
whitebox/darkmatter  (v2) SAE-error MP/TW spectral null + topology-with-shuffle-null, residual_structure
whitebox/gluing      (v2) sheaf-obstruction on a cover's nerve, explanation_obstruction (no verdict)
whitebox/activations forward-hook extraction; HF hidden-state helper
benchmarks/          batteries, calibration, validation; results/*.csv + 300-dpi figures
AUDIT.md             what the paper's code/results were sorted into
VALIDATION.md        per-signal reliability table — a signal is public only if it has a row here
```

## Citing the methods used

Ripser (Bauer 2021); giotto-ph (Burella Pérez et al. 2021); persim; RTD
(Barannikov, Trofimov, Sotnikov, Trimbach, Filippov, Burnaev 2022);
persistent entropy (Rucco, Castiglione, Merelli, Pettini 2016);
subsampling for persistence (Fasy et al. 2014); maxmin landmarks (de Silva &
Carlsson 2004).
