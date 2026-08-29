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
(recommended, multi-threaded exact PH); torch only for model hooks.

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
whitebox/signals     memorization (calibrated per family), correspondence (RTD)
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
