**Table 1: GPT-2 124M memorization signatures under label noise. TC separation clean:corrupt $\approx$2,885$\times$.**

| Noise % | TC (final) | SV peak | beta1 |
|---|---|---|---|
| 0 (clean) | 607.68 | 107.6 | 14 |
| 25 | 288.79 | 197.1 | 17 |
| 50 | 192.33 | 214.3 | 12 |
| 75 | 181.96 | 201.3 | 8 |
| 100 (corrupt) | 0.21 | 151.0 | 1 |

**Table 2: Topological failure-detection statistics (ResNet-18, layer 3).**

| Group | TC (layer 3) | beta1 (layer 3) | n |
|---|---|---|---|
| Correctly classified | 0.0439 | 26.9 | 8,947 |
| Misclassified | 0.0372 | 27.1 | 1,053 |
| Mann--Whitney p | 8.2e-06 | 0.99 | --- |

**Table 3: $\alpha$ sweep for gender-bias directional erasure (GPT-2 124M).**

| alpha | Bias reduction | Perplexity cost | Notes |
|---|---|---|---|
| 0.1 | -0.3% | -0.2% | No effect |
| 0.3 | 32.1% | +0.9% | Moderate |
| 0.5 | 70.8% | +1.9% | Headline |

**Table 4: Six-problem scorecard.**

| Problem | WHITEBOX result | Status |
|---|---|---|
| P1 Concept robustness | FPL<->acc rho=-0.84, p=0.002 | Deepened |
| P3 Bias intervention | 70.8% bias - @ +1.9% ppl (controlled); COMPAS/toxicity fail | Partial / scoped |
| P4 Failure detection | TC 0.044 vs 0.037, p=8.2e-06 | Supported |
| P5 Fragility under shift | SV<->acc-drop rho=0.05, p=0.88 (n.s.) | Negative (clean) |
| P6 Memorization | clean:corrupt TC ~2,885x (GPT-2), 1,461x (ResNet) | Solved |
| P7 Cross-modal universality | ViT peak t=0.03 vs TinyGPT t=0.60 (do not align) | Refuted |

