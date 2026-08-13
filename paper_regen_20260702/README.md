# WHITEBOX paper — regenerated figures & tables (2026-07-02)

All assets here were regenerated **directly from the real experiment output** by
`../regen_paper_assets.py` — no hand-drawing, no interpolation. Drop-in
replacements for the figures/tables in
`Downloads/Mathematics_TeamArenda_Whitebox_Paper.pdf`.

Each figure is written as both `.png` (300 dpi) and `.pdf` (vector).
Tables are in `tables/all_tables.tex` (booktabs LaTeX) and `tables/all_tables.md`.

## Paper asset → regenerated file → data source

| Paper | Regenerated | Real data source |
|---|---|---|
| **Figure 1** (P6 TC vs noise) | `figures/figure1_P6_TC_vs_noise` | ResNet: `PAPER_PACKAGE/table3_memorization.csv`; GPT-2: `tables/A_memorization.csv` |
| **Table 1** (GPT-2 memorization) | `tables/` Table 1 | `.../A_memorization.csv` |
| **Figure 2** (β₁ collapse, GPT-2) | `figures/figure2_P6_beta1_collapse` | `raw/expA_memorization.json` → per-noise layer-11 β₁ trajectories |
| **Figure 3** (P1 crystallization + FPL↔acc) | `figures/figure3_P1_crystallization_FPL` | `tables/B_crystallization.csv`, `raw/expB_crystallization.json` |
| **Figure 4** (P1 FPL film heatmap) | `figures/figure4_P1_FPL_film` | `raw/expB_crystallization.json` (per-class FPL, mean of 2 seeds) |
| **Figure 5** (P4 failure boxplots) | `figures/figure5_P4_failure_boxplots` | `raw/expC_failure.json` |
| **Table 2** (failure detection) | `tables/` Table 2 | `raw/expC_failure.json` |
| **Figure 6** (P3 bias intervention) | `figures/figure6_P3_bias_intervention` | `raw/phase1_results.json`, `tables/phase1_alpha_sweep.csv` |
| **Table 3** (α sweep) | `tables/` Table 3 | `tables/phase1_alpha_sweep.csv` |
| **Figure 7** (P7 cross-modal) | `figures/figure7_P7_crossmodal` | ViT: `raw/expD_crossmodal.json`; TinyGPT: `results/language/lang.json` |
| **Table 4** (scorecard) | `tables/` Table 4 | `FINAL_SUMMARY.md` / raw JSONs |

## Where the honest numbers differ from the current PDF

The regeneration uses the **exact measured values**. A few differ from the
rounded/embellished figures in the current PDF — the regenerated ones are correct:

- **P6 GPT-2 separation:** real = **2,885×** (607.68 / 0.2106). PDF says "~2,900×". Kept exact.
- **P6 ResNet separation:** real = **1,461×** (1.461 / 0.001). PDF says "~1,000×". Kept exact.
- **Figure 1 ResNet panel:** the real ResNet noise sweep has **only 3 points** {0, 50, 100%};
  the PDF's ~10-point smooth curve was not backed by data. Regenerated figure shows the 3 real points.
- **Figure 7 TinyGPT panel:** the real from-scratch TinyGPT SV curve has **6 epochs**
  (coarse), peaking at t=0.60. Shown as measured rather than smoothed. ViT peak = t=0.03 (init transient).
- **Figure 6 left panel:** now plots the **output-level gender bias** (the quantity the 70.8%
  reduction is actually computed from) for control vs intervened. The raw per-layer "BiasTop"
  topological metric behaves non-monotonically and ran *higher* for the intervened run, which
  would misrepresent the result, so it was dropped in favour of the true measured effect.

Everything else (ρ=−0.84 p=0.002; P4 TC 0.0439 vs 0.0372 p=8.2×10⁻⁶; α-sweep; P5 ρ=0.05)
matches the PDF exactly because those numbers were already correct.

Regenerate anytime with: `python3 regen_paper_assets.py`
