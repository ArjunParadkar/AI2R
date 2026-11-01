# RUNLOG4 — Run 4 (The Tower of Quotients). Started 2026-09-20 15:59 CDT.
Read SPEC4.md first (brief docs/RUN4_BRIEF.md; amendments A-13+). Outputs results/run4/, FINDINGS4_*.md, PAUSE4_n.md.
* 16:00 specimens launched (wb-r4-spec). 16:02 SPEC4.md written. 16:14 A-13: GPU is power-capped by the system (510 MHz); ResNet suspended
  (SIGSTOP) and to be restarted by wb-r4-spec2 after Z113 wave 1; wave 2 shortened. Scripts ready: analysis/run4_z113.py (finals|sweep),
  analysis/run4_resnet.py, scripts/run4_interrupt.sh (P3.2; launch when the GPU is idle). Planted tests tests/test_run4_delta_symmetry.py pass (4).
* 21:36 DISCOVERED: finals job crashed at 17:15 (requires_grad on a fresh model); my waiter watched for a line that could not appear -> 4 h idle. All specimens trained (ResNet memorisers 99.6/99.4% train, 10% test; clean 99% train, 58% test); 6 sweeps done. Fixed; launched wb-r4-a2 (finals + decoy sweeps + resnet analysis) and wb-r4-int (P3.2, waits for the ResNet analysis to leave the GPU).
* 22:05 ALL RUN-4 ANALYSIS DONE except P3.2 (interrupt, wb-r4-int, ~100 penalised steps/min on the capped GPU; expected to end ~00:15).
  By the letter: PASS = P1.1 (group discovered on unseen seeds), P1.2, P1.4, P3.1 (dominance rule incl. its registered limitation), P6.2, P6.5.
  FAIL = P6 block (2/5; concentration null), P6.3, P6.4, P1.3 (linearisation artefacts), P7.1–7.3 (generator LEADS grokking).
  Written: FINDINGS4_P6/P1/P7/P3.md, PAUSE4_1..3.md, CLAIMS.md run-4 section, figures/grid_four_runs.png + run4_*.png, demo_run4, report
  https://claude.ai/artifact/WfqvVDbibnGXGKaZD1wo6f . TO FINISH: when runs/r4_interrupt_s*/history.json reach step 12000 →
  `python analysis/run4_fill_p32.py && python analysis/fig_run4.py && python analysis/make_grid_run4.py`, copy figures to report/figures, republish
  report/blinding-light-run4.html. Tests 206 pass / 1 xfail. Nothing committed to git.
* 23:49 P3.2 LANDED: PASS 5/5 (dec@6000 0.999/1/1/1/1; 0 floor firings in (6000,12000], 12 checks per seed; test@12000 0.999–1.000; max λ 3). Seed 0 transient dip 0.943 at step 7500, recovered by 8000 (recorded in FINDINGS4_P3/CLAIMS). Filler + figures + grid regenerated; report republished; logbook Run-4 entry filled. RUN 4 ENDS 23:49 (7 h 50 min wall, incl. 4 h 20 idle). Bylines now Arjun Paradkar & Reva Paradkar, independent.
