# RUNLOG3 — Run 3 (The Symmetry Group). Started 2026-09-20 13:38 CDT, deadline 16:38.
Read SPEC3.md first (brief: docs/RUN3_BRIEF.md). Outputs results/run3/, FINDINGS_{S,P1,P6,P3,P7,P2}.md, PAUSE3_n.md.
* 13:39 fresh-seed training launched (wb-r3-fresh). 13:42 SPEC3.md written. Agents IMG (S1,S2,S4,S6-rot, P1.4, P6.1, P6.2, P4/P5) and
  P3 (S5, P3.1–P3.4) delegated; main session: S3, S7, P1.1–1.3, P1.5, P6.3, P6.4, P7, P2.
* 13:52 first launch of blockZ died of CUDA OOM (GPU full of training + agents); block Z moved to CPU (WB_DEV=cpu) and relaunched.
* 14:06 load 195 / 4 GB free (17 python procs: fresh training x5, agent P3 x6, agent IMG, mine x5). Stopped my 4 sweeps (partial results in results/run3/sweep_*.json are overwritten on relaunch); finals keeps running; sweeps relaunch when agent P3's p31 scan ends.
* 14:35 PAUSE3_1.md: P1.1 group NOT MET by letter (orders 1,1,2,13,5); ring NOT MET; P1.2, P1.5, P2.1 met. POST-HOC (A-9/A-10): full 113-cycle on 5/5 grokked, not memoriser; characters 5,4,4,4,4 of 5. Sweeps running (wb-r3-sw4-*), agents IMG + P3 running, fresh seeds training (ResNet very slow under contention).
* 15:10 agent P3 done (FINDINGS_P3.md saved): P3.1 NOT MET (Spearman 0.68; decoy FPs 39/102; fires 122/122 where floor fires incl. 7/7 evasion). P3.3 stopped, NOT SCORED (A-12). Fresh Z113 sparse sweeps + P6.4 launched.
* 15:40 RUN-3 WORK FINISHED except the tail of the scrambled_s0 sweep (memoriser; ≤1 on every checkpoint scored; rerun `python analysis/run3_score_sweeps.py`
  when it ends — it cannot change a verdict). All findings, PAUSE3_0..3, CLAIMS.md run-3 section, figures/grid_three_runs.png, demo_run3, report
  written. Tests 202 pass / 1 xfail. Report: https://claude.ai/artifact/NerxDzPFX1pR2rCBv9SJ34 . P3.3 NOT SCORED (A-12). Nothing committed to git.
* 15:47 scrambled_s0 sweep finished: letter order 1 and no post-hoc cycle on all 51 checkpoints (73 memoriser checkpoints in total); counts updated, grid + report republished. No wb-* units running.
