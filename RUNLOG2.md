# RUNLOG2 — Run 2 (Morse-theoretic upgrade). Started 2026-09-20 09:24 CDT, deadline 16:24.
Read SPEC2.md (pre-registration) and AMENDMENTS.md first. Brief: docs/RUN2_BRIEF.md.
Resume: `systemctl --user list-units 'wb-*'`; block outputs in results/run2/, findings in FINDINGS_*.md, pauses in PAUSE_n.md.
Rules learned in run 1: launch via scripts/unit.sh under systemd-run; cap BLAS threads; few concurrent CUDA processes (30 GB RAM).

* 09:28 SPEC2.md + AMENDMENTS A-1 written before any computation. Blocks A, B, C delegated to three background agents
  (each: analysis/run2_block{A,B,C}.py → results/run2/block{A,B,C}.json, FINDINGS_{A,B,C}.md, new packages
  whiteboxtopo/{depthprofile,kernel,morse}). Main session does G → E1/E2 → D → E3 → F → report.
* 09:30 regenerating the re-encoded model (A-1): units wb-r2-reenc{0,1} → runs/r2_reencoded_s{0,1}/ckpt_*.pt
* 09:33 Block G done: FAIL by the letter (CV ring 0.29 / GPT-2 0.24; ResNet penultimate m_plateau 0; ordering wrong). FINDINGS_G.md. E1/E2 running (wb-r2-E12).
* 09:40 PAUSE_1.md written: E1 NOT MET (0/5 named, 3 seeds), E2 prediction holds 2/3, G FAIL. D1 first pass: raw-NMI threshold saturates (>1); AMI separates (clean ≤0.12, decoy ≤0.35, shortcut 1.0).
* 10:10 PAUSE_2.md (D2 NOT MET by letter; direct NMI floor catches evasion, Mapper-AMI false-alarms on decoys), PAUSE_3.md (A, B, C all FAIL). Agents A/B/C finished; FINDINGS_A/B/C.md on disk. F: FAIL (unseen decoy_s1 breaks it). D3/D4 running under A-3 (wb-r2-D34).
* 10:22 E3 (Z2 quotient MET incl. UNSEEN; rot-MNIST NOT MET), D5 null, F FAIL, figures run2_{alignment_vs_h1,vineyards,mplateau}.png, demo_run2, grid script done. D3/D4 (wb-r2-D34) running: ~100 min per schedule. Early: generalised KL penalty worse than run 1's (4/5 seeds escalated to 300 by step 1500).
* 10:52 CLAIMS.md run-2 section, grid_both_runs.png, report template (report/run2-report.template.html), demo_run2 written. Waiting on wb-r2-D34 (R, 12000 steps) and wb-r2-D34ph (A-4 arm). When R ends: STOP wb-r2-D34 (it would start G at 12000 steps) and launch G with --steps 6000 (A-5). Then: python analysis/run2_d34_summary.py → fill __D34_ROW__/__D34_TEXT__/__END__ in the template → publish; write FINDINGS_D.md, PAUSE_4.md.
* 11:21 D3 (R, post-hoc trigger) at step 6000: 1/5 (s3). A-4 arm at step 2000: 5/5 ≥0.991 at λ=3. Killed collapsed R seeds 0,1,2,4 (A-6); launched G 6000 steps as wb-r2-D34G. NOTE: when R s3 reaches 12000 the wb-r2-D34 script will start its own G loop at 12000 steps -> stop unit wb-r2-D34 then.
* 12:18 G seeds 1,2 died of CUDA OOM at step 500 (11 procs on 8 GB); relaunch queued behind the A-4 arm (wb-r2-D34G12). (operational; logged in RUNLOG2 only)
* 13:20 ALL RUN-2 WORK FINISHED. D3 post-hoc trigger: registered penalty 1/5 (R) and 1/5 (G), four seeds flattened by escalation;
  A-4 arm (run-1 penalty + direct trigger) 5/5 at λ=3 incl. seed 1; D4 on R seed 3: 0 firings in 12 checks. PAUSE_4.md, FINDINGS_D.md,
  CLAIMS.md (run-2 section), figures/grid_both_runs.png, report/blinding-light-run2.html written. Tests 174 pass / 1 xfail.
  Report artifact: https://claude.ai/artifact/7UiAFijxyRyKQ7AbHKBEK9 (run 1: https://claude.ai/artifact/Hwri64g186N9qiLK7TsTys).
  No wb-* units running. Nothing committed to git (not asked). Snapshot in ~/blindinglight_backups/.
