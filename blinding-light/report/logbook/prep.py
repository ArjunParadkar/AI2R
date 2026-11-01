from PIL import Image
import os
A=os.path.expanduser('~/WhiteBOX/WhiteBOX - Archive/'); B=os.path.expanduser('~/blindinglight/figures/')
src={
 'may_sv':A+'v1/original/results/figures/figure1_SV_trajectory.png',
 'jun_int':A+'v1/experiments/results/whitebox_breakthrough/20260628_055217/figures/INT-1_headline.png',
 'jun_tc':A+'v1/experiments/results/whitebox_3hour/20260628_151432/figures/A_TC_vs_noise.png',
 'aug_e2':A+'v2-programme/E2/results/fig_agreement.png',
 'aug_e4':A+'v2-programme/E4/results/fig_phase_Q_sparsity.png',
 'aug_p6':A+'whitebox-v2/benchmarks/results/phase6/phase6_invariance_cv.png',
 'r1_grok':B+'grokking.png','r1_int':B+'interrupt.png','r1_inv':B+'inventory.png',
 'r2_align':B+'run2_alignment_vs_h1.png','r2_mplat':B+'run2_mplateau.png',
 'r3_form':B+'run3_group_formation.png','r3_group':B+'run3_discovered_group.png','r3_fresh':B+'run3_fresh_seeds.png',
 'r4_form':B+'run4_formation.png','r4_p6':B+'run4_p6_instruments.png','r4_dom':B+'run4_dominance.png','r4_int':B+'run4_interrupt.png','grid4':B+'grid_four_runs.png','grid5':B+'grid_five_runs.png','grid6':B+'grid_six_runs.png','r6_roc':B+'run6_roc.png','grid7':B+'grid_seven_runs.png','r7_inv':B+'run7_inventory.png',
}
for k,p in src.items():
    if not os.path.exists(p): print('MISSING',k); continue
    im=Image.open(p).convert('RGB'); W=2200 if k not in ('grid4','grid5','grid6','grid7') else 3400
    if im.width>W: im=im.resize((W,int(im.height*W/im.width)),Image.LANCZOS)
    im.save(f'img/{k}.png',optimize=True); print(k,im.size,os.path.getsize(f'img/{k}.png')//1024)
