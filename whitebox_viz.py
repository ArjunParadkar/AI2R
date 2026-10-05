"""
WhiteBOX Demo Visualizer
python3 whitebox_viz.py

A ~60s looping walkthrough, same style/spirit as PRISM-AD's tokamak_viz.py:
  1. a specific trained model
  2. the Jacobian as a distance metric (pullback g = J^T J)
  3. try relabelings, keep the ones that don't move anything
  4. read the frequencies off by Fourier, walk a full cycle as a test
  5. P1-P7: how that object (the metric / the group) answers each problem
  6. the library, end to end
Numbers quoted in the labels are the real scorecard from CLAIMS.md — passes,
fails and mixed results alike. Nulls are results.
"""
import subprocess, sys, os

def _bootstrap():
    try:
        import numpy, matplotlib
    except ImportError:
        print("Installing dependencies (one-time setup)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib"],
                              stdout=subprocess.DEVNULL)
        os.execv(sys.executable, [sys.executable] + sys.argv)

_bootstrap()

import numpy as np
import platform
import matplotlib
_os = platform.system()
if _os == 'Linux':
    matplotlib.use("gtk4agg")
elif _os == 'Darwin':
    matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ── palette (same family as the tokamak demo) ──────────────────────────────
CHALK   = "#0d1117"
BCOLOR  = "#0d1f0d"
BHOVER  = "#1a3a1a"
WHITE   = "#e8e8e8"
CYAN    = "#00e5ff"
MAGENTA = "#ff44ff"
YELLOW  = "#ffe14a"
ORANGE  = "#ff9900"
GREEN   = "#44ff88"
RED     = "#ff3b3b"
GREY    = "#6b7280"

FPS = 30.0
def fr(s): return s * FPS

# ── timeline (seconds) ──────────────────────────────────────────────────────
T_TITLE = (0.0, 3.2)
T_MODEL = (3.2, 9.8)
T_JAC   = (9.8, 17.8)
T_RELAB = (17.8, 25.3)
T_FOUR  = (25.3, 32.8)
T_PROB  = (32.8, 52.8)
T_LIB   = (52.8, 58.8)
T_END   = (58.8, 60.8)
TOTAL   = T_END[1]

PROBLEMS = [
    ("P1", "Name what it learned",
     "read off the characters χₖ(a)=e^{2πika/113}\nof G — the group found in step 3", GREEN,
     "PASS given G  ·  unsupervised: FAIL"),
    ("P2", "Explain the decision boundary",
     "the boundary sits where the shift Δ\n best preserves those same character phases", GREEN,
     "PASS   r = 1.000"),
    ("P3", "Catch & remove a shortcut",
     "the input-pullback metric's dominant\nslot names the token the net leans on", GREEN,
     "PASS   5/5 seeds, needs z known"),
    ("P4", "Flag unfamiliar inputs",
     "same metric g = JᵀJ — but it deletes\nexactly the directions the head ignores", RED,
     "FAIL   loses to softmax"),
    ("P5", "Predict failure before it happens",
     "curvature × Lipschitz of the manifold\nthat g defines around each point", YELLOW,
     "PASS on transformer  ·  FAIL on CNN"),
    ("P6", "Tell understanding from memorizing",
     "g's slots collapse onto (a, b); the kept\nswap symmetry from step 3 stays kept", YELLOW,
     "PASS on transformer  ·  partial on CNN"),
    ("P7", "Time it against grokking",
     "the full 113-cycle generator from step 4\narrives exactly with the accuracy jump", GREEN,
     "PASS   never on memorizers"),
]

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.2, 8.0), facecolor=CHALK)
ax = fig.add_axes([0.0, 0.185, 1.0, 0.79])
ax.set_facecolor(CHALK)
ax.set_xlim(0, 16); ax.set_ylim(0, 9)
ax.set_axis_off()

title_top = fig.text(0.5, 0.975, "WHITEBOX", ha="center", fontsize=15,
                      fontweight="bold", color=WHITE, fontfamily="monospace", alpha=0.55)
caption = fig.text(0.5, 0.138, "", ha="center", fontsize=12.5, color=WHITE,
                    fontfamily="monospace", alpha=0.0)
subcaption = fig.text(0.5, 0.110, "", ha="center", fontsize=9.5, color=GREY,
                       fontfamily="monospace", alpha=0.0)

def set_caption(text, sub="", a=1.0):
    caption.set_text(text); caption.set_alpha(a)
    subcaption.set_text(sub); subcaption.set_alpha(a * 0.9)

# ── generic timing helpers ──────────────────────────────────────────────────
def window(t, t0, t1, fin=0.4, fout=0.4):
    """0..1 envelope: fades in over `fin`, holds, fades out over `fout`."""
    if t < t0 or t > t1: return 0.0
    if t < t0 + fin: return (t - t0) / fin
    if t > t1 - fout: return max(0.0, (t1 - t) / fout)
    return 1.0

def lerp(a, b, f): return a + (b - a) * float(np.clip(f, 0, 1))

def local(t, t0, t1):
    return float(np.clip((t - t0) / (t1 - t0), 0, 1))

# =============================================================================
# SCENE 0 — TITLE CARD
# =============================================================================
s0_title = ax.text(8, 5.4, "WHITEBOX", ha='center', va='center',
                    fontsize=46, fontweight='bold', color=WHITE,
                    fontfamily='monospace', alpha=0)
s0_sub = ax.text(8, 4.3, "what does a trained network know?", ha='center', va='center',
                  fontsize=16, color=CYAN, fontfamily='monospace', alpha=0)
s0_by = ax.text(8, 3.4, "Arjun Paradkar & Reva Paradkar", ha='center', va='center',
                fontsize=11, color=GREY, fontfamily='monospace', alpha=0)

def draw_title(t):
    a = window(t, *T_TITLE, fin=0.5, fout=0.5)
    s0_title.set_alpha(a); s0_sub.set_alpha(a * 0.9); s0_by.set_alpha(a * 0.75)
    if a > 0:
        set_caption("seven runs. one library.", "", a)

# =============================================================================
# SCENE 1 — THE MODEL
# =============================================================================
def mkbox(x, y, w, h, label, color, fs=12):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.14",
                        linewidth=1.6, edgecolor=color, facecolor="#101820", alpha=0)
    ax.add_patch(p)
    tx = ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fs,
                 color=color, fontfamily='monospace', fontweight='bold', alpha=0)
    return p, tx

box_in   = mkbox(0.8,  4.2, 2.6, 1.4, "(a, b)\nmod 113", CYAN)
box_emb  = mkbox(4.3,  4.2, 2.3, 1.4, "embed", MAGENTA)
box_blk  = mkbox(7.4,  4.2, 3.0, 1.4, "attention\n+ MLP", ORANGE)
box_out  = mkbox(11.2, 4.2, 3.0, 1.4, "logits\n(113 classes)", YELLOW)
S1_BOXES = [box_in, box_emb, box_blk, box_out]

def mkarrow(p1, p2):
    a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=16,
                         color=WHITE, lw=1.6, alpha=0)
    ax.add_patch(a)
    return a

arr1 = mkarrow((3.4, 4.9), (4.3, 4.9))
arr2 = mkarrow((6.6, 4.9), (7.4, 4.9))
arr3 = mkarrow((10.4, 4.9), (11.2, 4.9))
S1_ARROWS = [arr1, arr2, arr3]

s1_label = ax.text(8, 6.3, "one trained model — grokked, mod-113 addition",
                    ha='center', fontsize=13, color=WHITE, fontfamily='monospace', alpha=0)

# the 113-point cloud lives here for the rest of the demo: it starts as a
# rough Euclidean blob (scene 1/2) and later straightens into a circle.
NPTS = 113
rng = np.random.default_rng(7)
BLOB_X = 8 + rng.normal(0, 1.35, NPTS)
BLOB_Y = 2.15 + rng.normal(0, 0.55, NPTS)
_theta = np.linspace(0, 2*np.pi, NPTS, endpoint=False)
CIRC_CX, CIRC_CY, CIRC_R = 8.0, 2.15, 1.35
CIRCLE_X = CIRC_CX + CIRC_R * np.cos(_theta)
CIRCLE_Y = CIRC_CY + CIRC_R * np.sin(_theta) * 0.62   # squashed to fit the band

cloud_rgba = np.tile(np.array([0.55, 0.85, 1.0, 0.0], dtype=np.float32), (NPTS, 1))
cloud = ax.scatter(BLOB_X, BLOB_Y, s=22, c=cloud_rgba, edgecolors='none', zorder=5)

s1_cloud_lbl = ax.text(8, 1.15, "hidden state (resid_mid), one dot per answer",
                        ha='center', fontsize=9.5, color=GREY, fontfamily='monospace', alpha=0)

def draw_model(t):
    env = window(t, *T_MODEL)
    if env <= 0:
        return
    lt = local(t, T_MODEL[0], T_MODEL[1] - 0.6)
    n_boxes = int(np.clip(lt * 4.6, 0, 4))
    for i, (p, tx) in enumerate(S1_BOXES):
        a = env if i < n_boxes else 0.0
        p.set_alpha(a); tx.set_alpha(a)
    for i, arr in enumerate(S1_ARROWS):
        a = env if i < n_boxes - 1 else 0.0
        arr.set_alpha(a)
    s1_label.set_alpha(env * min(1.0, lt*3))
    cloud_a = env * float(np.clip((lt-0.55)*3, 0, 1))
    cloud_rgba[:, 3] = cloud_a
    cloud.set_offsets(np.c_[BLOB_X, BLOB_Y])
    cloud.set_facecolors(cloud_rgba)
    s1_cloud_lbl.set_alpha(cloud_a)
    if env > 0:
        set_caption("This is a specific model.", "input -> embed -> attention/MLP -> 113-way output", env)

# =============================================================================
# SCENE 2 — THE JACOBIAN AS A DISTANCE METRIC
# =============================================================================
s2_formula = ax.text(8, 7.1, "g = JᵀJ", ha='center', fontsize=30,
                      fontweight='bold', color=WHITE, fontfamily='monospace', alpha=0)
s2_sub = ax.text(8, 6.15, "the Jacobian of the network, used as a local ruler",
                  ha='center', fontsize=11.5, color=CYAN, fontfamily='monospace', alpha=0)

# a handful of tiny ellipses = the local metric "stretching" the blob
ELLIPSE_ANCHORS = [(-2.2, 0.4), (-0.8, -0.9), (0.6, 0.7), (1.7, -0.4), (2.4, 0.6), (-1.6,-0.5)]
s2_ellipses = []
for (dx, dy) in ELLIPSE_ANCHORS:
    e = Ellipse((8+dx, 2.15+dy*0.4), width=0.9, height=0.28, angle=float(rng.uniform(0,180)),
                edgecolor=ORANGE, facecolor='none', lw=1.4, alpha=0)
    ax.add_patch(e)
    s2_ellipses.append(e)

eu_tag  = ax.text(4.6, 3.55, "Euclidean\nr ≈ 0.75–0.80", ha='center', fontsize=10.5,
                   color=RED, fontfamily='monospace', alpha=0)
pb_tag  = ax.text(11.4, 3.55, "pullback g\none clean circle\nr ≈ 0.98–0.99", ha='center',
                   fontsize=10.5, color=GREEN, fontfamily='monospace', alpha=0)

def draw_jacobian(t):
    env = window(t, *T_JAC)
    if env <= 0:
        return
    lt = local(t, *T_JAC)
    for i, p, tx in [(0, box_in[0], box_in[1])]:
        pass
    for (p, tx) in S1_BOXES:
        p.set_alpha(0.0); tx.set_alpha(0.0)
    for arr in S1_ARROWS:
        arr.set_alpha(0.0)
    s1_label.set_alpha(0.0)

    s2_formula.set_alpha(env * min(1.0, lt*3))
    s2_sub.set_alpha(env * min(1.0, max(0,lt-0.08)*3))
    ell_a = env * float(np.clip((lt-0.15)*2.2, 0, 1))
    for e in s2_ellipses:
        e.set_alpha(ell_a * 0.85)

    # morph blob -> circle across the middle third of the scene, hold, then
    # relax slightly back toward blob near the very end so scene 3 can start clean
    morph = float(np.clip((lt - 0.30) / 0.45, 0, 1))
    xs = lerp(BLOB_X, CIRCLE_X, morph)
    ys = lerp(BLOB_Y, CIRCLE_Y, morph)
    cloud.set_offsets(np.c_[xs, ys])
    cloud_rgba[:, 3] = env
    # blend color pink(blob)->cyan(circle)
    cloud_rgba[:, 0] = lerp(0.55, 0.25, morph)
    cloud_rgba[:, 1] = lerp(0.85, 0.95, morph)
    cloud_rgba[:, 2] = 1.0
    cloud.set_facecolors(cloud_rgba)

    fade_tags = float(np.clip((lt-0.55)*2.5, 0, 1))
    eu_tag.set_alpha(env*fade_tags*(1-morph*0.3))
    pb_tag.set_alpha(env*fade_tags*morph if morph>0.5 else 0.0)
    s1_cloud_lbl.set_alpha(0.0)
    if env > 0:
        set_caption("We find the Jacobian here — as a distance metric.",
                    "the same cloud: a blob in Euclidean distance, one circle under g", env)

# =============================================================================
# SCENE 3 — TRY RELABELINGS, KEEP THE ONES THAT DON'T MOVE ANYTHING
# =============================================================================
CANDIDATES = ["a ↔ a+1", "a ↔ 2a", "random π", "a ↔ b",
              "a ↔ c−a", "shift +5", "reverse", "random π"]
CAND_POS = [(2.4,7.3), (5.8,7.3), (9.2,7.3), (12.6,7.3),
            (2.4,5.6), (5.8,5.6), (9.2,5.6), (12.6,5.6)]
KEEP_IDX = 4  # "a <-> c-a"

s3_badges = []
for lbl, (x, y) in zip(CANDIDATES, CAND_POS):
    p = FancyBboxPatch((x-0.9, y-0.35), 1.8, 0.7, boxstyle="round,pad=0.06,rounding_size=0.18",
                        linewidth=1.3, edgecolor=GREY, facecolor="#101820", alpha=0)
    ax.add_patch(p)
    tx = ax.text(x, y, lbl, ha='center', va='center', fontsize=10, color=WHITE,
                 fontfamily='monospace', alpha=0)
    xm = ax.text(x, y, "✕", ha='center', va='center', fontsize=22, color=RED,
                 fontfamily='monospace', alpha=0, fontweight='bold')
    s3_badges.append(dict(p=p, tx=tx, xm=xm, x=x, y=y, keep=(lbl == "a ↔ c−a" and (x,y)==CAND_POS[KEEP_IDX])))

s3_headline = ax.text(8, 8.2, "try relabelings of the inputs", ha='center', fontsize=13,
                       color=WHITE, fontfamily='monospace', alpha=0)
s3_kept = ax.text(8, 3.3, "kept: a ↔ c−a  —  distances under g don't move",
                   ha='center', fontsize=12.5, color=GREEN, fontfamily='monospace',
                   fontweight='bold', alpha=0)

# fold arcs — show the circle folding onto itself under the kept relabeling
fold_line, = ax.plot([], [], color=GREEN, lw=2.0, alpha=0)

def draw_relabel(t):
    env = window(t, *T_RELAB)
    if env <= 0:
        return
    lt = local(t, *T_RELAB)
    s2_formula.set_alpha(0); s2_sub.set_alpha(0)
    for e in s2_ellipses: e.set_alpha(0)
    eu_tag.set_alpha(0)

    s3_headline.set_alpha(env * min(1.0, lt*4))
    # cloud sits as the circle, dim, off to the side conceptually (keep same axes coords)
    cloud_rgba[:, 3] = env * 0.55
    cloud_rgba[:,0]=0.25; cloud_rgba[:,1]=0.95; cloud_rgba[:,2]=1.0
    cloud.set_offsets(np.c_[CIRCLE_X, CIRCLE_Y])
    cloud.set_facecolors(cloud_rgba)
    pb_tag.set_alpha(0)

    n_settled = int(np.clip((lt-0.12)/0.09, 0, len(s3_badges)))
    for i, b in enumerate(s3_badges):
        appear = env * float(np.clip((lt*10 - i*0.9), 0, 1))
        if i < n_settled and not b['keep']:
            rej = float(np.clip((lt - (0.12+0.09*i))*6, 0, 1))
            b['p'].set_alpha(appear*(1-0.35*rej)); b['p'].set_edgecolor(RED if rej>0.4 else GREY)
            b['tx'].set_alpha(appear*(1-rej))
            b['xm'].set_alpha(appear*rej)
        elif b['keep']:
            glow = float(np.clip((lt-0.55)*4, 0, 1))
            b['p'].set_alpha(appear); b['p'].set_edgecolor(GREEN if glow>0 else GREY)
            b['p'].set_linewidth(1.3+2.2*glow)
            b['tx'].set_alpha(appear); b['tx'].set_color(GREEN if glow>0 else WHITE)
            b['xm'].set_alpha(0)
        else:
            b['p'].set_alpha(appear); b['tx'].set_alpha(appear); b['xm'].set_alpha(0)

    kept_a = env * float(np.clip((lt-0.68)*4, 0, 1))
    s3_kept.set_alpha(kept_a)
    if kept_a > 0.05:
        th = np.linspace(0, np.pi, 60)
        fold_line.set_data(CIRC_CX + CIRC_R*np.cos(th), CIRC_CY + CIRC_R*0.62*np.sin(th)*0.3+CIRC_CY*0)
        fold_line.set_alpha(kept_a*0.0)  # visual fold effect kept minimal/clean
    if env > 0:
        set_caption("Then we try relabelings — keep the ones that don't change any distance.",
                    "one survives: swapping a with c−a, inside each answer's fiber", env)

# =============================================================================
# SCENE 4 — READ THE FREQUENCIES OFF BY FOURIER; WALK A FULL CYCLE AS A TEST
# =============================================================================
NBARS = 24
bar_x = np.arange(NBARS) + 1
rngb = np.random.default_rng(11)
bar_heights_full = rngb.uniform(0.05, 0.28, NBARS)
WINNERS = [2, 5, 12, 17, 20]   # indices standing in for {3,6,26,50,51}-style winners
for w in WINNERS:
    bar_heights_full[w] = rngb.uniform(0.75, 0.98)

bars = ax.bar(2.2 + bar_x*0.5, np.zeros(NBARS), width=0.42, bottom=0.9,
              color=GREY, alpha=0)
s4_head = ax.text(8, 8.2, "read the frequencies off by Fourier", ha='center', fontsize=13,
                   color=WHITE, fontfamily='monospace', alpha=0)
s4_sub = ax.text(8, 7.55, "top-5 at r > 0.9: frequencies {3, 6, 26, 50, 51}", ha='center',
                  fontsize=10.5, color=YELLOW, fontfamily='monospace', alpha=0)

clock_cx, clock_cy, clock_r = 8.0, 4.85, 1.3
clock_face, = ax.plot([], [], color=CYAN, lw=2.2, alpha=0)
clock_ticks = ax.scatter([], [], s=10, color=YELLOW, alpha=0)
clock_label = ax.text(clock_cx, clock_cy, "", ha='center', va='center', fontsize=13,
                       fontweight='bold', color=WHITE, fontfamily='monospace', alpha=0)
clock_caveat = ax.text(8, 0.32, "(order isn't always recoverable by the letter —\nconfirmed post-hoc, on fresh unseen seeds)",
                        ha='center', fontsize=8.6, color=GREY, fontfamily='monospace', alpha=0)

def draw_fourier(t):
    env = window(t, *T_FOUR)
    if env <= 0:
        return
    lt = local(t, *T_FOUR)
    s3_headline.set_alpha(0); s3_kept.set_alpha(0)
    for b in s3_badges:
        b['p'].set_alpha(0); b['tx'].set_alpha(0); b['xm'].set_alpha(0)
    cloud_rgba[:, 3] = 0.0
    cloud.set_facecolors(cloud_rgba)
    pb_tag.set_alpha(0)

    s4_head.set_alpha(env*min(1.0, lt*4))
    grow = float(np.clip((lt-0.06)/0.30, 0, 1))
    heights = bar_heights_full * grow * 1.5
    for i, rect in enumerate(bars):
        rect.set_height(heights[i])
        rect.set_alpha(env*0.85)
        rect.set_color(YELLOW if i in WINNERS and grow>0.6 else GREY)
    s4_sub.set_alpha(env*float(np.clip((lt-0.42)*4,0,1)))

    cyc = float(np.clip((lt-0.45)/0.35, 0, 1))
    face_a = env*float(np.clip((lt-0.42)*4,0,1))
    th = np.linspace(0, 2*np.pi*cyc, max(2,int(cyc*220)+2))
    clock_face.set_data(clock_cx+clock_r*np.cos(th-np.pi/2), clock_cy+clock_r*np.sin(th-np.pi/2))
    clock_face.set_alpha(face_a)
    n_ticks = int(cyc*113)
    tth = np.linspace(0, 2*np.pi, 113, endpoint=False)[:n_ticks]
    clock_ticks.set_offsets(np.c_[clock_cx+ (clock_r-0.12)*np.cos(tth-np.pi/2),
                                   clock_cy+ (clock_r-0.12)*np.sin(tth-np.pi/2)])
    clock_ticks.set_alpha(face_a)
    stamp_a = env*float(np.clip((lt-0.82)*6,0,1))
    clock_label.set_text(f"{n_ticks}" if cyc < 0.995 else "order 113")
    clock_label.set_alpha(face_a if cyc<0.995 else stamp_a)
    clock_label.set_color(WHITE if cyc<0.995 else GREEN)
    clock_caveat.set_alpha(stamp_a*0.9)
    if env > 0:
        set_caption("Walk them forward a full cycle — as a test.",
                    "the generator + its characters recover 4–5 of 5 frequencies", env)

# =============================================================================
# SCENE 5 — P1 THROUGH P7
# =============================================================================
# one big spotlighted card (so the math explanation is actually readable) +
# a thin breadcrumb strip of P1..P7 pills across the top
N_PROB = len(PROBLEMS)
BC_Y = 7.85
BC_XS = np.linspace(2.2, 13.8, N_PROB)
s5_head = ax.text(8, 8.55, "same G, same g — seven questions:", ha='center', fontsize=13.5,
                   color=WHITE, fontfamily='monospace', alpha=0)

s5_crumbs = []
for i, ((num, *_rest), bx) in enumerate(zip(PROBLEMS, BC_XS)):
    pill = FancyBboxPatch((bx-0.62, BC_Y-0.28), 1.24, 0.56,
                           boxstyle="round,pad=0.03,rounding_size=0.14",
                           linewidth=1.3, edgecolor=GREY, facecolor="#101820", alpha=0)
    ax.add_patch(pill)
    lbl = ax.text(bx, BC_Y, num, ha='center', va='center', fontsize=10.5,
                   fontweight='bold', color=GREY, fontfamily='monospace', alpha=0)
    s5_crumbs.append(dict(p=pill, tx=lbl))

CARD_CX, CARD_CY, CARD_W, CARD_H = 8.0, 3.75, 11.4, 5.7
s5_card = FancyBboxPatch((CARD_CX-CARD_W/2, CARD_CY-CARD_H/2), CARD_W, CARD_H,
                          boxstyle="round,pad=0.10,rounding_size=0.22",
                          linewidth=2.2, edgecolor=WHITE, facecolor="#101820", alpha=0)
ax.add_patch(s5_card)
s5_num_title = ax.text(CARD_CX, CARD_CY+2.15, "", ha='center', va='center', fontsize=21,
                        fontweight='bold', color=WHITE, fontfamily='monospace', alpha=0)
s5_mech = ax.text(CARD_CX, CARD_CY+0.65, "", ha='center', va='center', fontsize=15.5,
                   color=WHITE, fontfamily='monospace', alpha=0, linespacing=1.7)
s5_verdict = ax.text(CARD_CX, CARD_CY-1.55, "", ha='center', va='center', fontsize=18,
                      fontweight='bold', color=WHITE, fontfamily='monospace', alpha=0)
s5_footer = ax.text(CARD_CX, CARD_CY-2.55, "built from G (step 3) + g = JᵀJ (step 2)",
                     ha='center', va='center', fontsize=10.5, color=GREY,
                     fontfamily='monospace', alpha=0)

def draw_problems(t):
    env = window(t, *T_PROB, fin=0.4, fout=0.5)
    if env <= 0:
        return
    for artist in (s4_head, s4_sub, clock_face, clock_ticks, clock_label, clock_caveat):
        artist.set_alpha(0)
    for rect in bars: rect.set_alpha(0)

    s5_head.set_alpha(env*0.9)
    seg = (T_PROB[1]-T_PROB[0]) / N_PROB
    lt = t - T_PROB[0]
    active = int(np.clip(lt/seg, 0, N_PROB-1))
    seg_t = lt - active*seg
    seg_a = window(seg_t, 0, seg, fin=0.3, fout=0.3)
    a = env * seg_a

    num, title_, mech, color, verdict = PROBLEMS[active]
    s5_card.set_alpha(a); s5_card.set_edgecolor(color)
    s5_num_title.set_text(f"{num} — {title_}"); s5_num_title.set_color(color)
    s5_num_title.set_alpha(a)
    s5_mech.set_text(mech); s5_mech.set_alpha(a)
    s5_verdict.set_text(verdict); s5_verdict.set_color(color); s5_verdict.set_alpha(a)
    s5_footer.set_alpha(a*0.85)

    for i, crumb in enumerate(s5_crumbs):
        pcolor, _, _, ccolor, _ = PROBLEMS[i]
        if i < active:
            crumb['p'].set_alpha(env*0.9); crumb['p'].set_edgecolor(ccolor)
            crumb['tx'].set_alpha(env*0.9); crumb['tx'].set_color(ccolor)
        elif i == active:
            crumb['p'].set_alpha(env); crumb['p'].set_edgecolor(color); crumb['p'].set_linewidth(2.4)
            crumb['tx'].set_alpha(env); crumb['tx'].set_color(WHITE)
        else:
            crumb['p'].set_alpha(env*0.5); crumb['p'].set_edgecolor(GREY); crumb['p'].set_linewidth(1.3)
            crumb['tx'].set_alpha(env*0.5); crumb['tx'].set_color(GREY)
    if env > 0:
        set_caption(f"{num}: {title_}", "the object we built — the group + the pullback metric — is what answers it", env)

# =============================================================================
# SCENE 6 — THE LIBRARY
# =============================================================================
LIB_LINES = [
    ("model = open_model(\"z113_clean_s0\")", "4.6s"),
    ("g = model.metric(\"pullback\")", "ok"),
    ("g.p1_inventory()   # frequencies + fibers", "5/5"),
    ("g.p2_boundary(x)   # geodesic explanation", "r=1.00"),
    ("g.p3_monitor()     # shortcut detector", "0 FP"),
    ("g.p6_understanding()  # slot collapse + swap", "PASS"),
]
s6_lines = []
for i, (code, tag) in enumerate(LIB_LINES):
    y = 7.0 - i*0.85
    tx = ax.text(2.0, y, code, ha='left', fontsize=11.5, color=WHITE,
                 fontfamily='monospace', alpha=0)
    tag_x = ax.text(13.6, y, tag, ha='right', fontsize=11, color=GREEN,
                     fontfamily='monospace', fontweight='bold', alpha=0)
    s6_lines.append((tx, tag_x))

s6_head = ax.text(8, 8.1, "and finally — the library", ha='center', fontsize=13,
                   color=WHITE, fontfamily='monospace', alpha=0)
s6_name = ax.text(8, 1.0, "whiteboxtopo", ha='center', fontsize=22, fontweight='bold',
                   color=CYAN, fontfamily='monospace', alpha=0)
s6_tag = ax.text(8, 0.35, "math first. nulls are results.", ha='center', fontsize=10.5,
                  color=GREY, fontfamily='monospace', alpha=0)

def draw_library(t):
    env = window(t, *T_LIB, fin=0.4, fout=0.35)
    if env <= 0:
        return
    s5_card.set_alpha(0)
    for t_ in (s5_num_title, s5_mech, s5_verdict, s5_footer): t_.set_alpha(0)
    for crumb in s5_crumbs: crumb['p'].set_alpha(0); crumb['tx'].set_alpha(0)
    s5_head.set_alpha(0)

    lt = local(t, *T_LIB)
    s6_head.set_alpha(env*min(1.0, lt*4))
    n_shown = int(np.clip(lt*8.0, 0, len(s6_lines)))
    for i, (tx, tag) in enumerate(s6_lines):
        a = env if i < n_shown else 0.0
        tx.set_alpha(a); tag.set_alpha(a)
    name_a = env*float(np.clip((lt-0.72)*5,0,1))
    s6_name.set_alpha(name_a)
    s6_tag.set_alpha(name_a*0.85)
    if env > 0:
        set_caption("Same object, seven different questions — one library.",
                    "", env)

def draw_end(t):
    env = window(t, *T_END, fin=0.2, fout=0.5)
    if env <= 0:
        return
    s6_name.set_alpha(env); s6_tag.set_alpha(env*0.85); s6_head.set_alpha(0)
    for i, (tx, tag) in enumerate(s6_lines):
        tx.set_alpha(0); tag.set_alpha(0)
    set_caption("whiteboxtopo", "Arjun Paradkar & Reva Paradkar", env)

# ── master reset (used before every frame's redraw) ─────────────────────────
ALL_TEXT_ALPHA_RESETTERS = []

def hide_everything():
    for p, tx in S1_BOXES:
        p.set_alpha(0); tx.set_alpha(0)
    for arr in S1_ARROWS: arr.set_alpha(0)
    s1_label.set_alpha(0); s1_cloud_lbl.set_alpha(0)
    s0_title.set_alpha(0); s0_sub.set_alpha(0); s0_by.set_alpha(0)
    s2_formula.set_alpha(0); s2_sub.set_alpha(0)
    for e in s2_ellipses: e.set_alpha(0)
    eu_tag.set_alpha(0); pb_tag.set_alpha(0)
    s3_headline.set_alpha(0); s3_kept.set_alpha(0)
    for b in s3_badges:
        b['p'].set_alpha(0); b['tx'].set_alpha(0); b['xm'].set_alpha(0)
    s4_head.set_alpha(0); s4_sub.set_alpha(0)
    for rect in bars: rect.set_alpha(0)
    clock_face.set_alpha(0); clock_ticks.set_alpha(0)
    clock_label.set_alpha(0); clock_caveat.set_alpha(0)
    s5_head.set_alpha(0)
    s5_card.set_alpha(0)
    for t_ in (s5_num_title, s5_mech, s5_verdict, s5_footer): t_.set_alpha(0)
    for crumb in s5_crumbs: crumb['p'].set_alpha(0); crumb['tx'].set_alpha(0)
    s6_head.set_alpha(0); s6_name.set_alpha(0); s6_tag.set_alpha(0)
    for tx, tag in s6_lines: tx.set_alpha(0); tag.set_alpha(0)
    cloud_rgba[:, 3] = 0.0
    cloud.set_facecolors(cloud_rgba)
    set_caption("", "", 0)

SCENES = [
    (T_TITLE, draw_title),
    (T_MODEL, draw_model),
    (T_JAC,   draw_jacobian),
    (T_RELAB, draw_relabel),
    (T_FOUR,  draw_fourier),
    (T_PROB,  draw_problems),
    (T_LIB,   draw_library),
    (T_END,   draw_end),
]

# ── state / controls ────────────────────────────────────────────────────────
STATE = dict(t0=None, paused=False, manual_t=None)

def on_pause(_):
    STATE['paused'] = not STATE['paused']
    btn_pause.label.set_text('▶ PLAY' if STATE['paused'] else '⏸ PAUSE')
    fig.canvas.draw_idle()

def jump_to(t):
    STATE['manual_t'] = t

BTN_STYLE = dict(color=BCOLOR, hovercolor=BHOVER)
def style_btn(btn):
    btn.label.set_color('white'); btn.label.set_fontfamily('monospace'); btn.label.set_fontsize(8)

ax_pause = fig.add_axes([0.005, 0.015, 0.075, 0.07])
ax_pause.set_facecolor(BCOLOR)
btn_pause = Button(ax_pause, '⏸ PAUSE', **BTN_STYLE); style_btn(btn_pause)
btn_pause.on_clicked(on_pause)

nav_defs = [("Title", T_TITLE[0]), ("Model", T_MODEL[0]), ("Jacobian", T_JAC[0]),
            ("Relabel", T_RELAB[0]), ("Fourier", T_FOUR[0]), ("P1-P7", T_PROB[0]),
            ("Library", T_LIB[0])]
nav_btns = []
bw, bh, bpad = 0.122, 0.07, 0.006
x0 = 0.09
for i, (label, tstart) in enumerate(nav_defs):
    axi = fig.add_axes([x0 + i*(bw+bpad), 0.015, bw, bh])
    axi.set_facecolor(BCOLOR)
    btn = Button(axi, label, **BTN_STYLE); style_btn(btn)
    btn.on_clicked(lambda _, tt=tstart: jump_to(tt))
    nav_btns.append(btn)

import time as _time

def update(_frame):
    if STATE['paused']:
        return
    now = _time.monotonic()
    if STATE['t0'] is None:
        STATE['t0'] = now
    if STATE['manual_t'] is not None:
        STATE['t0'] = now - STATE['manual_t']
        STATE['manual_t'] = None
    t = (now - STATE['t0']) % TOTAL

    hide_everything()
    for (t0, t1), fn in SCENES:
        if t0 - 0.6 <= t <= t1 + 0.6:
            fn(t)

anim = FuncAnimation(fig, update, frames=None, interval=1000.0/FPS,
                     blit=False, cache_frame_data=False)
plt.show()
