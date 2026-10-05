"""Deliverable 3: the inventory figures.  Every panel plots a recovered circular coordinate
against the ground-truth variable it should be (a straight diagonal = the coordinate IS the
variable).  Coordinates are defined up to rotation and reflection; they are aligned for display.
"""
import sys, os, json, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *
from whiteboxtopo.topology import isotypic_decomposition, persistence, circular_coordinates, circular_correlation


def align(theta, truth):
    """Best rotation/reflection of theta onto truth (display only)."""
    best = None
    for sgn in (1, -1):
        off = np.angle(np.mean(np.exp(1j * (truth - sgn * theta))))
        t = np.mod(sgn * theta + off, 2 * np.pi)
        err = np.abs(np.angle(np.exp(1j * (t - truth)))).mean()
        if best is None or err < best[0]:
            best = (err, t)
    return best[1]


def circle_coord(Y, D=None):
    res = persistence(Y if D is None else D, distance_matrix=D is not None); h1 = res['dgms'][1]
    j = int(np.argmax(h1[:, 1] - h1[:, 0]))
    return circular_coordinates(Y, h1, res['cocycles'][1], D=res['D'], class_idx=j)['theta']


def panel(ax, truth, theta, title, color):
    truth = np.mod(truth, 2 * np.pi)
    ax.scatter(truth, align(theta, truth), s=9, color=color, linewidths=0)
    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], color=INK2, lw=0.8, ls=':')
    ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi); ax.set_aspect('equal')
    ax.set_xticks([0, np.pi, 2 * np.pi], ['0', 'π', '2π']); ax.set_yticks([0, np.pi, 2 * np.pi], ['0', 'π', '2π'])
    ax.set_title(f"{title}\nr = {circular_correlation(theta, truth):.3f}", fontsize=9)


def main():
    setup()
    from analysis.z113_common import checkpoints, load, slice_probe, acts
    from systems.z113 import P
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.6))
    # ---- row 1: Z113, the five isotypic planes (symmetry-informed: the probe sweeps a at b=0)
    m = load(checkpoints('runs/z113_clean_s1')[0][-1], seed=1); sl = slice_probe()
    X = acts(m, sl['toks'], 'resid_mid'); a = sl['a']
    dec, _ = isotypic_decomposition(X, a, P, n_freq=5, detect='dft', var_floor=0.0)
    for ax, d in zip(axes[0], dec):
        th = circle_coord(np.ascontiguousarray(d['proj']).astype(np.float32))
        panel(ax, 2 * np.pi * d['freq'] * a / P, th, f"Z113  k={d['freq']}  ({100*d['var']:.0f}% var.)", BLUE)
        ax.set_xlabel('2π·k·a / 113')
    axes[0, 0].set_ylabel('H¹ circular coordinate')
    # ---- row 2: ring RNN, Euclidean vs pullback-geodesic, then rot-MNIST layers
    from systems.ring_rnn import RingRNN
    from analysis.ring_pullback import cloud
    from whiteboxtopo.extract import jacobians
    from whiteboxtopo.manifold import fit_manifold
    from whiteboxtopo.problems.core import geodesic_distance_matrix
    r = RingRNN(seed=1).cuda(); r.load_state_dict(torch.load(sorted(glob.glob('runs/ring_s1/ckpt_*.pt'))[-1])); r.eval()
    Xr, th, _ = cloud(r, 1000, seed=7)
    panel(axes[1, 0], th, circle_coord(Xr), 'ring RNN  Euclidean Rips', ORANGE)
    Dg = geodesic_distance_matrix(fit_manifold(Xr, jacobians(r.psi, Xr).astype(np.float32), metric='g'))
    panel(axes[1, 1], th, circle_coord(None, Dg), 'ring RNN  pullback-geodesic Rips', BLUE)
    for ax in axes[1, :2]:
        ax.set_xlabel('true heading')
    axes[1, 0].set_ylabel('H¹ circular coordinate')
    # rot-MNIST: median coordinate/angle agreement per layer, trained vs untrained vs pixels
    ax = axes[1, 2]; gs = ax.get_gridspec()
    for a_ in axes[1, 2:]:
        a_.remove()
    ax = fig.add_subplot(gs[1, 2:])
    c = json.load(open('results/rotmnist_controls.json'))
    layers = ['pixels', 'conv1', 'conv2', 'conv3_gap', 'fc1']; xs = np.arange(len(layers))
    tr = [c['trained']['r_median'][L] for L in layers]; un = [np.nan] + [c['untrained_s0']['r_median'][L] for L in layers[1:]]
    ax.plot(xs, tr, '-o', color=BLUE, ms=7, label='trained'); ax.plot(xs, un, '-o', color=ORANGE, ms=7, label='untrained (random weights)')
    ax.axhline(0.9, color=INK2, lw=0.8, ls=':'); ax.annotate('certification threshold r = 0.9', (len(layers) - 1, 0.9), xytext=(0, 4), textcoords='offset points', ha='right', fontsize=8, color=INK2)
    end_label(ax, xs[-1], tr[-1], f'trained {tr[-1]:.2f}'); end_label(ax, xs[-1], un[-1], f'untrained {un[-1]:.2f}')
    ax.set_xticks(xs, ['input pixels', 'conv1', 'conv2', 'conv3 + pool', 'fc1']); ax.set_ylim(0, 1.05); ax.set_xlim(-0.3, len(layers) - 0.2)
    ax.set_ylabel('median r(coordinate, rotation angle)'); ax.legend(loc='lower left')
    ax.set_title('rotated-MNIST: the rotation circle is inherited from the input; training destroys it', fontsize=9)
    fig.tight_layout(); fig.savefig('figures/inventory.png'); print('figures/inventory.png')


if __name__ == '__main__':
    main()
