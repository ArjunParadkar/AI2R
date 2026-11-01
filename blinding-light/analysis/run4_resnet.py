"""Run 4, ResNet blocks (SPEC4.md): sanity gate, P6.1 concentration of g-bar, P6.3 shift consistency, P6.4 connection-scale slope, and the
(report-only) tower for a 2-px shift and a 15-degree rotation.  True memorisers vs matched clean models on the same 5,000 images.
Neighbouring work: LieGG (Moskalev et al., NeurIPS 2022); shift consistency: Zhang 2019, Azulay & Weiss 2019; consistency regularisation: Laine & Aila 2017."""
import sys, os, json, glob, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.resnet_cifar import make_model
from systems.resnet_subset import subset_data
from analysis.resnet_depth import all_layers, LAYERS
from analysis.run3_blockP64 import conn_slope
from whiteboxtopo.metric import pullback_input
DEV = 'cuda'


def load(ck, seed):
    m = make_model(seed).to(DEV); m.load_state_dict(torch.load(ck, map_location=DEV)); m.eval(); return m


def concentration(m, x):
    J = pullback_input(lambda z: m(z).float(), torch.as_tensor(x).to(DEV), batch=50).reshape(len(x), 10, -1).double().cpu()
    sv = torch.linalg.svdvals(J.reshape(-1, J.shape[-1])); lam = (sv ** 2).numpy()
    per = torch.linalg.eigvalsh(J @ J.transpose(1, 2)).clamp_min(0); pr_in = ((per.sum(1) ** 2) / (per ** 2).sum(1).clamp_min(1e-300)).mean().item()
    return float(lam.sum() ** 2 / (lam ** 2).sum()), float(pr_in)


def shift_consistency(m, x):
    X = torch.as_tensor(x).to(DEV); out = []
    with torch.no_grad():
        base = torch.cat([m(X[i:i + 500]).argmax(1) for i in range(0, len(X), 500)])
        for sh, dim in ((2, 3), (-2, 3), (2, 2), (-2, 2)):
            Xs = torch.roll(X, sh, dim); out.append(float((torch.cat([m(Xs[i:i + 500]).argmax(1) for i in range(0, len(X), 500)]) == base).float().mean()))
    return float(np.mean(out)), out


def tower(m, x):
    import torchvision.transforms.functional as TF
    X = torch.as_tensor(x[:300]); res = {}
    for name, Xt in (('shift_2px', torch.roll(X, 2, 3)), ('rotate_15deg', TF.rotate(X, 15.0))):
        A0, l0 = all_layers(m, X.numpy()); A1, l1 = all_layers(m, Xt.numpy()); A0['logits'], A1['logits'] = l0, l1; res[name] = {}
        for L in LAYERS + ['logits']:
            perm = np.random.default_rng(0).permutation(len(X)); res[name][L] = float(np.linalg.norm(A1[L] - A0[L], axis=1).mean() / np.linalg.norm(A0[L][perm] - A0[L], axis=1).mean())
    return res


def main():
    out = {}; t00 = time.time()
    for mode in ('clean', 'scrambled'):
        for s in (1, 2):
            run = f'r4_resnet_{mode}_s{s}'
            if not os.path.exists(f'runs/{run}/history.json'):
                print('[r4] missing', run); continue
            t0 = time.time(); H = json.load(open(f'runs/{run}/history.json')); h = H['history'][-1]; x, y, _, _ = subset_data(mode, s)
            m = load(sorted(glob.glob(f'runs/{run}/ckpt_[0-9]*.pt'))[-1], s); m0 = load(f'runs/{run}/ckpt_init.pt', s)
            sub = np.random.default_rng(0).choice(len(x), 500, replace=False); pr, pri = concentration(m, x[sub]); pr0, pri0 = concentration(m0, x[sub])
            pr1, _ = concentration(make_model(s + 100).to(DEV).eval(), x[sub]); sc, scs = shift_consistency(m, x); sc0, _ = shift_consistency(m0, x)
            rng = np.random.default_rng(0); sel = np.concatenate([rng.choice(np.where(y == c)[0], 150, replace=False) for c in range(10)]); A, _ = all_layers(m, x[sel]); clouds = []
            for L in LAYERS:
                X = A[L]
                if X.shape[1] > 1024:
                    Xc = X - X.mean(0); U, S, _ = np.linalg.svd(Xc, full_matrices=False); X = (U[:, :256] * S[:256]).astype(np.float32)
                clouds.append(X)
            out[run] = dict(mode=mode, seed=s, epochs=H['meta']['epochs'], train_acc=h['train_acc'], test_acc=h['test_acc'],
                            P6_1=dict(PR=pr, PR_untrained=pr0, C=1 - pr / pr0, C_second_untrained=1 - pr1 / pr0, PR_per_input=pri, PR_per_input_untrained=pri0, C_per_input=1 - pri / pri0),
                            P6_3=dict(shift_consistency=sc, per_shift=scs, untrained=sc0), P6_4=conn_slope(clouds, y[sel]), tower=tower(m, x), wall_s=time.time() - t0)
            r = out[run]; print(f"[r4] {run}: {r['epochs']} ep, acc {r['train_acc']:.3f}/{r['test_acc']:.3f} | C {r['P6_1']['C']:+.3f} (PR {pr:.0f} vs {pr0:.0f}; per-input C {r['P6_1']['C_per_input']:+.3f}) | "
                             f"shift consistency {sc:.3f} (untrained {sc0:.3f}) | conn slope {r['P6_4']['slope']:+.3f} {[round(v, 2) for v in r['P6_4']['per_layer']]} ({r['wall_s']:.0f}s)", flush=True)
            json.dump(out, open('results/run4/resnet.json', 'w'), indent=1, default=float); del m, m0; torch.cuda.empty_cache()
    print(f'wall {time.time()-t00:.0f}s')


if __name__ == '__main__':
    main()
