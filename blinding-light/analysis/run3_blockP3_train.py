"""Run 3, P3.3 (SPEC3.md): the b_z-triggered interrupt.  Adapted from analysis/run2_blockD34.py.
Trigger = b_z above its clean-null threshold (results/run3/blockP3.json) at a 500-step check.  Penalty = run-1 invariance_penalty,
lambda = 3, switched on ONCE at the first firing (per seed), never escalated.  12000 steps.

All seeds are trained IN ONE PROCESS, stacked with torch.func.vmap (host RAM was at 2.3 GB available with swap full when this was
launched; five CUDA processes cost ~1.35 GB RSS each).  The seeds do not interact: the loss is the sum of per-seed losses, AdamW is
elementwise, each seed has its own data (build_data(seed)), its own init (Z113Transformer(seed)), its own lambda and its own penalty draws."""
import sys, os, json, time, argparse, copy
import torch, torch.nn.functional as F
from torch.func import functional_call, vmap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.z113 import Z113Transformer, build_data, P, NULL, EQ
from analysis.run2_blockD12 import probe
from analysis.z113_interrupt import invariance_penalty
from analysis.run3_blockP3 import slot_fn, slots_of
from whiteboxtopo.symmetry_breaking import breaking_score


class Shim:
    """What invariance_penalty needs from a model: .W_E.device and __call__."""
    def __init__(self, base, params):
        self.base, self.params, self.W_E = base, params, params['W_E']

    def __call__(self, toks):
        return functional_call(self.base, self.params, (toks,))


def run(seeds, steps=12000, lam0=3.0, check_every=500, graph=False, tag='r3_bz'):
    thr = json.load(open('results/run3/blockP3.json'))['threshold']['threshold']
    S = len(seeds); T = lambda x: torch.as_tensor(x, device='cuda')
    D = [build_data(s, 'shortcut', frac=0.5) for s in seeds]
    st = lambda k: torch.stack([T(d[k]) for d in D])
    xtr, ytr, xte, xdec, yte, worst, leaky = st('train_toks'), st('train_y'), st('test_toks'), st('dec_toks'), st('test_y'), st('dec_worst'), st('train_leaky')
    probes = [probe(s, 'shortcut', 0.5)[0] for s in seeds]
    models = [Z113Transformer(seed=s) for s in seeds]; base = Z113Transformer(seed=0).cuda()
    params = {k: torch.stack([dict(m.named_parameters())[k].detach() for m in models]).cuda().requires_grad_(True) for k, _ in base.named_parameters()}
    fwd = vmap(lambda p, x: functional_call(base, p, (x,)))
    pen = vmap(lambda p: invariance_penalty(Shim(base, p)), randomness='different')
    with torch.no_grad():                                                     # the stacked forward IS the per-seed forward
        for i, m in enumerate(models):
            assert torch.allclose(fwd(params, xte)[i], m.cuda()(xte[i]), atol=1e-5); m.cpu()
    torch.manual_seed(1000 + sum(seeds))
    opt = torch.optim.AdamW(list(params.values()), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98), capturable=graph)
    lam = torch.zeros(S, device='cuda'); hist = [[] for _ in seeds]; t0, t_bz = time.time(), 0.0
    STATE = f'runs/{tag}_s{seeds[0]}/stacked_state.pt'; start = 0; names = list(params)
    if os.path.exists(STATE):                                                 # resume after a CUDA OOM on the shared GPU (state saved at every check, before the check is logged)
        st8 = torch.load(STATE, map_location='cuda')
        with torch.no_grad():
            for k in names:
                params[k].copy_(st8['params'][k])
        opt.load_state_dict(st8['opt']); lam = st8['lam'].cuda(); hist = st8['hist']; start = st8['step']; t0 -= st8['wall_s']; t_bz = st8['t_bz']
    logs = [open(f'logs/{tag}_s{s}.log', 'a' if start else 'w') for s in seeds]
    for s in seeds:
        os.makedirs(f'runs/{tag}_s{s}', exist_ok=True)
    g = None

    def one_step(zero=True):
        if zero:
            opt.zero_grad(set_to_none=True)
        # same gradient as one backward of sum_i [CE_i + lam_i * pen_i]; split into passes only to bound peak GPU memory.
        # With graph=True the penalty is always evaluated (lam_i = 0 contributes exactly zero gradient) so that the step has no control flow.
        F.cross_entropy(fwd(params, xtr).reshape(-1, P), ytr.reshape(-1), reduction='none').reshape(S, -1).mean(1).sum().backward()
        for j in range(0, S, 2):
            if graph or (lam[j:j + 2] > 0).any():
                (lam[j:j + 2] * pen({k: v[j:j + 2] for k, v in params.items()})).sum().backward()
        opt.step()

    for step in range(start, steps + 1):
        if step % check_every == 0 and not (start and step == start):
            with torch.no_grad():
                acc = lambda x, y: fwd(params, x).argmax(-1) == y
                ok, ltr, lte = acc(xdec, yte), acc(xtr, ytr), acc(xte, yte)
                al = acc(torch.cat([xtr[..., :2], torch.full_like(xtr[..., 2:3], NULL), xtr[..., 3:]], -1), ytr)
            for i, s in enumerate(seeds):
                t1 = time.time(); m = models[i]
                m.load_state_dict({k: v[i].detach().cpu() for k, v in params.items()}); m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
                b = breaking_score(slot_fn(m), slots_of(m, probes[i]), nuisance=2, batch=50); t_bz += time.time() - t1
                rec = dict(step=step, dec_acc=ok[i].float().mean().item(), worst=ok[i][worst[i]].float().mean().item(), test_acc=lte[i].float().mean().item(),
                           train_acc=ltr[i].float().mean().item(), reliance=(ltr[i][leaky[i]].float().mean() - al[i][leaky[i]].float().mean()).item(),
                           b_z=b['b_z'], b_z_rom=b['b_z_rom'], fires=bool(b['b_z'] > thr), lam=float(lam[i]))
                if rec['fires'] and lam[i] == 0:
                    lam[i:i + 1].fill_(lam0); rec['switched_on'] = True               # once; never escalated
                hist[i].append(rec)
                print(f"[bz s{s}] {step:6d} dec {rec['dec_acc']:.3f} test {rec['test_acc']:.3f} worst {rec['worst']:.3f} reliance {rec['reliance']:.3f} b_z {rec['b_z']:.4f} (thr {thr:.4f}) "
                      f"fires {int(rec['fires'])} lam {float(lam[i]):g}  [{time.time()-t0:.0f}s]", file=logs[i], flush=True)
                json.dump(dict(meta=dict(seed=s, threshold=thr, trigger='b_z > threshold', penalty='run-1 invariance_penalty', lam0=lam0, escalation='none', steps=steps,
                                         stacked_with=seeds, bz_device='cpu', cuda_graph=graph, wall_s=time.time() - t0, bz_s_all_seeds=t_bz), history=hist[i]),
                          open(f'runs/{tag}_s{s}/history.json', 'w'), indent=1, default=float)
            torch.save(dict(params={k: v.detach() for k, v in params.items()}, opt=opt.state_dict(), lam=lam, hist=hist, step=step, wall_s=time.time() - t0, t_bz=t_bz), STATE + '.tmp')
            os.replace(STATE + '.tmp', STATE)
        if step == steps:
            break
        if g is not None:
            g.replay()
        else:
            one_step()
            if graph and step - start == 2:                                   # steps start..start+2 ran eagerly (they are real steps); capture the step once, then replay it
                torch.cuda.synchronize(); g = torch.cuda.CUDAGraph(); opt.zero_grad(set_to_none=True)
                with torch.cuda.graph(g):
                    one_step(zero=False)
        if step % 100 == 0:
            torch.cuda.synchronize(); print(f'[stacked] step {step} {time.time()-t0:.0f}s', flush=True)
    for i, s in enumerate(seeds):
        torch.save({k: v[i].detach().cpu() for k, v in params.items()}, f'runs/{tag}_s{s}/final.pt')
        print(f'[bz s{s}] done {time.time()-t0:.0f}s', file=logs[i], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--steps', type=int, default=12000); ap.add_argument('--graph', type=int, default=0); ap.add_argument('--tag', default='r3_bz')
    a = ap.parse_args(); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run([int(s) for s in a.seeds.split(',')], steps=a.steps, graph=bool(a.graph), tag=a.tag)
