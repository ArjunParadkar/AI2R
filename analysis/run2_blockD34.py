"""Run 2, Block D3/D4 (SPEC2.md): the alignment-triggered interrupt.  Trigger = D1 Mapper alignment above the clean-null
threshold (read from results/run2/blockD12.json) at a 500-step check.  Penalty = lambda * E_{x, z != z'} symKL(psi(x,z) || psi(x,z')),
z, z' uniform on Z113 u {NULL}, x uniform random pairs.  D3 is scored at step 6000, D4 over (6000, 12000]."""
import sys, os, json, time, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.z113 import Z113Transformer, build_data, P, NULL, EQ
from analysis.run2_blockD12 import probe, alignment
from analysis.z113_interrupt import invariance_penalty


def align_penalty(model, n=4096):
    dev = model.W_E.device
    ab = torch.randint(0, P, (n, 2), device=dev); eq = torch.full((n, 1), EQ, device=dev)
    z = torch.randint(0, P + 1, (n, 1), device=dev); z2 = (z + torch.randint(1, P + 1, (n, 1), device=dev)) % (P + 1)   # z' != z, values 0..113 (113 = NULL)
    lp = F.log_softmax(model(torch.cat([ab, z, eq], 1)), -1); lq = F.log_softmax(model(torch.cat([ab, z2, eq], 1)), -1)
    return 0.5 * ((lp.exp() * (lp - lq)).sum(1) + (lq.exp() * (lq - lp)).sum(1)).mean()


def run(seed, outdir, sched, steps=12000, lam0=3.0, check_every=500, penalty='align'):
    esc, max_rounds = {'R': (10.0, 3), 'G': (3.0, 5), 'ONE': (1.0, 1)}[sched]        # ONE (run 4, P3.2): lambda = 3 switched on once, never escalated
    V = json.load(open('results/run2/blockD12.json'))['_verdict']; thr = V['threshold']; thr_d = V['threshold_direct']   # A-3: POST-HOC trigger = direct NMI
    os.makedirs(outdir, exist_ok=True); d = build_data(seed, 'shortcut', frac=0.5)
    T = lambda x: torch.as_tensor(x, device='cuda')
    xtr, ytr, xdec, yte, worst = T(d['train_toks']), T(d['train_y']), T(d['dec_toks']), T(d['test_y']), T(d['dec_worst'])
    leaky = T(d['train_leaky']); tk, z, _ = probe(seed, 'shortcut', 0.5); xte_in = T(d['test_toks'])
    torch.manual_seed(seed); model = Z113Transformer(seed=seed).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    lam, rounds, hist, t0 = 0.0, 0, [], time.time()
    for step in range(steps + 1):
        if step % check_every == 0:
            model.eval()
            with torch.no_grad():
                ok = model(xdec).argmax(-1) == yte; ltr = model(xtr).argmax(-1) == ytr; tin = (model(xte_in).argmax(-1) == yte).float().mean().item()
                al = model(torch.cat([xtr[:, :2], torch.full_like(xtr[:, 2:3], NULL), xtr[:, 3:]], 1)).argmax(-1) == ytr
            a, _ = alignment(model, tk, z)
            rec = dict(step=step, dec_acc=ok.float().mean().item(), worst=ok[worst].float().mean().item(), train_acc=ltr.float().mean().item(), test_acc_in_dist=tin,
                       reliance=(ltr[leaky].float().mean() - al[leaky].float().mean()).item(), lam=lam, fires_letter=bool(a['A'] > thr), fires=bool(a['direct_nmi'] > thr_d), **a)
            if rec['fires'] and rounds < max_rounds:
                lam = lam0 if lam == 0 else lam * esc; rounds += 1; rec['escalated_to'] = lam
            hist.append(rec)
            print(f"[align-{sched} s{seed}] {step:6d} dec {rec['dec_acc']:.3f} worst {rec['worst']:.3f} reliance {rec['reliance']:.3f} A {a['A']:.4f} direct {a['direct_nmi']:.3f} (thr {thr_d:.3f}) "
                  f"fires {int(rec['fires'])} lam {lam:g}", flush=True)
            json.dump(dict(meta=dict(seed=seed, sched=sched, thr_letter=thr, thr_direct=thr_d, trigger='direct NMI (POST-HOC, A-3)', lam0=lam0, esc=esc, max_rounds=max_rounds, wall_s=time.time() - t0), history=hist),
                      open(f'{outdir}/history.json', 'w'), indent=1, default=float)
            model.train()
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xtr), ytr)
        if lam > 0:
            loss = loss + lam * (align_penalty(model) if penalty == 'align' else invariance_penalty(model))   # 'run1' = run-1's z vs z+1 and z vs NULL
        loss.backward(); opt.step()
    print(f'[align-{sched} s{seed}] done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--sched', default='R')
    ap.add_argument('--penalty', default='align'); ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args(); tag = a.sched if a.penalty == 'align' else f'{a.sched}run1pen'
    run(a.seed, a.outdir or f'runs/r2_align_{tag}_s{a.seed}', a.sched, steps=a.steps, penalty=a.penalty)
