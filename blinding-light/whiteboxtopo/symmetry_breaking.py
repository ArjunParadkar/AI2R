"""Symmetry-breaking score b_z (SPEC3 S5).

The input of the network is a stack of blocks ("slots"), shape (n_slots, d) per example.  One block is the
nuisance z.  If the function were invariant under arbitrary changes of that block, the Jacobian of the output
with respect to it would vanish.  The breaking score is the share of the squared Frobenius norm of the
input-output Jacobian that falls in the nuisance block:

    b_z = E_x [ ||J_z(x)||_F^2 / ||J(x)||_F^2 ]              (mean of per-input ratios; primary)
    b_z_rom = E_x ||J_z(x)||_F^2 / E_x ||J(x)||_F^2          (ratio of means; = hint share of trace E[g])

where g(x) = J(x)^T J(x) is the output-pullback metric on the slot space.  NOTE: like any ablation-style
statistic, b_z needs to be TOLD which block is the nuisance.  It discovers nothing.

J is computed with torch.func (jacrev + vmap) in batches of <= 256.
"""
from __future__ import annotations
import torch
from torch.func import jacrev, vmap

__all__ = ['slot_jacobians', 'breaking_score']


def slot_jacobians(f, slots, batch=256):
    """f: (n_slots, d) -> (k,) for ONE example (or a tuple of such outputs).  slots: (N, n_slots, d).
    Yields, per batch, J of shape (B, k, n_slots, d) (or a tuple of them)."""
    assert batch <= 256
    jf = vmap(jacrev(f))
    for i in range(0, len(slots), batch):
        yield jf(slots[i:i + batch])


def _one(J, z, acc):
    # J: (B, k, n_slots, d)
    per_slot = (J.double() ** 2).sum(dim=(1, 3))                  # (B, n_slots)
    tot = per_slot.sum(1)
    ratio = torch.where(tot > 0, per_slot[:, z] / tot.clamp_min(1e-300), torch.zeros_like(tot))
    acc['ratio_sum'] += ratio.sum(); acc['n'] += len(ratio); acc['n_zero'] += int((tot <= 0).sum())
    acc['slot_sum'] += per_slot.sum(0)
    acc['ratios'].append(ratio.float().cpu())
    if acc['G'] is not None:
        Jf = J.double().reshape(J.shape[0], J.shape[1], -1)
        acc['G'] += torch.einsum('bki,bkj->ij', Jf, Jf)


def breaking_score(f, slots, nuisance=-1, batch=256, return_metric=False, names=None):
    """Breaking score of the nuisance block.

    f        one-example function (n_slots, d) -> (k,), or -> tuple of (k_l,) tensors (one per layer target).
    slots    (N, n_slots, d) tensor of embedded inputs.
    nuisance index of the nuisance block.
    Returns a dict (or a list of dicts, one per target, if f returns a tuple; `names` labels them):
      b_z            mean over inputs of ||J_z||^2 / ||J||^2
      b_z_rom        ratio of means
      slot_share     share of E||J||^2 in each block (ratio of means), sums to 1
      n, n_zero      inputs used / inputs with J == 0 (their ratio is counted as 0)
      ratios         per-input ratios (float32 tensor, cpu)
      metric         E_x[g(x)], (n_slots*d, n_slots*d), if return_metric
    """
    n_slots, d = slots.shape[1], slots.shape[2]
    z = nuisance % n_slots
    accs = None
    for J in slot_jacobians(f, slots, batch):
        Js = J if isinstance(J, (tuple, list)) else (J,)
        if accs is None:
            accs = [dict(ratio_sum=torch.zeros((), dtype=torch.float64, device=slots.device), n=0, n_zero=0,
                         slot_sum=torch.zeros(n_slots, dtype=torch.float64, device=slots.device), ratios=[],
                         G=torch.zeros(n_slots * d, n_slots * d, dtype=torch.float64, device=slots.device) if return_metric else None)
                    for _ in Js]
        for Jl, acc in zip(Js, accs):
            _one(Jl, z, acc)
    out = []
    for i, acc in enumerate(accs):
        tot = acc['slot_sum'].sum()
        share = (acc['slot_sum'] / tot) if tot > 0 else torch.zeros_like(acc['slot_sum'])
        r = dict(b_z=float(acc['ratio_sum'] / acc['n']), b_z_rom=float(share[z]), slot_share=[float(s) for s in share],
                 n=acc['n'], n_zero=acc['n_zero'], ratios=torch.cat(acc['ratios']))
        if names is not None:
            r['name'] = names[i]
        if return_metric:
            r['metric'] = (acc['G'] / acc['n']).cpu()
        out.append(r)
    return out if len(out) > 1 or names is not None else out[0]
