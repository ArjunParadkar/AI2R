import sys,torch,numpy as np; sys.path.insert(0,'.')
from analysis.z113_interrupt import *
import torch.nn.functional as F
d=build_data(0,'shortcut',frac=0.5)
T=lambda x: torch.as_tensor(x,device='cuda')
xtr,ytr=T(d['train_toks']),T(d['train_y']); xdec=T(d['dec_toks']); yte=T(d['test_y'])
torch.manual_seed(0); m=Z113Transformer(seed=0).cuda()
opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1.0,betas=(0.9,0.98))
for step in range(4001):
    opt.zero_grad(); ce=F.cross_entropy(m(xtr),ytr); pen=invariance_penalty(m,xtr)
    (ce+ (3.0*pen if step>=300 else 0*pen)).backward(); opt.step()
    if step%500==0:
        with torch.no_grad(): dec=(m(xdec).argmax(-1)==yte).float().mean().item(); te=(m(T(d['test_toks'])).argmax(-1)==yte).float().mean().item(); tr=(m(xtr).argmax(-1)==ytr).float().mean().item()
        print('step %4d ce %.4f pen %.5f train %.3f test %.3f dec %.3f'%(step,ce.item(),pen.item(),tr,te,dec),flush=True)
