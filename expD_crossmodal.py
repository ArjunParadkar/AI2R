"""EXPERIMENT D — Cross-modal universality at MATCHED conditions (both from scratch).
Tiny ViT (6L, d=256, 4 heads, ~6M) from scratch on CIFAR-10, 30 epochs. SV trajectory on
layers {0,2,4,5}. Compare normalized-time SV peak to TinyGPT-from-scratch (existing lang.json)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision, torchvision.transforms as T
import whitebox as wb

DEVICE="cuda"; LAYERS=[0,2,4,5]; PCA_D=20

class ViT(nn.Module):
    def __init__(self,d=256,L=6,H=4,patch=4,img=32,ncls=10):
        super().__init__(); np_=(img//patch)**2; self.patch=patch
        self.proj=nn.Conv2d(3,d,patch,patch); self.cls=nn.Parameter(torch.zeros(1,1,d))
        self.pos=nn.Parameter(torch.zeros(1,np_+1,d))
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(d,H,4*d,0.1,batch_first=True,activation="gelu") for _ in range(L)])
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,ncls)
    def forward(self,x,return_hidden=False):
        B=x.shape[0]; h=self.proj(x).flatten(2).transpose(1,2)
        h=torch.cat([self.cls.expand(B,-1,-1),h],1)+self.pos
        hid=[h]
        for blk in self.blocks: h=blk(h); hid.append(h)
        out=self.head(self.norm(h)[:,0])
        if return_hidden: return out,hid
        return out

@torch.no_grad()
def probe_resid(model,loader,layers):
    model.eval(); outs={l:[] for l in layers}
    for x,_ in loader:
        x=x.to(DEVICE); _,hid=model(x,return_hidden=True)
        for l in layers: outs[l].append(hid[l].mean(1).float().cpu().numpy())
    return {l:np.concatenate(v,0) for l,v in outs.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=30)
    a=ap.parse_args(); torch.manual_seed(0); np.random.seed(0)
    mean=[0.4914,0.4822,0.4465]; std=[0.247,0.243,0.262]
    tr=T.Compose([T.RandomCrop(32,4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(mean,std)])
    te=T.Compose([T.ToTensor(),T.Normalize(mean,std)])
    root="/home/god/whitebox_experiments_clone/data"
    train=torchvision.datasets.CIFAR10(root,True,download=False,transform=tr)
    probe_src=torchvision.datasets.CIFAR10(root,True,download=False,transform=te)
    rng=np.random.default_rng(0); tgt=np.array(probe_src.targets); idx=[]
    for c in range(10): idx+=rng.choice(np.where(tgt==c)[0],30,replace=False).tolist()
    pl=DataLoader(Subset(probe_src,idx),256,shuffle=False,num_workers=2)
    tl=DataLoader(train,128,shuffle=True,num_workers=4,pin_memory=True,drop_last=True,persistent_workers=True)
    model=ViT().to(DEVICE); nP=sum(p.numel() for p in model.parameters())
    print(f"[D] ViT {nP/1e6:.1f}M",flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.05); scaler=torch.amp.GradScaler("cuda")
    a0=probe_resid(model,pl,LAYERS); pca={l:wb.fit_pca(a0[l],n_components=PCA_D) for l in LAYERS}
    film={l:[] for l in LAYERS}
    def rec():
        ac=probe_resid(model,pl,LAYERS)
        for l in LAYERS: film[l].append(wb.persistence_diagrams(ac[l],pca[l],maxdim=1))
    rec()
    for ep in range(1,a.epochs+1):
        model.train()
        for x,y in tl:
            x=x.to(DEVICE);y=y.to(DEVICE);opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",dtype=torch.float16): loss=F.cross_entropy(model(x),y)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update()
        rec()
        if ep%5==0: print(f"[D ep{ep}] loss={loss:.3f}",flush=True)
    sv={}
    for l in LAYERS:
        f=film[l]; sv[l]=[0.0]+[wb.structural_velocity(f[i][1],f[i-1][1]) for i in range(1,len(f))]
    # use deepest layer as the SV signal; normalized peak time
    deep=sv[LAYERS[-1]]; peak=int(np.argmax(deep)); norm_peak=peak/max(len(deep)-1,1)
    out={"params_M":nP/1e6,"layers":LAYERS,"SV":{str(l):sv[l] for l in LAYERS},
         "deep_SV":deep,"norm_peak_time":norm_peak,"peak_epoch":peak,"epochs":a.epochs}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,"w"),indent=2)
    print(f"[D done] ViT SV peak at normalized t={norm_peak:.2f} (ep{peak})",flush=True)

if __name__=="__main__": main()
