"""EXPERIMENT B — P1 Concept crystallization deep dive + topological robustness (FPL).
ResNet-18/CIFAR-10, 60 epochs, 2 seeds, ckpt every epoch. Per class per epoch: residual SV,
mean FPL of top-3 longest-lived B1 features. t*_c = first epoch residual SV<delta for 3 consec.
Correlations: crystallization rank vs class similarity; FPL vs per-class test acc (NEW);
residual SV vs accuracy drop under Gaussian noise. Saves model ckpt at epoch 60 for Exp C."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy.stats import spearmanr
import torchvision, torchvision.transforms as T
import whitebox as wb, exp_vision as V

DEVICE="cuda"; PCA_D=30; DELTA=0.5

def top3_fpl(D):
    if D.size==0: return 0.0
    lt=np.sort(D[:,1]-D[:,0])[::-1][:3]; return float(np.mean(lt)) if len(lt) else 0.0

@torch.no_grad()
def per_class_acc(model,loader,sigma=0.0):
    model.eval(); cor=np.zeros(10); tot=np.zeros(10)
    for x,y in loader:
        x=x.to(DEVICE)
        if sigma>0: x=x+sigma*torch.randn_like(x)
        p=model(x).argmax(1).cpu().numpy(); y=y.numpy()
        for c in range(10):
            m=y==c; cor[c]+=(p[m]==c).sum(); tot[c]+=m.sum()
    return cor/np.maximum(tot,1)

def run_seed(seed,epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    train,test,probe_src=V.load_cifar10()
    probe_set,probe_labels=V.build_probe(probe_src,50,seed=0)
    tl=DataLoader(train,128,shuffle=True,num_workers=4,pin_memory=True,drop_last=True,persistent_workers=True)
    testl=DataLoader(test,512,shuffle=False,num_workers=2,pin_memory=True)
    pl=DataLoader(probe_set,256,shuffle=False,num_workers=2)
    model=V.ResNet18().to(DEVICE)
    opt=torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); scaler=torch.amp.GradScaler("cuda")
    a0=V.extract_activations(model,pl,DEVICE); pca=wb.fit_pca(a0["fc"],n_components=PCA_D)
    sv={c:[] for c in range(10)}; fpl={c:[] for c in range(10)}; prevD={c:None for c in range(10)}
    t0=time.time()
    for ep in range(0,epochs+1):
        if ep>0:
            model.train()
            for x,y in tl:
                x=x.to(DEVICE);y=y.to(DEVICE); opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda",dtype=torch.float16):
                    loss=F.cross_entropy(model(x),y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sch.step()
        ac=V.extract_activations(model,pl,DEVICE)["fc"]
        for c in range(10):
            Xc=ac[probe_labels==c]; D=wb.persistence_diagrams(Xc,pca,maxdim=1)
            sv[c].append(0.0 if prevD[c] is None else wb.structural_velocity(D[1],prevD[c]))
            fpl[c].append(top3_fpl(D[1])); prevD[c]=D[1]
        if ep%10==0: print(f"[B s{seed} ep{ep}] ({time.time()-t0:.0f}s)",flush=True)
    # t*_c: first epoch residual SV<DELTA for 3 consec
    tcs={}
    for c in range(10):
        s=sv[c]; tcs[c]=epochs
        for i in range(len(s)-2):
            if s[i]<DELTA and s[i+1]<DELTA and s[i+2]<DELTA: tcs[c]=i; break
    test_acc=per_class_acc(model,testl,0.0); noisy_acc=per_class_acc(model,testl,0.1)
    resid_sv={c:sv[c][-1] for c in range(10)}
    post_fpl={c:float(np.mean(fpl[c][tcs[c]:])) if tcs[c]<len(fpl[c]) else fpl[c][-1] for c in range(10)}
    # save model ckpt for Exp C (seed 1)
    return {"seed":seed,"epochs":epochs,"sv":sv,"fpl":fpl,"tcs":tcs,
            "test_acc":test_acc.tolist(),"noisy_acc":noisy_acc.tolist(),
            "resid_sv":resid_sv,"post_fpl":post_fpl},model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--ckpt",required=True)
    ap.add_argument("--epochs",type=int,default=60); a=ap.parse_args()
    runs=[]; last_model=None
    for seed in [0,1]:
        r,model=run_seed(seed,a.epochs); runs.append(r)
        if seed==1: last_model=model
    torch.save(last_model.state_dict(),a.ckpt); print(f"[B] saved ckpt {a.ckpt}",flush=True)
    # aggregate across seeds
    tcs=np.array([[r["tcs"][c] for c in range(10)] for r in runs]).mean(0)
    test_acc=np.array([r["test_acc"] for r in runs]).mean(0)
    noisy_acc=np.array([r["noisy_acc"] for r in runs]).mean(0)
    post_fpl=np.array([[r["post_fpl"][c] for c in range(10)] for r in runs]).mean(0)
    resid_sv=np.array([[r["resid_sv"][c] for c in range(10)] for r in runs]).mean(0)
    acc_drop=test_acc-noisy_acc
    rho_fpl,p_fpl=spearmanr(post_fpl,test_acc)
    rho_frag,p_frag=spearmanr(resid_sv,acc_drop)
    # class similarity proxy: final test acc rank as distinctness; crystallization rank vs distinctness
    rho_cryst,p_cryst=spearmanr(tcs,-test_acc)
    out={"runs":runs,"tcs_mean":tcs.tolist(),"test_acc":test_acc.tolist(),
         "noisy_acc":noisy_acc.tolist(),"post_fpl":post_fpl.tolist(),"resid_sv":resid_sv.tolist(),
         "acc_drop":acc_drop.tolist(),
         "spearman_FPL_vs_acc":[float(rho_fpl),float(p_fpl)],
         "spearman_SV_vs_fragility":[float(rho_frag),float(p_frag)],
         "spearman_cryst_vs_distinct":[float(rho_cryst),float(p_cryst)]}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,"w"),indent=2)
    print(f"[B done] FPL~acc rho={rho_fpl:.3f}(p={p_fpl:.3f}) SV~fragility rho={rho_frag:.3f}(p={p_frag:.3f})",flush=True)

if __name__=="__main__": main()
