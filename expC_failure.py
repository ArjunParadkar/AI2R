"""EXPERIMENT C — P4 (failure neighborhoods) + P5 (fragility) on Exp-B epoch-60 model.
P4: correct vs misclassified test inputs -> layer3 activation clouds -> bootstrap TC & B1
    distributions -> Mann-Whitney U. P5: per-class residual SV (from B) vs accuracy drop
    under Gaussian noise -> Spearman. Negatives reported clearly."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import mannwhitneyu, spearmanr
import whitebox as wb, exp_vision as V

DEVICE="cuda"

@torch.no_grad()
def collect(model,loader):
    model.eval(); A3=[];Afc=[];corr=[]
    for x,y in loader:
        x=x.to(DEVICE);y=y.to(DEVICE)
        out,acts=model(x,return_acts=True)
        a3=F.adaptive_avg_pool2d(acts["layer3"],1).flatten(1)
        A3.append(a3.cpu().numpy()); Afc.append(acts["fc"].cpu().numpy())
        corr.append((out.argmax(1)==y).cpu().numpy())
    return np.concatenate(A3),np.concatenate(Afc),np.concatenate(corr)

def boot_tc_b1(X,pca,n=40,size=150,seed=0):
    rng=np.random.default_rng(seed); tcs=[];b1s=[]
    for _ in range(n):
        idx=rng.choice(len(X),min(size,len(X)),replace=False)
        D=wb.persistence_diagrams(X[idx],pca,maxdim=1)
        tcs.append(wb.topological_complexity(D[1]))
        sc=np.median([(p[0]+p[1])/2 for p in D[1]]) if D[1].size else 0.0
        b1s.append(wb.betti_at_scale(D[1],sc))
    return np.array(tcs),np.array(b1s)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--bjson",required=True)
    ap.add_argument("--out",required=True); a=ap.parse_args()
    model=V.ResNet18().to(DEVICE); model.load_state_dict(torch.load(a.ckpt,map_location=DEVICE))
    _,test,_=V.load_cifar10(); testl=DataLoader(test,512,shuffle=False,num_workers=2)
    A3,Afc,corr=collect(model,testl)
    pcaC=wb.fit_pca(A3[corr],n_components=30);
    # P4: bootstrap TC/B1 for correct vs misclassified at layer3 (shared PCA from correct)
    tcC,b1C=boot_tc_b1(A3[corr],pcaC,seed=1); tcM,b1M=boot_tc_b1(A3[~corr],pcaC,seed=2)
    u_tc,p_tc=mannwhitneyu(tcC,tcM,alternative="two-sided")
    u_b1,p_b1=mannwhitneyu(b1C,b1M,alternative="two-sided")
    # P5: residual SV (from B) vs accuracy drop under noise (from B)
    B=json.load(open(a.bjson))
    resid_sv=np.array(B["resid_sv"]); acc_drop=np.array(B["acc_drop"])
    rho,p=spearmanr(resid_sv,acc_drop)
    out={"P4":{"TC_correct_mean":float(tcC.mean()),"TC_mis_mean":float(tcM.mean()),
               "B1_correct_mean":float(b1C.mean()),"B1_mis_mean":float(b1M.mean()),
               "mannwhitney_TC_p":float(p_tc),"mannwhitney_B1_p":float(p_b1),
               "TC_correct":tcC.tolist(),"TC_mis":tcM.tolist(),"B1_correct":b1C.tolist(),"B1_mis":b1M.tolist(),
               "verdict":"misclassified topologically sparser" if (tcM.mean()<tcC.mean() and p_tc<0.05) else "no significant difference"},
         "P5":{"spearman_SV_vs_accdrop":[float(rho),float(p)],"resid_sv":resid_sv.tolist(),"acc_drop":acc_drop.tolist(),
               "verdict":"residual SV predicts fragility" if (rho>0 and p<0.05) else "no significant prediction"}}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,"w"),indent=2)
    print(f"[C done] P4 TC corr={tcC.mean():.2f} mis={tcM.mean():.2f} p={p_tc:.4f} | P5 rho={rho:.3f} p={p:.3f}",flush=True)

if __name__=="__main__": main()
