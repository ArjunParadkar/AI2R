import sys,glob,numpy as np,torch; sys.path.insert(0,'.')
from systems.ring_rnn import RingRNN
from whiteboxtopo.topology import persistence,circular_coordinates,circular_correlation
m=RingRNN(seed=0).cuda(); m.load_state_dict(torch.load(sorted(glob.glob('runs/ring_s0/ckpt_*.pt'))[-1])); m.eval()
rng=np.random.default_rng(7)
for sigma,settle in [(0.3,0),(0.6,0),(0.3,5),(0.6,5)]:
    v=torch.as_tensor(rng.normal(0,sigma,size=(800,25)),dtype=torch.float32,device='cuda')
    with torch.no_grad():
        h=m(v)
        for _ in range(settle): h=m.step(h,torch.zeros(800,device='cuda'))
        out=m.readout(h).cpu().numpy()
    th=np.mod(torch.cumsum(v,1)[:,-1].cpu().numpy(),2*np.pi)
    X=h.cpu().numpy().astype(np.float32)
    s=np.linalg.svd(X-X.mean(0),compute_uv=False); ev=s**2/(s**2).sum()
    r=persistence(X); h1=r['dgms'][1]; life=np.sort(h1[:,1]-h1[:,0]); j=int(np.argmax(h1[:,1]-h1[:,0]))
    cc=circular_coordinates(X,h1,r['cocycles'][1],D=r['D'],class_idx=j)
    print('sigma %.1f settle %d | PCA top4 %s | heading R=%.2f | life %.3f vs %.3f | r=%.3f | decode err %.3f'%(
      sigma,settle,np.round(ev[:4],3),abs(np.exp(1j*th).mean()),life[-1],life[-2],circular_correlation(cc['theta'],th),
      np.abs(np.angle(np.exp(1j*(np.arctan2(out[:,1],out[:,0])-th)))).mean()),flush=True)
