"""Head-direction integration task (1-D continuous attractor; Cueva & Wei 2018 / Sorscher et al. 2019 style).
Input (B,T,3): [omega_t, cos th0 * [t==0], sin th0 * [t==0]]. Target (B,T): heading bin in 0..K-1.
Metric = e4lib.tasks.metric (mean log-prob of the correct bin) so E4's discovery pipeline runs unchanged."""
import numpy as np, torch

class HDTask:
    def __init__(self, T=100, K=16, sigma=0.08, tau=10.0, clip=np.pi / 4, shuffled_targets=False):
        self.T, self.K, self.sigma, self.tau, self.clip = T, K, sigma, tau, clip
        self.shuffled = shuffled_targets; self.chance = 1.0 / K; self.n_classes = K; self.S = T

    def batch(self, n, seed, split="eval"):
        rng = np.random.default_rng(int(seed) * 7919 + (0 if split == "train" else 1_000_003))
        th0 = rng.uniform(-np.pi, np.pi, n)
        om = np.zeros((n, self.T)); v = rng.normal(0, self.sigma, n)
        for t in range(self.T):                                       # OU angular velocity
            v = v - v / self.tau + rng.normal(0, self.sigma * np.sqrt(2 / self.tau), n)
            om[:, t] = np.clip(v, -self.clip, self.clip)
        th = th0[:, None] + np.cumsum(om, 1)
        x = np.zeros((n, self.T, 3), np.float32); x[:, :, 0] = om; x[:, 0, 1] = np.cos(th0); x[:, 0, 2] = np.sin(th0)
        y = (np.floor((th % (2 * np.pi)) / (2 * np.pi) * self.K)).astype(np.int64) % self.K
        if self.shuffled:                                             # control: no integrator exists for these targets
            y = rng.permutation(y.ravel()).reshape(y.shape)
        return torch.tensor(x), torch.tensor(y)

def accuracy_tail(logits, targets, tail=50, tol=0):
    """Exact-bin accuracy over the last `tail` steps; tol=1 counts +-1 bin (circular) as correct (reported, not the guard)."""
    K = logits.shape[-1]; d = (logits[:, -tail:].argmax(-1) - targets[:, -tail:]) % K; d = torch.minimum(d, K - d)
    return (d <= tol).float().mean().item()

CURRICULUM = ((400, 20), (800, 40), (1200, 70))   # (until step, T); T = 100 afterwards

def curriculum_T(step, T_full=100):
    for until, T in CURRICULUM:
        if step < until: return T
    return T_full
