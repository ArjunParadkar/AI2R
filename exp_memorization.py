"""Validation 3 (memorization): ResNet-18 on CIFAR-10 with label noise.

WHITEBOX §4.3: noise levels {0%, 50%, 100%}, 60 epochs each (we use 25 to fit budget).
Hypothesis 4: random labels -> no SV peak, monotonic TC, no high-FPL.
"""
import argparse
from exp_vision import train_one_seed

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--ckpt-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    layers = ["layer1", "layer2", "layer3", "fc"]
    for noise in [0.0, 0.5, 1.0]:
        train_one_seed(seed=args.seed, epochs=args.epochs, ckpt_every=args.ckpt_every,
                       layers=layers, label_noise=noise, tag=f"v3_noise{int(noise*100)}")
