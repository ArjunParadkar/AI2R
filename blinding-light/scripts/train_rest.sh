#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"
source .venv/bin/activate
mkdir -p runs logs
for s in 0 1 2; do
  python systems/ring_rnn.py --seed $s --steps 4000 --outdir runs/ring_s$s > logs/ring_s$s.log 2>&1 &
done
python systems/rot_mnist.py --seed 0 --epochs 10 --outdir runs/rotmnist_s0 > logs/rotmnist_s0.log 2>&1 &
wait
echo "RING+ROTMNIST DONE"
python systems/resnet_cifar.py --mode clean     --seed 0 --epochs 20 --outdir runs/resnet_clean_s0     > logs/resnet_clean_s0.log 2>&1
python systems/resnet_cifar.py --mode scrambled --seed 0 --epochs 20 --outdir runs/resnet_scrambled_s0 > logs/resnet_scrambled_s0.log 2>&1
echo "ALL REST DONE"
