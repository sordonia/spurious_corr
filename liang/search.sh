#!/bin/bash

python train.py --method "base" --name base --max_epoch 20 --gpus 1

for i in 3 5 10; do
   python train.py --method "reweight" --name reweight_weight$i --max_epoch 20 --gpus 1 --min_reweight_factor $i
done

for rho in "0.5" "0.1" "1"; do
    python train.py --method "sam_min" --name sam_min_rho$rho --max_epoch 20 --gpus 1 --rho $rho
done
