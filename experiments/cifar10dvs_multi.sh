#!/bin/sh
python ./src/cifar10dvs_multiple_2.py -T 20 -device cuda:0 -b 16 -epochs 256 -data-dir ./datasets/CIFAR10DVS/ -out-dir ./results/cifar10dvs/ -amp -cupy -opt adam -lr 0.001 -j 2 -augment -plif -loss-rate 0.5 -exp-number 0
