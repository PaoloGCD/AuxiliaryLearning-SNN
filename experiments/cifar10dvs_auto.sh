#!/bin/sh
python python ./src/cifar10dvs_auto.py -T 20 -device cuda:0 -b 8 -epochs 256 -data-dir ./datasets/CIFAR10DVS/ -out-dir ./results/cifar10dvs-auto/ -amp -opt adam -lr 0.001 -j 2 -augment -plif -aux-net linear -exp-number 0
