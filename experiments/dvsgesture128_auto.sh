#!/bin/sh
python ./src/dvsgesture_auto.py -T 16 -device cuda:0 -b 8 -epochs 256 -data-dir ./datasets/DVSGesture/ -out-dir ./results/ -opt adam -lr 0.001 -j 2 -augment -plif -exp-number 0
