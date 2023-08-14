# TM-SNN: Threshold Modulated Spiking Neural Network for Multi-task Learning

This repository contains code implementation for the paper Cachi PG, Ventura S, Cios KJ. Improving Spiking Neural Network Performance with Auxiliary Learning. Machine Learning and Knowledge Extraction. 2023; 5(3):1010-1022. https://doi.org/10.3390/make5030052. The code implements the use of auxiliary learning for improving training performance of Spiking Neural Networks. Tests are performed on neuromorphic DVS-CIFAR10 and DVS128-Gesture datasets. The results indicate that training with auxiliary learning tasks improves their accuracy, albeit slightly. Different scenarios, including manual and automatic combination losses using implicit differentiation, are explored to analyze the usage of auxiliary tasks.

## Installation

### Prerequisites

The code runs in Python3.9 using the [SpikingJelly](https://github.com/fangwei123456/spikingjelly) neuromorphic framework and [auxilearn](https://github.com/AvivNavon/AuxiLearn) package for implicit differentiation.

### Clone

Clone this repo to your local machine using `https://github.com/PaoloGCD/AuxiliaryLearning-SNN.git`

### Dataset

The experiments are performed on the CIFAR10DVS and DVSGesture-128 datasets. Download and extract them to '.datasets' folder from https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671/2 and https://research.ibm.com/interactive/dvsgesture/, respectively.

## Running the tests

To train the SNN for single task classification (base-case) for the CIFAR10DVS dataset, execute:

```shell
$ sh ./experiments/cifar10dvs_single.sh
```

To train the SNN using multiple auxiliary tasks, execute:

```shell
$ sh ./experiments/cifar10dvs_multi.sh
```

To train the SNN using multiple auxiliary tasks with implicit differentiation, execute:

```shell
$ sh ./experiments/cifar10dvs_auto.sh
```

## Authors

* **Paolo G. Cachi** - *Virginia Commonwealth University* - USA

## Acknowledgments

* The code is based on [SpikingJelly's classification example](https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/classify_dvsg.html). 
