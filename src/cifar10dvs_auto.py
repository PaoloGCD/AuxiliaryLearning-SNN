import sys
import time
import os
import argparse

import numpy as np

from copy import deepcopy

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.cuda import amp

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based import layer

from augment_cifar10dvs import CIFAR10DVS

import torchvision.transforms as torch_transform

from auxilearn.hypernet import MonoHyperNet, MonoLinearHyperNet, MonoNonlinearHyperNet
from auxilearn.optim import MetaOptimizer


class CIFAR10DVSNet(nn.Module):
    def __init__(self, num_outputs: List = 10, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        self.train_times = 0
        self.num_tasks = len(num_outputs)

        if isinstance(num_outputs, int):
            self.output_list = [num_outputs]
        elif isinstance(num_outputs, list):
            while len(num_outputs) < 5:
                num_outputs.append(1)
            self.output_list = num_outputs
        else:
            raise ValueError("output parameter must be int or list of ints")

        conv = []
        for i in range(4):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))

        self.feature_extraction = nn.Sequential(
            *conv,
            layer.Flatten()
        )

        self.task_classifier_1 = nn.Sequential(
            layer.Dropout(0.5),
            layer.Linear(channels * 8 * 8, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, self.output_list[0] * 10),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

        self.task_classifier_2 = nn.Sequential(
            layer.Dropout(0.5),
            layer.Linear(channels * 8 * 8, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, self.output_list[1] * 10),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

        self.task_classifier_3 = nn.Sequential(
            layer.Dropout(0.5),
            layer.Linear(channels * 8 * 8, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, self.output_list[2] * 10),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

        self.task_classifier_4 = nn.Sequential(
            layer.Dropout(0.5),
            layer.Linear(channels * 8 * 8, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, self.output_list[3] * 10),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

        self.task_classifier_5 = nn.Sequential(
            layer.Dropout(0.5),
            layer.Linear(channels * 8 * 8, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, self.output_list[4] * 10),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

        self.task_classifiers = [self.task_classifier_1,
                                 self.task_classifier_2,
                                 self.task_classifier_3,
                                 self.task_classifier_4,
                                 self.task_classifier_5]

    def forward(self, x: torch.Tensor):
        x = self.feature_extraction(x)
        output = []
        for i in range(self.num_tasks):
            output.append(self.task_classifiers[i](x))
        return output


def main():
    # python dvsgesture_multiple_2.py -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir ./datasets/DVSGesture/
    # -out-dir ./results/1/ -amp -cupy -opt adam -lr 0.001 -j 2

    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-augment', action='store_true', help='use data augmentation')
    parser.add_argument('-plif', action='store_true', help='use plif neurons')
    parser.add_argument('-aux-net', type=str, default='linear', help='linear nonlinear auxiliary net')
    parser.add_argument('-exp-number', type=int, help='save in a subfolder with name exp_number')
    parser.add_argument('-num-tasks',  default=5, type=int, help='save in a subfolder with name exp_number')

    args = parser.parse_args()
    print(args)

    # ============
    # Load dataset
    # ============

    transform = None
    if args.augment:
        transform = torch_transform.Compose([
            torch_transform.RandomAffine(5, translate=(0.125, 0.125)),
            torch_transform.RandomApply([torch_transform.RandomErasing(p=0.25, scale=(0.02, 0.15))], p=0.5)
        ])

    raw_dataset = CIFAR10DVS(
        root=args.data_dir,
        data_type='frame',
        frames_number=args.T,
        split_by='number',
        transform=transform
    )

    split_generator = torch.Generator().manual_seed(42)
    train_set, aux_set, test_set = torch.utils.data.random_split(raw_dataset, [0.875, 0.025, 0.1], generator=split_generator)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    aux_data_loader = torch.utils.data.DataLoader(
        dataset=aux_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    # =====
    # tasks
    # =====

    task_1_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64)
    task_4_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64)
    task_2_labels = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.int64)
    task_3_labels = torch.tensor([0, 1, 2, 3, 4, 3, 5, 4, 0, 1], dtype=torch.int64)
    task_5_labels = torch.tensor([0, 3, 2, 1, 1, 0, 2, 1, 3, 1], dtype=torch.int64)

    all_task_labels = [task_1_labels, task_2_labels, task_3_labels, task_4_labels, task_5_labels]

    num_tasks = args.num_tasks
    task_labels = []
    task_number_classes = []
    for i in range(num_tasks):
        task_labels.append(all_task_labels[i])
        task_number_classes.append(all_task_labels[i].max().item() + 1)

    # =====
    # model
    # =====

    if args.plif:
        neuron_model = neuron.ParametricLIFNode
    else:
        neuron_model = neuron.LIFNode

    net = CIFAR10DVSNet(
        num_outputs=task_number_classes,
        channels=args.channels,
        spiking_neuron=neuron_model,
        surrogate_function=surrogate.ATan(),
        detach_reset=True
    )

    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron_model)

    print(net)

    net.to(args.device)

    # ===============
    # auxiliary model
    # ===============
    auxiliary_mapping = dict(
        linear=MonoLinearHyperNet,
        nonlinear=MonoNonlinearHyperNet,
    )

    auxiliary_config = dict(input_dim=num_tasks, main_task=0, weight_normalization=False)

    if args.aux_net == 'nonlinear':
        auxiliary_config['hidden_sizes'] = 3

    auxiliary_net = auxiliary_mapping[args.aux_net](**auxiliary_config)
    auxiliary_net = auxiliary_net.to(args.device)

    # ==========
    # optimizers
    # ==========

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1
    max_train_acc = -1

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    meta_opt = torch.optim.SGD(
        auxiliary_net.parameters(),
        lr=0.001,
        momentum=.9,
        weight_decay=0.0001
    )

    meta_optimizer = MetaOptimizer(
        meta_optimizer=meta_opt, hpo_lr=1e-4, truncate_iter=3, max_grad_norm=25
    )

    # ==============
    # hypergrad step
    # ==============
    def hyperstep():
        meta_val_loss = .0
        for n_val_step, val_batch in enumerate(aux_data_loader):
            if n_val_step < 1:
                val_data, val_raw_label = val_batch

                val_data = val_data.to(args.device)
                val_data = val_data.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]

                val_labels = [(task_label[val_raw_label]).to(args.device) for task_label in task_labels]
                val_onehot_labels = [F.one_hot(val_labels[i], task_number_classes[i]).float() for i in range(num_tasks)]

                val_predicted_outputs = net(val_data)
                val_predicted_outputs = [val_out.mean(0) for val_out in val_predicted_outputs]

                val_train_losses = [F.mse_loss(val_predicted_outputs[i], val_onehot_labels[i], reduction='none') for i in range(num_tasks)]
                val_train_losses = [val_train_loss.sum(axis=1) for val_train_loss in val_train_losses]

                val_train_losses_stack = torch.stack(val_train_losses, dim=1)

                meta_val_loss += val_train_losses_stack[:, 0].mean(0)

        # inner_loop_end_train_loss, e.g. dL_train/dw
        total_meta_train_loss = 0.
        for n_train_step, train_batch in enumerate(train_data_loader):
            if n_train_step < 1:
                train_meta_data, train_meta_raw_label = train_batch

                train_meta_data = train_meta_data.to(args.device)
                train_meta_data = train_meta_data.transpose(0, 1)

                train_meta_labels = [(task_label[train_meta_raw_label]).to(args.device) for task_label in task_labels]
                meta_onehot_labels = [F.one_hot(train_meta_labels[i], task_number_classes[i]).float() for i in range(num_tasks)]

                train_meta_spikes = net(train_meta_data)
                train_meta_frequencies = [spikes_out.mean(0) for spikes_out in train_meta_spikes]

                train_meta_losses = [F.mse_loss(train_meta_frequencies[i], meta_onehot_labels[i], reduction='none') for i in range(num_tasks)]
                train_meta_losses = [train_meta_loss.sum(axis=1) for train_meta_loss in train_meta_losses]

                train_loss = torch.stack(train_meta_losses, dim=1)

                meta_train_loss = auxiliary_net(train_loss)
                total_meta_train_loss += meta_train_loss
            else:
                break

        # hyperpatam step
        curr_hypergrads = meta_optimizer.step(
            val_loss=meta_val_loss,
            train_loss=total_meta_train_loss,
            aux_params=list(auxiliary_net.parameters()),
            parameters=list(net.parameters()),
            return_grads=True
        )

        return curr_hypergrads

    # ===================
    # load training state
    # ===================

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_train_acc = checkpoint['max_train_acc']
        max_test_acc = checkpoint['max_test_acc']

    if args.exp_number is not None:
        args.out_dir = args.out_dir + f'{args.exp_number}/'
        os.makedirs(args.out_dir[:-1], exist_ok=True)

    out_dir = os.path.join(args.out_dir,
                           f'CIFAR10DVS-T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}_aux{args.aux_net}_tasks{num_tasks}')

    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if args.augment:
        out_dir += '_augment'

    if args.plif:
        out_dir += '_plif'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    # ==========
    # train
    # ==========
    print('Initialize training')
    hypergrad_every = len(train_set) // args.b // 2
    for epoch in range(start_epoch, args.epochs):
        net.train()
        raw_dataset.is_train = True
        train_losses_list = [0 for _ in range(num_tasks)]
        train_accuracies_list = [0 for _ in range(num_tasks)]
        train_samples = 0
        train_start_time = time.time()
        for frame, label in train_data_loader:
            optimizer.zero_grad()

            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]

            true_labels = [(task_label[label]).to(args.device) for task_label in task_labels]
            one_hot_labels = [F.one_hot(true_labels[i], task_number_classes[i]).float() for i in range(num_tasks)]

            if scaler is not None:
                with amp.autocast():
                    predicted_outputs = net(frame)
                    predicted_outputs = [predicted_output.mean(0) for predicted_output in predicted_outputs]

                    train_losses = [F.mse_loss(predicted_outputs[i], one_hot_labels[i], reduction='none') for i in range(num_tasks)]
                    train_losses_sum = [train_loss.sum(axis=1) for train_loss in train_losses]

                    train_loss = torch.stack(train_losses_sum, dim=1)
                    avg_train_losses = train_loss.mean(0)

                    # task weights
                    loss = auxiliary_net(train_loss)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predicted_outputs = net(frame)
                predicted_outputs = [predicted_output.mean(0) for predicted_output in predicted_outputs]

                train_losses = [F.mse_loss(predicted_outputs[i], one_hot_labels[i], reduction='none') for i in range(num_tasks)]
                train_losses_sum = [train_loss.sum(axis=1) for train_loss in train_losses]

                train_loss = torch.stack(train_losses_sum, dim=1)
                avg_train_losses = train_loss.mean(0)

                # task weights
                loss = auxiliary_net(train_loss)

                loss.backward()
                optimizer.step()

            train_samples += label.numel()

            for i in range(num_tasks):
                train_losses_list[i] = train_losses_list[i] + train_losses_sum[i].sum().item()
                train_accuracies_list[i] = train_accuracies_list[i] + (predicted_outputs[i].argmax(1) == true_labels[i]).float().sum().item()

            functional.reset_net(net)

            net.train_times += 1

            # hyperparams step
            if net.train_times % hypergrad_every == 0:
                curr_hypergrads = hyperstep()

                if isinstance(auxiliary_net, MonoHyperNet):
                    # monotonic network
                    auxiliary_net.clamp()

        train_time = time.time() - train_start_time

        for i in range(num_tasks):
            train_losses_list[i] = train_losses_list[i] / train_samples
            train_accuracies_list[i] = train_accuracies_list[i] / train_samples

        if train_accuracies_list[0] > max_train_acc:
            max_train_acc = train_accuracies_list[0]

        for i in range(num_tasks):
            writer.add_scalar('train_loss_%d' % (i+1), train_losses_list[i], epoch)
            writer.add_scalar('train_acc_%d' % (i+1), train_accuracies_list[i], epoch)

        lr_scheduler.step()

        net.eval()
        raw_dataset.is_train = False

        test_losses_list = [0 for _ in range(num_tasks)]
        test_accuracies_list = [0 for _ in range(num_tasks)]
        test_samples = 0
        test_start_time = time.time()
        test_true_labels = []
        test_predict_labels = []
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]

                true_labels = [(task_label[label]).to(args.device) for task_label in task_labels]
                one_hot_labels = [F.one_hot(true_labels[i], task_number_classes[i]).float() for i in range(num_tasks)]

                predicted_outputs = net(frame)
                predicted_output_means = [predicted_output.mean(0) for predicted_output in predicted_outputs]

                test_losses = [F.mse_loss(predicted_output_means[i], one_hot_labels[i]) for i in range(num_tasks)]
                test_samples += label.numel()

                for i in range(num_tasks):
                    test_losses_list[i] = test_losses_list[i] + test_losses[i].item() * label.numel()
                    test_accuracies_list[i] = test_accuracies_list[i] + (predicted_output_means[i].argmax(1) == true_labels[i]).float().sum().item()

                test_true_labels.extend(true_labels[0].tolist())
                test_predict_labels.extend(predicted_output_means[0].argmax(1).tolist())

                functional.reset_net(net)
        test_time = time.time() - test_start_time

        for i in range(num_tasks):
            test_losses_list[i] = test_losses_list[i] / test_samples
            test_accuracies_list[i] = test_accuracies_list[i] / test_samples

        for i in range(num_tasks):
            writer.add_scalar('test_loss_%d' % (i+1), test_losses_list[i], epoch)
            writer.add_scalar('test_acc_%d' % (i+1), test_accuracies_list[i], epoch)

        save_max = False
        if test_accuracies_list[0] > max_test_acc:
            save_max = True
            max_test_acc = test_accuracies_list[0]
            np.save(out_dir + '/true_labels.npy', test_true_labels)
            np.save(out_dir + '/predict_labels.npy', test_predict_labels)

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_train_acc': max_train_acc,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        if ((epoch % 10) == 0) or (epoch == (args.epochs - 1)):
            print('%03d, ' % epoch, end='')

            print('train_acc: ', end='')
            for i in range(num_tasks):
                print('%0.4f, ' % train_accuracies_list[i], end='')

            print('test_acc: ', end='')
            for i in range(num_tasks):
                print('%0.4f, ' % test_accuracies_list[i], end='')

            print('max_train: %0.4f, ' % max_train_acc, end='')
            print('max_test: %0.4f, ' % max_test_acc, end='')

            print('speed: %8.4f, %8.4f' % (train_time, test_time))


if __name__ == '__main__':
    main()
