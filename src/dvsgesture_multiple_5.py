import sys
import time
import os
import argparse

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

from src.augment_dvs128gesture import DVS128Gesture

import torchvision.transforms as torch_transform


class DVSGestureNetMultiple(nn.Module):
    def __init__(self, num_outputs: List = 10, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        if isinstance(num_outputs, int):
            self.output_list = [num_outputs]
        elif isinstance(num_outputs, list):
            self.output_list = num_outputs
        else:
            raise ValueError("output parameter must be int or list of ints")

        conv = []
        for i in range(5):
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
            layer.Flatten(),
        )

        self.task_classifier_1 = nn.Sequential(
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, self.output_list[0]*10),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

        self.task_classifier_2 = nn.Sequential(
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, self.output_list[1] * 10),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

        self.task_classifier_3 = nn.Sequential(
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, self.output_list[2] * 10),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

        self.task_classifier_4 = nn.Sequential(
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, self.output_list[3] * 10),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

        self.task_classifier_5 = nn.Sequential(
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, self.output_list[4] * 10),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        x = self.feature_extraction(x)
        output_1 = self.task_classifier_1(x)
        output_2 = self.task_classifier_2(x)
        output_3 = self.task_classifier_3(x)
        output_4 = self.task_classifier_4(x)
        output_5 = self.task_classifier_5(x)
        return output_1, output_2, output_3, output_4, output_5


def main():
    # python dvsgesture_multiple_2.py -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir ./datasets/DVSGesture/
    # -out-dir ./results/1/ -amp -cupy -opt adam -lr 0.001 -j 2

    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',  help='number of total epochs to run')
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
    parser.add_argument('-loss-rate', default=0.5, type=float, help='loss combination rate')
    parser.add_argument('-exp-number', type=int, help='save in a subfolder with name exp_number')

    args = parser.parse_args()
    print(args)

    if args.plif:
        neuron_model = neuron.ParametricLIFNode
    else:
        neuron_model = neuron.LIFNode

    task_1_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64)
    task_2_labels = torch.tensor([0, 1, 2, 1, 1, 2, 2, 0, 0, 3, 4], dtype=torch.int64)
    task_3_labels = torch.tensor([1, 3, 2, 3, 3, 2, 2, 0, 0, 4, 5], dtype=torch.int64)
    task_4_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64)
    task_5_labels = torch.tensor([2, 2, 2, 3, 1, 2, 0, 1, 0, 0, 2], dtype=torch.int64)
    task_1_number_classes = task_1_labels.max().item() + 1
    task_2_number_classes = task_2_labels.max().item() + 1
    task_3_number_classes = task_3_labels.max().item() + 1
    task_4_number_classes = task_4_labels.max().item() + 1
    task_5_number_classes = task_5_labels.max().item() + 1

    net = DVSGestureNetMultiple(
        num_outputs=[task_1_number_classes, task_2_number_classes, task_3_number_classes, task_4_number_classes, task_5_number_classes],
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

    # ============
    # Load dataset
    # ============

    transform = None
    if args.augment:
        transform = torch_transform.Compose([
            torch_transform.RandomAffine(5, translate=(0.125, 0.125)),
            torch_transform.RandomApply([torch_transform.RandomErasing(p=0.25, scale=(0.02, 0.15))], p=0.5)
        ])

    train_set = DVS128Gesture(root=args.data_dir,
                              train=True,
                              data_type='frame',
                              frames_number=args.T,
                              split_by='number',
                              transform=transform)
    test_set = DVS128Gesture(root=args.data_dir,
                             train=False,
                             data_type='frame',
                             frames_number=args.T,
                             split_by='number')

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
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

    out_dir = os.path.join(args.out_dir, f'DVSGestureMulti5-T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}_loss{args.loss_rate}')

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

    for epoch in range(start_epoch, args.epochs):
        net.train()
        train_loss_1 = 0
        train_loss_2 = 0
        train_loss_3 = 0
        train_loss_4 = 0
        train_loss_5 = 0
        train_acc_1 = 0
        train_acc_2 = 0
        train_acc_3 = 0
        train_acc_4 = 0
        train_acc_5 = 0
        train_samples = 0
        train_start_time = time.time()
        for frame, label in train_data_loader:
            optimizer.zero_grad()

            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            true_label_1 = (task_1_labels[label]).to(args.device)
            true_label_2 = (task_2_labels[label]).to(args.device)
            true_label_3 = (task_3_labels[label]).to(args.device)
            true_label_4 = (task_4_labels[label]).to(args.device)
            true_label_5 = (task_5_labels[label]).to(args.device)

            onehot_label_1 = F.one_hot(true_label_1, task_1_number_classes).float()
            onehot_label_2 = F.one_hot(true_label_2, task_2_number_classes).float()
            onehot_label_3 = F.one_hot(true_label_3, task_3_number_classes).float()
            onehot_label_4 = F.one_hot(true_label_4, task_4_number_classes).float()
            onehot_label_5 = F.one_hot(true_label_5, task_5_number_classes).float()

            if scaler is not None:
                with amp.autocast():
                    predicted_output_1, predicted_output_2, predicted_output_3, predicted_output_4, predicted_output_5 = net(frame)
                    predicted_output_1 = predicted_output_1.mean(0)
                    predicted_output_2 = predicted_output_2.mean(0)
                    predicted_output_3 = predicted_output_3.mean(0)
                    predicted_output_4 = predicted_output_4.mean(0)
                    predicted_output_5 = predicted_output_5.mean(0)
                    loss_1 = F.mse_loss(predicted_output_1, onehot_label_1)
                    loss_2 = F.mse_loss(predicted_output_2, onehot_label_2)
                    loss_3 = F.mse_loss(predicted_output_3, onehot_label_3)
                    loss_4 = F.mse_loss(predicted_output_4, onehot_label_4)
                    loss_5 = F.mse_loss(predicted_output_5, onehot_label_5)
                    loss = (1-args.loss_rate)*loss_1 + args.loss_rate*(loss_2 + loss_3 + loss_4 + loss_5)/4
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predicted_output_1, predicted_output_2, predicted_output_3, predicted_output_4, predicted_output_5 = net(frame)
                predicted_output_1 = predicted_output_1.mean(0)
                predicted_output_2 = predicted_output_2.mean(0)
                predicted_output_3 = predicted_output_3.mean(0)
                predicted_output_4 = predicted_output_4.mean(0)
                predicted_output_5 = predicted_output_5.mean(0)
                loss_1 = F.mse_loss(predicted_output_1, onehot_label_1)
                loss_2 = F.mse_loss(predicted_output_2, onehot_label_2)
                loss_3 = F.mse_loss(predicted_output_3, onehot_label_3)
                loss_4 = F.mse_loss(predicted_output_4, onehot_label_4)
                loss_5 = F.mse_loss(predicted_output_5, onehot_label_5)
                loss = (1-args.loss_rate)*loss_1 + args.loss_rate*(loss_2 + loss_3 + loss_4 + loss_5)/4
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss_1 += loss_1.item() * label.numel()
            train_loss_2 += loss_2.item() * label.numel()
            train_loss_3 += loss_3.item() * label.numel()
            train_loss_4 += loss_4.item() * label.numel()
            train_loss_5 += loss_5.item() * label.numel()
            train_acc_1 += (predicted_output_1.argmax(1) == true_label_1).float().sum().item()
            train_acc_2 += (predicted_output_2.argmax(1) == true_label_2).float().sum().item()
            train_acc_3 += (predicted_output_3.argmax(1) == true_label_3).float().sum().item()
            train_acc_4 += (predicted_output_4.argmax(1) == true_label_4).float().sum().item()
            train_acc_5 += (predicted_output_5.argmax(1) == true_label_5).float().sum().item()

            functional.reset_net(net)

        train_time = time.time() - train_start_time
        train_loss_1 /= train_samples
        train_loss_2 /= train_samples
        train_loss_3 /= train_samples
        train_loss_4 /= train_samples
        train_loss_5 /= train_samples
        train_acc_1 /= train_samples
        train_acc_2 /= train_samples
        train_acc_3 /= train_samples
        train_acc_4 /= train_samples
        train_acc_5 /= train_samples

        if train_acc_1 > max_train_acc:
            max_train_acc = train_acc_1

        writer.add_scalar('train_loss_1', train_loss_1, epoch)
        writer.add_scalar('train_loss_2', train_loss_2, epoch)
        writer.add_scalar('train_loss_3', train_loss_3, epoch)
        writer.add_scalar('train_loss_4', train_loss_4, epoch)
        writer.add_scalar('train_loss_5', train_loss_5, epoch)
        writer.add_scalar('train_acc_1', train_acc_1, epoch)
        writer.add_scalar('train_acc_2', train_acc_2, epoch)
        writer.add_scalar('train_acc_3', train_acc_3, epoch)
        writer.add_scalar('train_acc_4', train_acc_4, epoch)
        writer.add_scalar('train_acc_5', train_acc_5, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss_1 = 0
        test_loss_2 = 0
        test_loss_3 = 0
        test_loss_4 = 0
        test_loss_5 = 0
        test_acc_1 = 0
        test_acc_2 = 0
        test_acc_3 = 0
        test_acc_4 = 0
        test_acc_5 = 0
        test_samples = 0
        test_start_time = time.time()
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                true_label_1 = (task_1_labels[label]).to(args.device)
                true_label_2 = (task_2_labels[label]).to(args.device)
                true_label_3 = (task_3_labels[label]).to(args.device)
                true_label_4 = (task_4_labels[label]).to(args.device)
                true_label_5 = (task_5_labels[label]).to(args.device)

                onehot_label_1 = F.one_hot(true_label_1, task_1_number_classes).float()
                onehot_label_2 = F.one_hot(true_label_2, task_2_number_classes).float()
                onehot_label_3 = F.one_hot(true_label_3, task_3_number_classes).float()
                onehot_label_4 = F.one_hot(true_label_4, task_4_number_classes).float()
                onehot_label_5 = F.one_hot(true_label_5, task_5_number_classes).float()

                predicted_output_1, predicted_output_2, predicted_output_3, predicted_output_4, predicted_output_5 = net(frame)
                predicted_output_1 = predicted_output_1.mean(0)
                predicted_output_2 = predicted_output_2.mean(0)
                predicted_output_3 = predicted_output_3.mean(0)
                predicted_output_4 = predicted_output_4.mean(0)
                predicted_output_5 = predicted_output_5.mean(0)

                loss_1 = F.mse_loss(predicted_output_1, onehot_label_1)
                loss_2 = F.mse_loss(predicted_output_2, onehot_label_2)
                loss_3 = F.mse_loss(predicted_output_3, onehot_label_3)
                loss_4 = F.mse_loss(predicted_output_4, onehot_label_4)
                loss_5 = F.mse_loss(predicted_output_5, onehot_label_5)
                loss = (1 - args.loss_rate) * loss_1 + args.loss_rate * (loss_2 + loss_3 + loss_4 + loss_5) / 5

                test_samples += label.numel()
                test_loss_1 += loss_1.item() * label.numel()
                test_loss_2 += loss_2.item() * label.numel()
                test_loss_3 += loss_3.item() * label.numel()
                test_loss_4 += loss_4.item() * label.numel()
                test_loss_5 += loss_5.item() * label.numel()
                test_acc_1 += (predicted_output_1.argmax(1) == true_label_1).float().sum().item()
                test_acc_2 += (predicted_output_2.argmax(1) == true_label_2).float().sum().item()
                test_acc_3 += (predicted_output_3.argmax(1) == true_label_3).float().sum().item()
                test_acc_4 += (predicted_output_4.argmax(1) == true_label_4).float().sum().item()
                test_acc_5 += (predicted_output_5.argmax(1) == true_label_5).float().sum().item()
                functional.reset_net(net)
        test_time = time.time() - test_start_time
        test_loss_1 /= test_samples
        test_loss_2 /= test_samples
        test_loss_3 /= test_samples
        test_loss_4 /= test_samples
        test_loss_5 /= test_samples
        test_acc_1 /= test_samples
        test_acc_2 /= test_samples
        test_acc_3 /= test_samples
        test_acc_4 /= test_samples
        test_acc_5 /= test_samples
        writer.add_scalar('test_loss_1', test_loss_1, epoch)
        writer.add_scalar('test_loss_2', test_loss_2, epoch)
        writer.add_scalar('test_loss_3', test_loss_3, epoch)
        writer.add_scalar('test_loss_4', test_loss_4, epoch)
        writer.add_scalar('test_loss_5', test_loss_5, epoch)
        writer.add_scalar('test_acc_1', test_acc_1, epoch)
        writer.add_scalar('test_acc_2', test_acc_2, epoch)
        writer.add_scalar('test_acc_3', test_acc_3, epoch)
        writer.add_scalar('test_acc_5', test_acc_5, epoch)

        save_max = False
        if test_acc_1 > max_test_acc:
            max_test_acc = test_acc_1
            save_max = True

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

            print('train: %0.4f, %0.4f, %0.4f, %0.4f, %0.4f | ' % (train_acc_1, train_acc_2, train_acc_3, train_acc_4, train_acc_5), end='')

            print('test: %0.4f, %0.4f, %0.4f, %0.4f, %0.4f  | ' % (test_acc_1, test_acc_2, test_acc_3, test_acc_4, test_acc_5), end='')

            print('max_train: %0.4f | ' % max_train_acc, end='')
            print('max_test: %0.4f | ' % max_test_acc, end='')

            print('speed: %8.4f, %8.4f' % (train_time, test_time))


if __name__ == '__main__':
    main()
