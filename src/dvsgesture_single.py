import sys
import time
import os
import argparse

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.cuda import amp

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from src.augment_dvs128gesture import DVS128Gesture

import torchvision.transforms as torch_transform


def main():

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
    parser.add_argument('-exp-number', type=int, help='save in a subfolder with name exp_number')

    args = parser.parse_args()
    print(args)

    if args.plif:
        neuron_model = neuron.ParametricLIFNode
    else:
        neuron_model = neuron.LIFNode

    net = parametric_lif_net.DVSGestureNet(
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

    optimizer = None
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

    out_dir = os.path.join(args.out_dir, f'DVSGestureSingle-T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')

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
        train_loss = 0
        train_acc = 0
        train_samples = 0
        train_start_time = time.time()
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 11).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time() - train_start_time
        train_speed = train_samples / train_time
        train_loss /= train_samples
        train_acc /= train_samples

        if train_acc > max_train_acc:
            max_train_acc = train_acc

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        test_start_time = time.time()
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 11).float()
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time() - test_start_time
        test_speed = test_samples / test_time
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
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

            print('train_acc: %0.4f, ' % train_acc, end='')
            print('test_acc: %0.4f, ' % test_acc, end='')

            print('max_train_acc: %0.4f, ' % max_train_acc, end='')
            print('max_test_acc: %0.4f, ' % max_test_acc, end='')

            print('speed: %8.4f, %8.4f' % (train_time, test_time))


if __name__ == '__main__':
    main()
