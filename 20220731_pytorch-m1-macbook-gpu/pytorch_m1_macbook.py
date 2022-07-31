# -*- coding: utf-8 -*-
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms as tt
from torchvision.models import resnet18

import os
from argparse import ArgumentParser
import time

def main(device):
    # ResNetのハイパーパラメータ
    n_epoch = 5            # エポック数
    batch_size = 512       # ミニバッチサイズ
    momentum = 0.9         # SGDのmomentum
    lr = 0.01              # 学習率
    weight_decay = 0.00005 # weight decay

    # 訓練データとテストデータを用意
    mean = (0.491, 0.482, 0.446)
    std = (0.247, 0.243, 0.261)
    train_transform = tt.Compose([
        tt.RandomHorizontalFlip(p=0.5),
        tt.RandomCrop(size=32, padding=4, padding_mode='reflect'),
        tt.ToTensor(),
        tt.Normalize(mean=mean, std=std)
    ])
    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])
    root = os.path.dirname(os.path.abspath(__file__))
    train_set = CIFAR10(root=root, train=True,
                        download=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=8)

    # ResNetの準備
    resnet = resnet18()
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)

    # 訓練
    criterion = CrossEntropyLoss()
    optimizer = SGD(resnet.parameters(), lr=lr,
                    momentum=momentum, weight_decay=weight_decay)
    train_start_time = time.time()
    resnet.to(device)
    resnet.train()
    for epoch in range(1, n_epoch+1):
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            train_loss += loss.item()
            del loss  # メモリ節約のため
            optimizer.step()
        print('Epoch {} / {}: time = {:.2f}[s], loss = {:.2f}'.format(
            epoch, n_epoch, time.time() - train_start_time, train_loss))
    print('Train time on {}: {:.2f}[s] (Train loss = {:.2f})'.format(
        device, time.time() - train_start_time, train_loss))

    # 評価
    test_set = CIFAR10(root=root, train=False, download=True,
                       transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=8)
    test_loss = 0.0
    test_start_time = time.time()
    resnet.eval()
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = resnet(inputs)
        labels = labels.to(device)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
    print('Test time on {}: {:.2f}[s](Test loss = {:.2f})'.format(
        device, time.time() - test_start_time, test_loss))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='mps',
                        choices=['cpu', 'mps'])
    args = parser.parse_args()
    device = torch.device(args.device)
    main(device)
