'''Train reference model for further poison generation.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from models import *
from utils import FastGradientSignUntargeted, options


torch.set_num_threads(1)



args = options().parse_args()

if not os.path.isdir(args.reference_path):
    os.mkdir(args.reference_path)

device = torch.device('cuda', args.gpu_id)
start_epoch = 0 

# Data
print('==> Preparing data..')


dm = torch.tensor([[[[0]],[[0]],[[0]]]]).to(device)
ds = torch.tensor([[[[1]],[[1]],[[1]]]]).to(device)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == 'CIFAR10':
    baseset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=False, transform=transform_train)
    num_class = 10
elif args.dataset == 'CIFAR100':
    baseset = torchvision.datasets.CIFAR100(
        root=args.data_path, train=True, download=False, transform=transform_train)
    num_class = 100


trainloader = torch.utils.data.DataLoader(
    baseset, batch_size=128, shuffle=True, num_workers=0)

if args.dataset == 'CIFAR10':
    testset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=False, transform=transform_test)
elif args.dataset == 'CIFAR100':
    testset = torchvision.datasets.CIFAR100(
        root=args.data_path, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18(num_classes=num_class)
# net = DenseNet121()
# net = MobileNetV2()
# net = ResNet34()
net = net.to(device)


def train(epoch, adv=True):
    print('\nEpoch: %d' % epoch)
    net.train()
    attack = FastGradientSignUntargeted(net, 
                                args.robust_eps, 
                                args.alpha, 
                                dm, 
                                ds, 
                                max_iters=args.k, 
                                device=device,
                                _type=args.perturbation_type)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if adv:
            adv_inputs = attack.perturb(inputs, targets, 'mean', True)
            outputs = net(adv_inputs)
        else:
            outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'loss: {test_loss/(batch_idx + 1)}, acc: {100. * correct/total}')

    # Save checkpoint.
    acc = 100.*correct/total
    return acc



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75,90],gamma=0.1)
for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    torch.save(net.state_dict(), os.path.join(args.reference_path,args.dataset+'_eps_'+str(int(args.robust_eps))+'.pth'))
    acc = test(epoch)
    scheduler.step()
