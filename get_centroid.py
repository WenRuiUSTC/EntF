'''Get centroid.'''
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


torch.set_num_threads(8)


args = options().parse_args()

print(args)
device = torch.device('cuda', args.gpu_id)
if not os.path.isdir(args.centroid_path):
    os.mkdir(args.centroid_path)


if args.dataset == 'CIFAR10':
    baseset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=False, transform=transforms.ToTensor())
    num_class = 10
elif args.dataset == 'CIFAR100':
    baseset = torchvision.datasets.CIFAR100(
        root=args.data_path, train=True, download=False, transform=transforms.ToTensor())
    num_class = 100



transform_train = transforms.Compose([
    transforms.ToTensor(),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])


baseloader = torch.utils.data.DataLoader(
    baseset, batch_size=128, shuffle=True, num_workers=0)



if args.dataset == 'CIFAR10':
    testset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=False, transform=transform_test)
elif args.dataset == 'CIFAR100':
    testset = torchvision.datasets.CIFAR100(
        root=args.data_path, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)


def get_emb(net,trainloader):
    net.eval()
    emb_list = [np.zeros(512) for i in range(10)]
    cnt = [0 for i in range(10)]
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        batch_size = targets.size(0)
        embeddings, outputs = net(inputs)
        embeddings = embeddings.cpu().detach().numpy().reshape(batch_size, -1)
        for i in range(targets.shape[0]):
            emb_list[targets[i].item()]+=embeddings[i]
            cnt[targets[i].item()]+= 1
    for i in range(10):
        emb_list[i] = emb_list[i] / cnt[i]
    return emb_list




net = ResNetEmb18(num_classes=num_class)
net = net.to(device)



net.load_state_dict(torch.load(os.path.join(args.reference_path,args.dataset+'_eps_'+str(int(args.robust_eps))+'.pth'), map_location=device))
emb_list = get_emb(net, baseloader)
torch.save(torch.tensor(emb_list,dtype=torch.float32), os.path.join(args.centroid_path,args.dataset+'_eps_'+str(int(args.robust_eps))+'.pt'))

