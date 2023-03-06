"""
this code is modified from https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks
original author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


def project(x, original_x, epsilon, ds, _type='linf'):

    if _type == 'linf':
        max_x = original_x + epsilon / ds /255
        min_x = original_x - epsilon / ds /255

        x = torch.max(torch.min(x, max_x), min_x)

    else:
        raise NotImplementedError

    return x

class FastGradientSignUntargeted():
    b"""
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, epsilon, alpha, dm, ds, max_iters, device, _type='linf'):
        """
        modify the min_val and max_val, now min_val
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.dm = dm
        self.ds = ds
        self.max_iters = max_iters

        self.device = device
        self._type = _type
        
    def perturb(self, original_images, labels, reduction4loss='mean', random_start=False):
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon/255, self.epsilon/255)
            rand_perturb = rand_perturb.to(self.device)
            x = original_images + rand_perturb
            x = torch.max(torch.min(x, (1 - self.dm) / self.ds), -self.dm / self.ds)
        else:
            x = original_images.clone()

        x.requires_grad = True 
        self.model.eval()

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                outputs = self.model(x)             

                loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)
                if reduction4loss == 'none':
                    grad_outputs = torch.ones(loss.shape).to(self.device)                    
                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, 
                        only_inputs=True)[0]

                x.data += self.alpha * torch.sign(grads.data) 
                x = project(x, original_images, self.epsilon, self.ds, self._type)
                x = torch.max(torch.min(x, (1 - self.dm) / self.ds), -self.dm / self.ds)
        self.model.train()

        return x
