"""Various utilities."""

import os
import csv
import socket
import datetime

from collections import defaultdict

import torch
import torch.nn.functional as F
import random
import numpy as np

NON_BLOCKING = True



def system_startup(args=None, defs=None):
    """Decide and print GPU / CPU / hostname info."""
    device = torch.device('cuda', args.gpu_id) if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float, non_blocking=NON_BLOCKING)
    print('Currently evaluating -------------------------------:')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    if args is not None:
        print(args)
    if defs is not None:
        print(repr(defs))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')

    return setup