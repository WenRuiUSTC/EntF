"""Interface for poison recipes."""
from .forgemaster_push import ForgemasterPush
from .forgemaster_pull import ForgemasterPull

import torch


def Forgemaster(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == 'pull':
        return ForgemasterPull(args, setup)
    elif args.recipe == 'push':
        return ForgemasterPush(args, setup)
    else:
        raise NotImplementedError()


__all__ = ['Forgemaster']
