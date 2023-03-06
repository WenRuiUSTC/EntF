"""Generate poisons."""

import torch
import os
import datetime
import time
from utils import system_startup, options

import village
from models import *
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_descriptor')


torch.set_num_threads(1)

args = options().parse_args()


if __name__ == "__main__":
    setup = system_startup(args)

    print('-----------------Load Reference Model-----------------------')
    net = ResNetEmb18(num_classes=10).to(setup['device'])
    net.load_state_dict(torch.load(os.path.join(args.reference_path,args.dataset+'_eps_'+str(int(args.robust_eps))+'.pth'), map_location=setup['device']))
    print('-----------------Load Successfully-----------------------')

    materials = village.Furnace(args, args.batch_size, args.data_aug, setup=setup)
    forgemaster = village.Forgemaster(args, setup=setup)

    if not os.path.isdir(args.poison_path):
        os.mkdir(args.poison_path)
    if not os.path.isdir(args.poison_path+'/data'):
        os.mkdir(args.poison_path+'/data')

    start_time = time.time()

    poison_delta = forgemaster.forge(net, materials)
    forge_time = time.time()

    # Export
    if args.save is not None:
        materials.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'--------------------------- forge time: {str(datetime.timedelta(seconds=forge_time - start_time))}')
    print('-------------Job finished.-------------------------')
