import argparse

def options():
    parser = argparse.ArgumentParser(description='Reference model training, poisons generation, and saving')


    ###########################################################################
    # Reference model training:
    parser.add_argument('--reference_path', type=str,default='./reference_model')
    parser.add_argument('--centroid_path', type=str,default='./centroid')
    parser.add_argument('--poison_path', type=str,default='./AT_noise')
    parser.add_argument('--data_path', default='/home/data', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--robust_eps', default=4.0, type=float,help='robustness of the reference model')
    parser.add_argument('--eps', default=8.0, type=float,help='poison budget')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', '-a', type=float, default=0.00784, 
            help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
    parser.add_argument('--k', '-k', type=int, default=10, 
        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf', 
            help='the type of the perturbation (linf or l2)')
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100'], default='CIFAR10', 
            help='choose dataset')
    parser.add_argument('--net', default='ResNet18', type=lambda s: [str(item) for item in s.split(',')])
    parser.add_argument('--attackoptim', default='PGD', type=str)
    parser.add_argument('--save', default='poison_dataset', help='Export poisons into a given format. Options are full/limited/automl/numpy.')
    parser.add_argument('--recipe', default='push', type=str, choices=['push','pull'])
    parser.add_argument('--data_aug', type=str, default='default', help='Mode of diff. data augmentation.')
    parser.add_argument('--poison_partition', default=None, type=int, help='How many poisons to craft at a time')
    parser.add_argument('--poisonkey', default=None, type=str, help='Initialize poison setup with this key.')
    parser.add_argument('--budget', default=1.0, type=float, help='Fraction of training data that is poisoned')
    parser.add_argument('--pbatch', default=512, type=int, help='Poison batch size during optimization')
    parser.add_argument('--pshuffle', action='store_true', help='Shuffle poison batch during optimization')
    parser.add_argument('--ensemble', default=1, type=int, help='Ensemble of networks to brew the poison on')
    parser.add_argument('--tau', default=0.05, type=float)
    parser.add_argument('--restarts', default=1, type=int, help='How often to restart the attack.')
    parser.add_argument('--init', default='randn', type=str)
    parser.add_argument("--local_rank", default=None, type=int, help='Distributed rank. This is an INTERNAL ARGUMENT! '
                                                                     'Only the launch utility should set this argument!')
    parser.add_argument('--attackiter', default=300, type=int)
    parser.add_argument('--clean_grad', action='store_true', help='Compute the first-order poison gradient.')
    parser.add_argument('--paugment', action='store_false', help='Do not augment poison batch during optimization')
    ###########################################################################
   
    return parser
