import os
import time
import logging
import math
import argparse
import numpy as np
import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as data
import warnings
from utils.util import setup_logger, print_args
from models import Trainer

def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='referenceSR Training')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--name', default='train_masa', type=str)
    parser.add_argument('--phase', default='train', type=str)

    ## device setting
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    ## network setting
    parser.add_argument('--net_name', default='MASA', type=str, help='RefNet | Baseline')
    parser.add_argument('--sr_scale', default=4, type=int)
    parser.add_argument('--input_nc', default=3, type=int)
    parser.add_argument('--output_nc', default=3, type=int)
    parser.add_argument('--nf', default=64, type=int)
    parser.add_argument('--n_blks', default='4, 4, 4', type=str)
    parser.add_argument('--nf_ctt', default=32, type=int)
    parser.add_argument('--n_blks_ctt', default='2, 2, 2', type=str)
    parser.add_argument('--num_nbr', default=1, type=int)
    parser.add_argument('--n_blks_dec', default=10, type=int)

    ## dataloader setting
    parser.add_argument('--data_root', default='/home/liyinglu/newData/datasets/SR/',type=str)
    parser.add_argument('--dataset', default='CUFED', type=str, help='CUFED')
    parser.add_argument('--testset', default='TestSet', type=str, help='TestSet')
    parser.add_argument('--save_test_root', default='generated', type=str)
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--batch_size', default=9, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--multi_scale', action='store_true')
    parser.add_argument('--data_augmentation', action='store_true')

    ## optim setting
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_D', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--start_iter', default=0, type=int)
    parser.add_argument('--max_iter', default=500, type=int)

    parser.add_argument('--loss_l1', action='store_true')
    parser.add_argument('--loss_mse', action='store_true')
    parser.add_argument('--loss_perceptual', action='store_true')
    parser.add_argument('--loss_adv', action='store_true')
    parser.add_argument('--gan_type', default='WGAN_GP', type=str)

    parser.add_argument('--lambda_l1', default=1, type=float)
    parser.add_argument('--lambda_mse', default=1, type=float)
    parser.add_argument('--lambda_perceptual', default=1, type=float)
    parser.add_argument('--lambda_adv', default=5e-3, type=float)

    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--resume_optim', default='', type=str)
    parser.add_argument('--resume_scheduler', default='', type=str)

    ## log setting
    parser.add_argument('--log_freq', default=10, type=int)
    parser.add_argument('--vis_freq', default=50000, type=int) #50000
    parser.add_argument('--save_epoch_freq', default=10, type=int) #100
    parser.add_argument('--test_freq', default=100, type=int) #100
    parser.add_argument('--save_folder', default='./weights', type=str)
    parser.add_argument('--vis_step_freq', default=100, type=int)
    parser.add_argument('--use_tb_logger', action='store_true')
    parser.add_argument('--save_test_results', action='store_true')

    ## for evaluate
    parser.add_argument('--ref_level', default=1, type=int)


    ## setup training environment
    args = parser.parse_args()
    set_random_seed(args.random_seed)

    ## setup training device
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        args.dist = False
        args.rank = -1
        print('Disabled distributed training.')
    else:
        args.dist = True
        init_dist()
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    args.save_folder = os.path.join(args.save_folder, args.name)
    args.vis_save_dir = os.path.join(args.save_folder,  'vis')
    args.snapshot_save_dir = os.path.join(args.save_folder,  'snapshot')
    log_file_path = args.save_folder + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'

    if args.rank <= 0:
        if os.path.exists(args.vis_save_dir) == False:
            os.makedirs(args.vis_save_dir)
        if os.path.exists(args.snapshot_save_dir) == False:
            os.mkdir(args.snapshot_save_dir)
        setup_logger(log_file_path)

    print_args(args)

    cudnn.benchmark = True

    ## train model
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
