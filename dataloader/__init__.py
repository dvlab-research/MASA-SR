# from .dataset import GoProDataset, MixDataset, VideoDataset
from .data_sampler import DistIterSampler
import torch
import torch.utils.data


def create_dataloader(dataset, args, sampler=None):
    phase = args.phase
    if phase == 'train':
        if args.dist:
            world_size = torch.distributed.get_world_size()
            num_workers = args.num_workers
            assert args.batch_size % world_size == 0
            batch_size = args.batch_size // world_size
            shuffle = False
        else:
            num_workers = args.num_workers * len(args.gpu_ids)
            batch_size = args.batch_size
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=False)
