import os
# from imageio import imread
from PIL import Image, ImageOps
import numpy as np
import glob
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
import sys
sys.path.append('..')
from utils import util


class TrainSet(Dataset):
    def __init__(self, args):
        self.input_list = sorted([os.path.join(args.data_root, args.dataset, 'train/input', name) for name in
                                  os.listdir(os.path.join(args.data_root, args.dataset, 'train/input'))])
        self.ref_list = sorted([os.path.join(args.data_root, args.dataset, 'train/ref', name) for name in
                                os.listdir(os.path.join(args.data_root, args.dataset, 'train/ref'))])
        self.scale = args.sr_scale
        self.data_augmentation = args.data_augmentation
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        if random.random() < 0.5:
            HR = cv2.imread(self.input_list[idx])
            Ref = cv2.imread(self.ref_list[idx])
        else:
            HR = cv2.imread(self.ref_list[idx])
            Ref = cv2.imread(self.input_list[idx])

        h, w, c = Ref.shape
        if h % 16 != 0 or w % 16 != 0:
            h_new = math.ceil(h / 16) * 16
            w_new = math.ceil(w / 16) * 16
            Ref = cv2.copyMakeBorder(Ref, 0, h_new - h, 0, w_new - w, cv2.BORDER_REPLICATE)

        h, w, c = HR.shape
        if h % 16 != 0 or w % 16 != 0:
            h_new = math.ceil(h / 16) * 16
            w_new = math.ceil(w / 16) * 16
            HR = cv2.copyMakeBorder(HR, 0, h_new - h, 0, w_new - w, cv2.BORDER_REPLICATE)

        if self.data_augmentation:
            alpha = random.uniform(0.7, 1.3)
            beta = random.uniform(-20, 20)
            if random.random() < 0.5:
                Ref = cv2.convertScaleAbs(Ref, alpha=alpha, beta=beta)
                Ref = np.clip(Ref, 0, 255)
            else:
                HR = cv2.convertScaleAbs(HR, alpha=alpha, beta=beta)
                HR = np.clip(HR, 0, 255)

        h, w, _ = HR.shape
        LR = np.array(Image.fromarray(HR).resize((w // self.scale, h // self.scale), Image.BICUBIC))
        h, w, _ = LR.shape
        LR_down = np.array(Image.fromarray(LR).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        h, w, _ = Ref.shape
        Ref_down = np.array(Image.fromarray(Ref).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        sample = {'HR': HR,
                  'LR': LR,
                  'Ref': Ref,
                  'Ref_down': Ref_down}

        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        return sample


class TestSet(Dataset):
    def __init__(self, args):
        ref_level = args.ref_level
        # ref_level = 5
        self.input_list = sorted(glob.glob(os.path.join(args.data_root, args.dataset, 'test/CUFED5', '*_0.png')))
        self.ref_list = sorted(glob.glob(os.path.join(args.data_root, args.dataset, 'test/CUFED5',
                                                       '*_' + str(ref_level) + '.png')))
        self.scale = args.sr_scale
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        HR_name = os.path.basename(self.input_list[idx])

        ### HR
        HR = cv2.imread(self.input_list[idx])
        Ref = cv2.imread(self.ref_list[idx])

        # pad HR to be mutiple of 64
        h, w, c = HR.shape
        if h % 32 != 0 or w % 32 != 0:
            h_new = math.ceil(h / 32) * 32
            w_new = math.ceil(w / 32) * 32
            pad_H_t = (h_new - h) // 2
            pad_H_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_H_l = (w_new - w) // 2
            pad_H_r = (w_new - w) // 2 + (w_new - w) % 2
            HR_pad = cv2.copyMakeBorder(HR.copy(), pad_H_t, pad_H_d, pad_H_l, pad_H_r, cv2.BORDER_REPLICATE)
        else:
            pad_H_t, pad_H_d, pad_H_l, pad_H_r = 0, 0, 0, 0
            HR_pad = HR

        # pad ref to be multiple of 16
        h, w, c = Ref.shape
        if h % 16 != 0 or w % 16 != 0:
            h_new = math.ceil(h / 16) * 16
            w_new = math.ceil(w / 16) * 16
            Ref = cv2.copyMakeBorder(Ref.copy(), 0, h_new - h, 0, w_new - w, cv2.BORDER_REPLICATE)

        h, w, c = HR.shape
        LR = np.array(Image.fromarray(HR).resize((w//self.scale, h//self.scale), Image.BICUBIC))

        h, w, c = HR_pad.shape
        LR_pad = np.array(Image.fromarray(HR_pad).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        h, w, c = LR_pad.shape
        LR_pad_down = np.array(Image.fromarray(LR_pad).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        h, w, _ = Ref.shape
        Ref_down = np.array(Image.fromarray(Ref).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        sample = {'HR': HR,
                  'HR_pad': HR_pad,
                  'LR': LR,
                  'LR_pad': LR_pad,
                  'LR_pad_down': LR_pad_down,
                  'Ref': Ref,
                  'Ref_down': Ref_down}

        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        pad_nums = [pad_H_t, pad_H_d, pad_H_l, pad_H_r]
        sample['pad_nums'] = pad_nums
        sample['HR_name'] = HR_name

        return sample


class Urban100(Dataset):
    def __init__(self, args):
        self.input_list = sorted(glob.glob('/home/liyinglu/newData/datasets/SR/SR_testing_datasets/Urban100/*.png'))
        self.scale = args.sr_scale
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        HR_name = os.path.basename(self.input_list[idx])

        ### HR
        HR = cv2.imread(self.input_list[idx])

        # pad HR to be mutiple of 64
        h, w, c = HR.shape
        if h % 64 != 0 or w % 64 != 0:
            h_new = math.ceil(h / 64) * 64
            w_new = math.ceil(w / 64) * 64
            pad_H_t = (h_new - h) // 2
            pad_H_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_H_l = (w_new - w) // 2
            pad_H_r = (w_new - w) // 2 + (w_new - w) % 2
            HR_pad = cv2.copyMakeBorder(HR.copy(), pad_H_t, pad_H_d, pad_H_l, pad_H_r, cv2.BORDER_REPLICATE)
        else:
            pad_H_t, pad_H_d, pad_H_l, pad_H_r = 0, 0, 0, 0
            HR_pad = HR

        h, w, c = HR.shape
        LR = np.array(Image.fromarray(HR).resize((w//self.scale, h//self.scale), Image.BICUBIC))

        h, w, c = HR_pad.shape
        LR_pad = np.array(Image.fromarray(HR_pad).resize((w // self.scale, h // self.scale), Image.BICUBIC))
        
        h, w, c = LR_pad.shape
        LR_pad_up = np.array(Image.fromarray(LR_pad).resize((w * self.scale, h * self.scale), Image.BICUBIC))


        h, w, c = LR_pad.shape
        LR_pad_down = np.array(Image.fromarray(LR_pad).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        LR_pad_down_x2 = np.array(Image.fromarray(LR_pad).resize((w // 2, h // 2), Image.BICUBIC))
        LR_pad_up_x2 = np.array(Image.fromarray(LR_pad).resize((w * 2, h * 2), Image.BICUBIC))

        sample = {'HR': HR,
                  'HR_pad': HR_pad,
                  'LR': LR,
                  'LR_pad': LR_pad,
                  'LR_pad_up': LR_pad_up,
                  'Ref': LR_pad,  # LR_pad_up_x2 | LR_pad_up | LR_pad
                  'Ref_down': LR_pad_down,  # LR_pad_down_x2 | LR_pad | LR_pad_down
                  'Ref_down_up': LR_pad}

        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        pad_nums = [pad_H_t, pad_H_d, pad_H_l, pad_H_r]
        sample['pad_nums'] = pad_nums
        sample['HR_name'] = HR_name

        return sample


class Sun80(Dataset):
    def __init__(self, args):
        self.input_list = sorted(glob.glob('/home/liyinglu/newData/datasets/SR/SR_testing_datasets/Sun_Hays_SR_groundtruth/*.jpg'))
        self.ref_folders = '/home/liyinglu/newData/datasets/SR/SR_testing_datasets/Sun_Hays_SR_scenematches'
        self.scale = args.sr_scale
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        HR_name = os.path.basename(self.input_list[idx])

        ref_folder = os.path.join(self.ref_folders, HR_name)
        ref_list = sorted(glob.glob(os.path.join(ref_folder, '*.jpg')))
        ref_idx = random.randint(0, len(ref_list)-1)


        ### HR
        HR = cv2.imread(self.input_list[idx])
        Ref = cv2.imread(ref_list[ref_idx])

        # pad HR to be mutiple of 64
        h, w, c = HR.shape
        if h % 64 != 0 or w % 64 != 0:
            h_new = math.ceil(h / 64) * 64
            w_new = math.ceil(w / 64) * 64
            pad_H_t = (h_new - h) // 2
            pad_H_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_H_l = (w_new - w) // 2
            pad_H_r = (w_new - w) // 2 + (w_new - w) % 2
            HR_pad = cv2.copyMakeBorder(HR.copy(), pad_H_t, pad_H_d, pad_H_l, pad_H_r, cv2.BORDER_REPLICATE)
        else:
            pad_H_t, pad_H_d, pad_H_l, pad_H_r = 0, 0, 0, 0
            HR_pad = HR

        # pad ref to be multiple of 16
        h, w, c = Ref.shape
        if h % 16 != 0 or w % 16 != 0:
            h_new = math.ceil(h / 16) * 16
            w_new = math.ceil(w / 16) * 16
            pad_t = (h_new - h) // 2
            pad_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_l = (w_new - w) // 2
            pad_r = (w_new - w) // 2 + (w_new - w) % 2
            Ref = cv2.copyMakeBorder(Ref.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_REPLICATE)


        h, w, c = HR.shape
        LR = np.array(Image.fromarray(HR).resize((w//self.scale, h//self.scale), Image.BICUBIC))

        h, w, c = HR_pad.shape
        LR_pad = np.array(Image.fromarray(HR_pad).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        h, w, c = LR_pad.shape
        LR_pad_down = np.array(Image.fromarray(LR_pad).resize((w // self.scale, h // self.scale), Image.BICUBIC))
        LR_pad_up = np.array(Image.fromarray(LR_pad).resize((w * self.scale, h * self.scale), Image.BICUBIC))


        h, w, _ = Ref.shape
        Ref_down = np.array(Image.fromarray(Ref).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        sample = {'HR': HR,
                  'HR_pad': HR_pad,
                  'LR': LR,
                  'LR_pad': LR_pad,
                  'LR_pad_down': LR_pad_down,
                  'Ref': Ref,  # Ref | LR_pad
                  'Ref_down': Ref_down,  # Ref_down | LR_pad_down
                  }

        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        pad_nums = [pad_H_t, pad_H_d, pad_H_l, pad_H_r]
        sample['pad_nums'] = pad_nums
        sample['HR_name'] = HR_name

        return sample


class TestSet_multi(Dataset):
    def __init__(self, args):
        self.input_list = sorted(glob.glob(os.path.join(args.data_root, args.dataset, 'test/CUFED5', '*_0.png')))
        self.ref_list1 = sorted(glob.glob(os.path.join(args.data_root, args.dataset, 'test/CUFED5', '*_1.png')))
        self.ref_list2 = sorted(glob.glob(os.path.join(args.data_root, args.dataset, 'test/CUFED5', '*_2.png')))
        self.ref_list3 = sorted(glob.glob(os.path.join(args.data_root, args.dataset, 'test/CUFED5', '*_3.png')))
        self.ref_list4 = sorted(glob.glob(os.path.join(args.data_root, args.dataset, 'test/CUFED5', '*_4.png')))
        self.ref_list5 = sorted(glob.glob(os.path.join(args.data_root, args.dataset, 'test/CUFED5', '*_5.png')))

        self.scale = args.sr_scale
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        HR_name = os.path.basename(self.input_list[idx])

        ### HR
        HR = cv2.imread(self.input_list[idx])
        Ref1 = cv2.imread(self.ref_list1[idx])
        Ref2 = cv2.imread(self.ref_list2[idx])
        Ref3 = cv2.imread(self.ref_list3[idx])
        Ref4 = cv2.imread(self.ref_list4[idx])
        Ref5 = cv2.imread(self.ref_list5[idx])
        Refs = [Ref1, Ref2, Ref3, Ref4, Ref5]

        h_new = 512
        w_new = 512
        Refs_pad = []
        for Ref in Refs:
            h, w, _ = Ref.shape
            pad_t = (h_new - h) // 2
            pad_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_l = (w_new - w) // 2
            pad_r = (w_new - w) // 2 + (w_new - w) % 2
            Ref_pad = cv2.copyMakeBorder(Ref.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_REPLICATE)
            Refs_pad.append(Ref_pad)
        Ref = cv2.vconcat(Refs_pad)

        # pad HR to be mutiple of 64
        h, w, c = HR.shape
        if h % 32 != 0 or w % 32 != 0:
            h_new = math.ceil(h / 32) * 32
            w_new = math.ceil(w / 32) * 32
            pad_H_t = (h_new - h) // 2
            pad_H_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_H_l = (w_new - w) // 2
            pad_H_r = (w_new - w) // 2 + (w_new - w) % 2
            HR_pad = cv2.copyMakeBorder(HR.copy(), pad_H_t, pad_H_d, pad_H_l, pad_H_r, cv2.BORDER_REPLICATE)
        else:
            pad_H_t, pad_H_d, pad_H_l, pad_H_r = 0, 0, 0, 0
            HR_pad = HR

        # pad ref to be multiple of 16
        h, w, c = Ref.shape
        if h % 16 != 0 or w % 16 != 0:
            h_new = math.ceil(h / 16) * 16
            w_new = math.ceil(w / 16) * 16
            Ref = cv2.copyMakeBorder(Ref.copy(), 0, h_new - h, 0, w_new - w, cv2.BORDER_REPLICATE)

        h, w, c = HR.shape
        LR = np.array(Image.fromarray(HR).resize((w//self.scale, h//self.scale), Image.BICUBIC))

        h, w, c = HR_pad.shape
        LR_pad = np.array(Image.fromarray(HR_pad).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        h, w, _ = Ref.shape
        Ref_down = np.array(Image.fromarray(Ref).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        sample = {'HR': HR,
                  'HR_pad': HR_pad,
                  'LR': LR,
                  'LR_pad': LR_pad,
                  'Ref': Ref,
                  'Ref_down': Ref_down}

        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        pad_nums = [pad_H_t, pad_H_d, pad_H_l, pad_H_r]
        sample['pad_nums'] = pad_nums
        sample['HR_name'] = HR_name

        return sample



