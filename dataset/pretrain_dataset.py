import os
import json
import math
import numpy as np

import random
from random import randint, shuffle
from random import random as rand

from PIL import Image
from PIL import ImageFile

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip, resize, pad
from torchvision.transforms import InterpolationMode

from dataset.utils import pre_caption


class pre_dataset(Dataset):
    def __init__(self, config, box_transform):
        self.image_root = config['image_root']
        self.max_words = config['max_words']
        self.icfg_rstp = config['icfg_rstp']
        self.eda = config['eda']

        self.h = config['h']
        self.w = config['w']
        self.box_transform = box_transform
        self.patch_size = config['patch_size']

        assert self.h % self.patch_size == 0
        assert self.w % self.patch_size == 0
        self.num_patch_h = int(self.h / self.patch_size)
        self.num_patch_w = int(self.w / self.patch_size)

        print('train_file_regions', config['train_file_regions'])
        ann_file = config['train_file_regions']
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.img_ids = {}
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        print('### image id mums:', n)


        self.box_ids = {}
        for ann in self.ann:
            for phrase in ann['phrases']:
                if phrase not in self.box_ids.keys():
                    self.box_ids[phrase] = n
                    n += 1
        print('### after add box id nums:', n)


    def __len__(self):
        return len(self.ann)


    def get_image_attns(self, x, y, w, h):
        mask_ratio = 1

        x_min = min(math.floor(x / self.patch_size), self.num_patch_w - 1)
        x_max = max(x_min + 1, min(math.ceil((x + w) / self.patch_size), self.num_patch_w))  # exclude

        y_min = min(math.floor(y / self.patch_size), self.num_patch_h - 1)
        y_max = max(y_min + 1, min(math.ceil((y + h) / self.patch_size), self.num_patch_h))  # exclude

        image_atts = [1] * (1 + self.num_patch_w * self.num_patch_h)
        for j in range(x_min, x_max):
            for i in range(y_min, y_max):
                index = self.num_patch_w * i + j + 1
                assert (index > 0) and (
                        index <= self.num_patch_h * self.num_patch_w), f"patch index out of range, " \
                                                                       f"index: {index} " \
                                                                       f"x_min, x_max {x_min} {x_max} " \
                                                                       f"y_min, y_max {y_min} {y_max}"
                if rand() < mask_ratio:
                    image_atts[index] = 0

        if sum(image_atts) == len(image_atts):
            print('### only 1 patch masked!')
            index = self.num_patch_w * y_min + x_min + 1
            image_atts[index] = 0

        return torch.tensor(image_atts, dtype=torch.long)


    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')

        W, H = image.size
        box_nums = len(ann['boxes'])
        if box_nums < 1:
            x, y, w, h = 0.5, 0.5, 1, 1
        else:
            box_i = random.choice(range(box_nums))
            x, y, w, h = ann['boxes'][box_i]

        # random crop
        x, y, w, h = int(x * W), int(y * H), int(w * W), int(h * H)
        x, y = (x - 0.5 * w), (y - 0.5 * h)
        x, y = max(x, 0), max(y, 0)
        w, h = min(w, W - x), min(h, H - y)
        assert (x >= 0) and (y >= 0) and (w > 0) and (h > 0), "boxes invalid"
        assert (x + w <= W) and (y + h <= H), "boxes invalid {x} {w} {W} {y} {h} {H}"

        image = pad(image, 10)
        x, y = x + 10, y + 10
        W, H = W + 20, H + 20
        x0, y0 = random.randint(0, math.floor(x)), random.randint(0, math.floor(y))
        x1, y1 = random.randint(min(math.ceil(x + w), W), W), random.randint(min(math.ceil(y + h), H), H)
        w0, h0 = x1 - x0, y1 - y0
        assert (x0 >= 0) and (y0 >= 0) and (x0 + w0 <= W) and (y0 + h0 <= H) and (w0 > 0) and (
                h0 > 0), "boxes randomcrop, invalid"
        image = image.crop((x0, y0, x0 + w0, y0 + h0))
        W, H = image.size

        do_hflip = False
        if rand() < 0.5:
            image = hflip(image)
            do_hflip = True

        image = resize(image, [self.h, self.w], interpolation=InterpolationMode.BICUBIC)
        image = self.box_transform(image)

        # axis transform: after crop
        x = x - x0
        y = y - y0

        if do_hflip:  # flipped applied
            x = (W - x) - w  # W is w0

        # resize applied
        x = self.w / W * x
        w = self.w / W * w
        y = self.h / H * y
        h = self.h / H * h

        # center_x = x + 1 / 2 * w
        # center_y = y + 1 / 2 * h
        # box = torch.tensor([center_x / self.w, center_y / self.h, w / self.w, h / self.h], dtype=torch.float)

        box_atts = self.get_image_attns(x, y, w, h)

        img_id = ann['image_id']
        
        cap = ann['caption']
        caption = pre_caption(cap, self.max_words, icfg_rstp=self.icfg_rstp)
        if self.eda:
            caption_eda = pre_caption(cap, self.max_words, self.icfg_rstp, True)
        else:
            caption_eda = {}

        if box_nums < 1:
            box_caption = pre_caption(cap, self.max_words, icfg_rstp=self.icfg_rstp)
            label = self.img_ids[img_id]
        else:
            box_caption = pre_caption(ann['phrases'][box_i], self.max_words)
            label = self.box_ids[ann['phrases'][box_i]]

        return image, caption, caption_eda, box_atts, box_caption, label, self.img_ids[img_id]
