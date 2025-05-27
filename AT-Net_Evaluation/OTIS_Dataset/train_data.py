import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import os
import config_tdrn as config
from random import randrange
import torch
import random
from scipy import signal

class TrainData(data.Dataset):
    def __init__(self, crop_size, root_dir):
        super().__init__()
        self.crop_size = crop_size
        self.samples = []

        # Scan all Pattern folders inside root_dir
        #pattern_dirs = [d for d in os.listdir(root_dir) if d.startswith("Pattern") and os.path.isdir(os.path.join(root_dir, d))]
        # Folder names corresponding to the patterns (Pattern3, Pattern4, etc.)
        pattern_dirs = ['Pattern1','Pattern13', 'Pattern4', 'Pattern5', 'Pattern6','Pattern7', 
                            'Pattern8' ,'Pattern10', 'Pattern11','Pattern12', 'Pattern14', 'Pattern15', 'Pattern16']
        for pattern_name in sorted(pattern_dirs):
            pattern_path = os.path.join(root_dir, pattern_name)
            gt_path = os.path.join(pattern_path, 'GT', f'{pattern_name}_GT.png')

            # Collect all .png input images except the GT image
            input_images = sorted([
                f for f in os.listdir(pattern_path)
                if f.endswith('.png') and f != f'{pattern_name}_GT.png'
            ])

            for img_name in input_images:
                self.samples.append({
                    'input_path': os.path.join(pattern_path, img_name),
                    'gt_path': gt_path,
                    'img_id': img_name[:-4]  # strip .png
                })

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        #input_name = self.input_names[index]
        #img_id = input_name[:-4]  # Strip .png
        sample = self.samples[index]


        input_img = Image.open(sample['input_path']).convert('RGB')
        gt_img = Image.open(sample['gt_path']).convert('RGB')

        width, height = input_img.size

        # Random crop
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        turb_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # Transform to tensor
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_img = transform_input(input_img)
        gt = transform_gt(gt_img)
        

        # Validate channels
        if input_img.shape[0] != 3 or gt.shape[0] != 3:
            raise Exception(f'Bad image channel count for: {input_name}')

        return input_img, gt, sample["img_id"]

    def __getitem__(self, index):
        crop_width, crop_height = self.crop_size
        #input_name = self.input_names[index]
        #img_id = input_name[:-4]  # Strip .png
        sample = self.samples[index]

        input_img = Image.open(sample['input_path']).convert('RGB')
        gt_img = Image.open(sample['gt_path']).convert('RGB')

        width, height = input_img.size

        # Random crop
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        turb_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # Transform to tensor
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_img = transform_input(turb_crop_img)
        gt = transform_gt(gt_crop_img)

        if input_img.shape[0] != 3 or gt.shape[0] != 3:
            raise Exception(f'Bad image channel count for: {input_name}')

        return input_img, gt, sample["img_id"]

    def __len__(self):
        return len(self.samples)
