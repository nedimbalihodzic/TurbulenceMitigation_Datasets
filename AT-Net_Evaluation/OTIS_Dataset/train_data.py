import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import os
import config_tdrn as config
import torch
import random
from scipy import signal

class TrainData(data.Dataset):
    def __init__(self, crop_size, root_dir):
        super().__init__()
        self.crop_size = crop_size
        self.samples = []

        # Scan all Pattern folders inside root_dir
        pattern_dirs = [d for d in os.listdir(root_dir) if d.startswith("Pattern") and os.path.isdir(os.path.join(root_dir, d))]

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


        # Load images
        #input_path = os.path.join(self.pattern_dir, input_name)
        #gt_img = Image.open(self.gt_path).convert('RGB')
        #input_img = Image.open(input_path).convert('RGB')
        input_img = Image.open(sample['input_path']).convert('RGB')
        gt_img = Image.open(sample['gt_path']).convert('RGB')

        # Resize if needed
        if input_img.size != (256, 256):
            input_img = input_img.resize((256+32, 256+42), Image.Resampling.LANCZOS)
            gt_img = gt_img.resize((256+32, 256+42), Image.Resampling.LANCZOS)

        # Transform to tensor
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_img = transform_input(input_img)
        gt = transform_gt(gt_img)

        ch, w, h = input_img.size()
        input_img = input_img.numpy()

        # Apply blur kernel
        #index_blr = random.randint(0, config.Num_k - 10)
        #for c in range(3):
        #    input_img[c, :, :] = signal.convolve(input_img[c, :, :], config.kernels[index_blr, :, :], mode='same')

        # Apply warp
        #index_d = random.randint(0, config.Num_D - 1)
        #xx, yy = np.meshgrid(np.arange(0, h), np.arange(0, w))
        #X_new = xx + config.Warp_mat[index_d, 0, :, :]
        #Y_new = yy + config.Warp_mat[index_d, 1, :, :]
        #for c in range(3):
        #    input_img[c, :, :] = config.warp(input_img[c, :, :], X_new, Y_new)

        #input_img = torch.from_numpy(input_img)

        ## Crop
        #x, y = 21, 16
        #input_img = input_img[:, x:x + crop_width, y:y + crop_height]
        #gt = gt[:, x:x + crop_width, y:y + crop_height]

        # Validate channels
        if input_img.shape[0] != 3 or gt.shape[0] != 3:
            raise Exception(f'Bad image channel count for: {input_name}')

        return input_img, gt, sample["img_id"]

    def __getitem__(self, index):
        crop_width, crop_height = self.crop_size
        #input_name = self.input_names[index]
        #img_id = input_name[:-4]  # Strip .png
        sample = self.samples[index]


        # Load images
        #input_path = os.path.join(self.pattern_dir, input_name)
        #gt_img = Image.open(self.gt_path).convert('RGB')
        #input_img = Image.open(input_path).convert('RGB')
        input_img = Image.open(sample['input_path']).convert('RGB')
        gt_img = Image.open(sample['gt_path']).convert('RGB')

        # Resize if needed
        if input_img.size != (128, 128):
            input_img = input_img.resize((128+32, 128+42), Image.Resampling.LANCZOS)
            gt_img = gt_img.resize((128+32, 128+42), Image.Resampling.LANCZOS)

        # Transform to tensor
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_img = transform_input(input_img)
        gt = transform_gt(gt_img)

        ch, w, h = input_img.size()
        input_img = input_img.numpy()
        if input_img.shape[0] != 3 or gt.shape[0] != 3:
            raise Exception(f'Bad image channel count for: {input_name}')

        return input_img, gt, sample["img_id"]
        #return self.get_images(index)

    def __len__(self):
        return len(self.samples)
