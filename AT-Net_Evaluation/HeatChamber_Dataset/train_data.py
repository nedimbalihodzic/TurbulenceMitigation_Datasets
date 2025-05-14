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

        # Scan all scene folders inside root_dir (scene_0001, scene_0002, ...)
        scene_dirs = [d for d in os.listdir(root_dir) if d.startswith("scene_") and os.path.isdir(os.path.join(root_dir, d))]

        for scene_name in sorted(scene_dirs):
            scene_path = os.path.join(root_dir, scene_name)
            gt_path = os.path.join(scene_path, 'gt.png')  # Ground truth image

            turb_dir = os.path.join(scene_path, 'turb')  # Turbulence images folder
            turb_images = sorted([f for f in os.listdir(turb_dir) if f.endswith('.png')]) # Omit last image


            # Collect all turbulence images along with the corresponding gt image path
            for turb_img_name in turb_images:
                self.samples.append({
                    'turb_path': os.path.join(turb_dir, turb_img_name),
                    'gt_path': gt_path,
                    'img_id': turb_img_name[:-4]  # Strip .png extension for image ID
                })

    def __getitem__(self, index):
        crop_width, crop_height = self.crop_size
        sample = self.samples[index]

        # Load turbulence and ground truth images
        turb_img = Image.open(sample['turb_path']).convert('RGB')
        gt_img = Image.open(sample['gt_path']).convert('RGB')

        # Resize images if needed (optional step, adjust size if necessary)
        if turb_img.size != (128, 128):
            turb_img = turb_img.resize((128+32, 128+42), Image.Resampling.LANCZOS)
            gt_img = gt_img.resize((128+32, 128+42), Image.Resampling.LANCZOS)

        # Transform to tensor
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        turb_img = transform_input(turb_img)
        gt = transform_gt(gt_img)

        # Optional: Crop image (adjusting the crop to random position)
        #x, y = random.randint(0, turb_img.size(1) - crop_width), random.randint(0, turb_img.size(2) - crop_height)
        #turb_img = turb_img[:, x:x + crop_width, y:y + crop_height]
        #gt = gt[:, x:x + crop_width, y:y + crop_height]

        # Validate channels (check that it's RGB)
        if turb_img.shape[0] != 3 or gt.shape[0] != 3:
            raise Exception(f'Bad image channel count for: {sample["turb_path"]}')

        return turb_img, gt, sample["img_id"]

    def __len__(self):
        return len(self.samples)
