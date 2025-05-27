import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import os
import random
from random import randrange

class TrainData(data.Dataset):
    def __init__(self, crop_size, root_dir):
        super().__init__()
        self.crop_size = crop_size
        self.samples = []

        # Iterate through all the scene folders inside the root_dir (scene_0001, scene_0002, ...)
        scene_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        
        # Loop through each scene folder
        for scene_name in sorted(scene_dirs):
            scene_path = os.path.join(root_dir, scene_name)
            
            # Path to the ground truth image
            gt_path = os.path.join(scene_path, 'gt.jpg')
            
            # Path to the 'turb' folder that contains degraded images
            turb_dir = os.path.join(scene_path, 'turb')
            
            # Ensure the 'turb' folder exists and contains exactly 50 images
            turb_images = sorted([f for f in os.listdir(turb_dir) if f.endswith('.jpg')])
            
            # Add the samples: each entry contains the path to the ground truth image and one degraded image
            for turb_img_name in turb_images:
                self.samples.append({
                    'turb_path': os.path.join(turb_dir, turb_img_name),
                    'gt_path': gt_path,
                    'img_id': turb_img_name[:-4]  # Strip .png extension for image ID
                })

    def __getitem__(self, index):
        crop_width, crop_height = self.crop_size
        #input_name = self.input_names[index]
        #img_id = input_name[:-4]  # Strip .png
        sample = self.samples[index]

        input_img = Image.open(sample['turb_path']).convert('RGB')
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
