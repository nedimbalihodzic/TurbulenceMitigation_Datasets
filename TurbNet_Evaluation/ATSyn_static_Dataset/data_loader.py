import torch.utils.data as data
import os
from PIL import Image
from random import randrange
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop
import torch

class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir="/content/Master/MyDrive/Training_Datasets_1800imgs/ATSyn_static_Dataset/kasbah_00003305/turb"):
        super().__init__()

        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

        self.inp_filenames = []
        self.gt_filenames = []

        # Iterate through each scene directory
        for scene_folder in sorted(os.listdir(train_data_dir)):
            scene_path = os.path.join(train_data_dir, scene_folder)
            turb_path = os.path.join(scene_path, 'turb')
            gt_path = os.path.join(scene_path, 'gt.jpg')

            if not os.path.exists(turb_path) or not os.path.exists(gt_path):
                print(f"Skipping {scene_folder}: missing turb/ or gt.png")
                continue

            # Collect all distorted images and associate them with the single GT image
            for fname in sorted(os.listdir(turb_path)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.inp_filenames.append(os.path.join(turb_path, fname))
                    self.gt_filenames.append(gt_path)

        assert len(self.inp_filenames) == len(self.gt_filenames), "Mismatch between input and ground truth files"

    def __len__(self):
        return len(self.inp_filenames)

    def __getitem__(self, index):
        crop_width, crop_height = self.crop_size

        resample_mode = Image.Resampling.LANCZOS

        turb_name = self.inp_filenames[index]
        gt_name = self.gt_filenames[index]

        turb_img = Image.open(turb_name).convert('RGB')
        gt_img = Image.open(gt_name).convert('RGB')

        width, height = turb_img.size

        # Resize if image is smaller than crop size
        if width < crop_width or height < crop_height:
            if width < height:
                new_width = 400
                new_height = int(height * 400 / width)
            else:
                new_height = 400
                new_width = int(width * 400 / height)
            turb_img = turb_img.resize((new_width, new_height), resample_mode)
            gt_img = gt_img.resize((new_width, new_height), resample_mode)
            width, height = new_width, new_height

        # Random crop
        x = randrange(0, width - crop_width + 1)
        y = randrange(0, height - crop_height + 1)
        turb_crop_img = turb_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        #transform = Compose([ToTensor()])
        transform = Compose([
        ToTensor()
        ])
        turb = transform(turb_crop_img)
        gt = transform(gt_crop_img)

        if turb.shape[0] != 3 or gt.shape[0] != 3:
            raise Exception(f'Bad image channel: {turb_name}')

        return turb, gt


class TestData(data.Dataset):
    def __init__(self, crop_size=[64, 64], val_data_dir="/content/Master/MyDrive/Test_Dataset/ATSyn_TestDataset"):
        super().__init__()
        self.crop_size = crop_size
        self.val_data_dir = val_data_dir

        self.inp_filenames = []
        self.gt_filenames = []

        # Iterate through scene folders
        for scene_folder in sorted(os.listdir(val_data_dir)):
            scene_path = os.path.join(val_data_dir, scene_folder)
            turb_path = os.path.join(scene_path, 'turb')
            gt_path = os.path.join(scene_path, 'gt.jpg')

            if not os.path.exists(turb_path) or not os.path.exists(gt_path):
                print(f"Skipping {scene_folder}: missing turb/ or gt.jpg")
                continue

            for fname in sorted(os.listdir(turb_path)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.inp_filenames.append(os.path.join(turb_path, fname))
                    self.gt_filenames.append(gt_path)

        assert len(self.inp_filenames) == len(self.gt_filenames), "Mismatch between input and ground truth files"

        self.transform = Compose([
            CenterCrop(self.crop_size),
            ToTensor()
        ])

    def __len__(self):
        return len(self.inp_filenames)

    def __getitem__(self, index):
        turb_name = self.inp_filenames[index]
        gt_name = self.gt_filenames[index]

        turb_img = Image.open(turb_name).convert('RGB')
        gt_img = Image.open(gt_name).convert('RGB')

        turb = self.transform(turb_img)
        gt = self.transform(gt_img)

        if turb.shape[0] != 3 or gt.shape[0] != 3:
            raise Exception(f'Bad image channel: {turb_name}')

        return turb, gt, turb_name
