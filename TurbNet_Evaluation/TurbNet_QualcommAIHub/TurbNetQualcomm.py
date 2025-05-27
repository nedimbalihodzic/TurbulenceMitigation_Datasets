import os
import sys
import time
import torch
import numpy as np
import argparse
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Resize
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.data_loader import TrainData, TestData
from model.TurbulenceNet import *
from utils.misc import to_psnr, adjust_learning_rate, print_log, ssim
from torchvision.models import vgg16
import qai_hub as hub  # Ensure you have the hub module installed
import torchvision.utils as utils
import math
import re
import yaml
with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6'
use_cuda = torch.cuda.is_available()
# === Device config === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def lr_schedule_cosdecay(t,T,init_lr=1e-4):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr

def save_image(turb_images, image_names, loc):
    turb_images = torch.split(turb_images, 1, dim=0)
    batch_num = len(turb_images)

    for ind in range(batch_num):
        # scaled_image = turb_images[ind].resize((400, 400), Image.ANTIALIAS)
        print('{}/{}'.format(loc,  '_'.join(image_names[ind].split("/")[-2:])))
        utils.save_image(turb_images[ind], '{}/{}'.format(loc, "output.png"))

def create_dir(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        print("Directory already exist!")
        sys.exit(0)
        
def validation(net, test_data_loader, save_dir, save_tag=True):
    print("Testing ...")
    psnr_list = []
    ssim_list = []

    net.eval()
    with torch.no_grad():
        for batch_id, val_data in enumerate(test_data_loader):
            turb, gt, image_names = val_data
            turb = turb.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            _, J, T, I = net(turb)

            psnr_list.extend(to_psnr(J, gt))
            ssim_list.extend(ssim(J, gt))

            #if save_tag:
            #    save_image(turb, image_names, save_dir + "/turb")
            #    save_image(gt, image_names, save_dir + "/gt")
            #    save_image(J, image_names, save_dir + "/J")
            #    save_image(T, image_names, save_dir + "/T")
            #    save_image(I, image_names, save_dir + "/I")

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim

if __name__ == "__main__":
    # === Paths ===
    onnx_model_path = "/content/Master/MyDrive/TurbNet-main/TurbNet-main/PTQ_ATSyn_exported/TurbNet_fp32.onnx"
    dataset_dir = "/content/Master/MyDrive/Training_Datasets_1800imgs/ATSyn_static_Dataset/landfill_00000302/turb"

    # === Prepare dataset ===
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    image_filenames = sorted(os.listdir(dataset_dir), key=extract_number)
    transform = Compose([Resize((64, 64)), ToTensor()])

    data = {"image": []}
    for filename in image_filenames:
        img_path = os.path.join(dataset_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # [1, 3, 64, 64]
        img_np = img_tensor.numpy().astype(np.float32)
        data["image"].append(img_np)

    # === Upload dataset to QAI Hub ===
    hub_dataset = hub.upload_dataset(data)
    print("📤 Dataset uploaded to QAI Hub")

    # === Compile float32 ONNX model for QCS8550 ===
    compile_job = hub.submit_compile_job(
        model=onnx_model_path,
        device=hub.Device("QCS8550 (Proxy)"),
        input_specs={"image": (1, 3, 64, 64)},
        options="--target_runtime qnn_lib_aarch64_android "
    )
    target_model = compile_job.get_target_model()
    print("⚙️ Model compiled for QCS8550")
    
    #models = hub.get_models()
    #for m in models:
    # print(f"🧠 {m.name} — ID: {m.model_id}")

    #model_id = "mn4kd4wwm"  # example
    #target_model = hub.get_model(model_id)

    # === Submit inference job ===
    inference_job = hub.submit_inference_job(
        model=target_model,
        device=hub.Device("QCS8550 (Proxy)"),
        inputs=hub_dataset
    )
    print("✅ Inference submitted")

    # === Submit profiling job ===
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=hub.Device("QCS8550 (Proxy)")
    )
    assert isinstance(profile_job, hub.ProfileJob)
    print("📊 Profiling job submitted")

