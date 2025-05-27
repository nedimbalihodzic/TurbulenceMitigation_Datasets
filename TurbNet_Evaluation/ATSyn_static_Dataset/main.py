import os
import sys
import time
import torch
import copy
import argparse
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.data_loader import TrainData, TestData
from model.TurbulenceNet import *
from utils.misc import to_psnr, adjust_learning_rate, print_log, ssim
from torchvision.models import vgg16
import torchvision.utils as utils
import math


import torch.ao.quantization as tq
from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig, FakeQuantize
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization.observer import HistogramObserver, default_observer, default_weight_observer, MovingAverageMinMaxObserver,MinMaxObserver
from torch.ao.quantization.qconfig import QConfig

from torch.quantization import QuantStub, DeQuantStub, convert, prepare, get_default_qconfig

from torch.fx.graph_module import GraphModule



# === Device config === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PTQWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        if isinstance(x, (list, tuple)):
            return [self.dequant(o) for o in x]
        return self.dequant(x)

def apply_static_quantization(model, calib_loader, device, save_path):
    model.eval()
    model = PTQWrapper(model).to(device)

    torch.backends.quantized.engine = "fbgemm"  # Qualcomm/QNN-compatible
    model.qconfig = get_default_qconfig("fbgemm")

    print("⚙️ Preparing for PTQ...")
    prepare(model, inplace=True)

    # Calibration with sanitized data
    print("📏 Running calibration...")
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            x = x.to(device)
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            model(x)
            if i >= 20:  # Enough batches for activation range estimation
                break

    print("🔁 Converting model to quantized version...")
    convert(model, inplace=True)

    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "ptq_model.pth"))
    print(f"✅ PTQ model saved to {save_path}/ptq_model.pth")

    return model

def lr_schedule_cosdecay(t, T, init_lr=1e-4):
    return 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr

def save_image(turb_images, image_names, loc):
    turb_images = torch.split(turb_images, 1, dim=0)
    for ind, img in enumerate(turb_images):
        utils.save_image(img, '{}/{}'.format(loc, '_'.join(image_names[ind].split("/")[-2:])))

def create_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir, "turb"))
        os.makedirs(os.path.join(save_dir, "gt"))
        os.makedirs(os.path.join(save_dir, "T"))
        os.makedirs(os.path.join(save_dir, "I"))
        os.makedirs(os.path.join(save_dir, "J"))
    else:
        print("Directory already exists!")
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
    crop_size = [64, 64]
    train_batch_size = 8
    test_batch_size = 2
    num_epochs = 50
    gps=3
    blocks=19
    lr=1e-4
    all_T = 100000
    old_val_psnr = 0
    alpha = 0.9
    save_dir = "./TurbNet_ATSyn"
    net = get_model()
    #net = torch.nn.DataParallel(net)
    net.cuda()
    #print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_data_loader = DataLoader(TrainData(crop_size), batch_size=train_batch_size, shuffle=True, num_workers=8)
    test_data_loader = DataLoader(TestData(), batch_size=test_batch_size, shuffle=True, num_workers=8)
    print("DATALOADER DONE!")
    
    #create_dir(save_dir)

    print("===> Training Start ...")
    for epoch in range(num_epochs):
        psnr_list = []
        start_time = time.time()

        # --- Save the network parameters --- #
        torch.save(net.state_dict(), '{}/turb_current{}.pth'.format(save_dir, epoch))

        for batch_id, train_data in enumerate(train_data_loader):
            if batch_id > 5000:
                break
            step_num = batch_id + epoch * 5000 + 1
            lr=lr_schedule_cosdecay(step_num,all_T)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            turb, gt = train_data
            turb = turb.cuda()
            gt = gt.cuda()

            optimizer.zero_grad()

            # --- Forward + Backward + Optimize --- #
            net.train()
            _, J, T, I = net(turb)
            Rec_Loss1 = F.smooth_l1_loss(J, gt)
            Rec_Loss2 = F.smooth_l1_loss(I, turb)
            loss = alpha * Rec_Loss1 + (1 - alpha) * Rec_Loss2

            loss.backward()
            optimizer.step()

            # --- To calculate average PSNR --- #
            psnr_list.extend(to_psnr(J, gt))

            if not (batch_id % 100):
                print('Epoch: {}, Iteration: {}, Loss: {:.3f}, Rec_Loss1: {:.3f}, Rec_loss2: {:.3f}'.format(epoch, batch_id, loss, Rec_Loss1, Rec_Loss2))

        # --- Calculate the average training PSNR in one epoch --- #
        train_psnr = sum(psnr_list) / len(psnr_list)
        print("Train PSNR : {:.3f}".format(train_psnr))

        # --- Use the evaluation model in testing --- #
        net.eval()
        one_epoch_time = time.time() - start_time
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
          val_psnr, val_ssim = validation(net, test_data_loader, save_dir)
          print(f"[Epoch {epoch + 1}/{num_epochs}] ⏱️ Time: {one_epoch_time:.2f}s")
          print(f"📈 Train PSNR: {train_psnr:.2f} dB | 📊 Val PSNR: {val_psnr:.2f} dB | 🧠 Val SSIM: {val_ssim:.4f}")

