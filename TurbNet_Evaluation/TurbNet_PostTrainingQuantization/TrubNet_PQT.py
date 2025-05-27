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



# === Setup === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
save_dir = "./PTQ_ATSyn_exported"
os.makedirs(save_dir, exist_ok=True)

# === Load Data === #
train_data_loader = DataLoader(
    TrainData(crop_size=[64, 64]),
    batch_size=1,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# === Load Model === #
model_path = "/content/Master/MyDrive/TurbNet-main/TurbNet-main/TurbNet_ATSyn/turb_current49.pth"
net = net = get_model().to(device)  # ✅ Directly use TurbulenceNet
checkpoint = torch.load(model_path, map_location=device)

net.load_state_dict(checkpoint)
net.eval()

# === Calibration Pass === #
print("📏 Calibrating model with training data (for QNN)...")
with torch.no_grad():
    for i, (x, _) in enumerate(train_data_loader):
        x = x.to(device)
        _ = net(x)
        if i >= 100:
            break

# === Export to float32 ONNX === #
example_input = next(iter(train_data_loader))[0].to(device)


onnx_model_path = os.path.join(save_dir, "TurbNet_fp32.onnx")
torch.onnx.export(
    net,
    example_input,
    onnx_model_path,
    input_names=["image"],
    output_names=["output"],
    dynamic_axes={"image": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=13
)

print(f"✅ Float32 ONNX model exported for QNN PTQ: {onnx_model_path}")
