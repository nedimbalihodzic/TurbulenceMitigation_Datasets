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
from torch.ao.quantization.observer import HistogramObserver, default_weight_observer, MovingAverageMinMaxObserver,MinMaxObserver
from torch.ao.quantization.qconfig import QConfig

from torch.fx.graph_module import GraphModule

class NaNSafeWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        x = torch.nan_to_num(x)
        out = self.module(x)
        if isinstance(out, tuple):
            return tuple(torch.nan_to_num(o) for o in out)
        return torch.nan_to_num(out)

# === Device config === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Helper to recreate observer with same args
def reset_observer(module):
    if isinstance(module.activation_post_process, HistogramObserver):
        obs_cls = type(module.activation_post_process)
        obs_args = module.activation_post_process.__dict__
        # Recreate clean observer
        new_obs = obs_cls(
            dtype=obs_args['dtype'],
            qscheme=obs_args['qscheme'],
            quant_min=obs_args['quant_min'],
            quant_max=obs_args['quant_max'],
            eps=obs_args['eps'],
            reduce_range=obs_args['reduce_range']
        )
        module.activation_post_process = new_obs
        print(f"🔄 Reset corrupted observer in: {module}")

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
    train_batch_size = 1
    test_batch_size = 1
    num_epochs = 50
    gps = 3
    blocks = 19
    lr = 1e-4
    all_T = 100000
    alpha = 0.9
    save_dir = "./QAT_fineTunning_attempt_fx"
    

    # === DataLoader === #
    train_data_loader = DataLoader(
        TrainData(crop_size),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    # === Model === #
    net = get_model().to(device)

    # Load checkpoint
    model_path = "/content/Master/MyDrive/TurbNet-main/TurbNet-main/HeatChamber_Dataset/turb_current99.pth"
    checkpoint = torch.load(model_path, map_location=device)

    # OPTIONAL: sanitize checkpoint weights before loading
    for k, v in checkpoint.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            print(f"🔧 Sanitizing checkpoint param: {k}")
            checkpoint[k] = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)

    # Load weights into base model (before wrapping)
    net.load_state_dict(checkpoint)

    # Now wrap AFTER loading weights
    

    # Sanitize again just in case
    for name, param in net.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"🔧 Fixing weights in {name}")
            with torch.no_grad():
                param.copy_(torch.nan_to_num(param, nan=0.0, posinf=1.0, neginf=-1.0))

    net.eval()

    example_input = next(iter(train_data_loader))[0].to(device)

    with torch.no_grad():
     outputs = net(example_input)
     for i, out in enumerate(outputs):
        if not torch.isfinite(out).all():
            print(f"❌ Original model output {i} has NaNs")
        else:
            print(f"✅ Output {i} is clean")

    with torch.no_grad():
     for turb, gt in train_data_loader:
         turb = turb.to(device)
         output = net(turb)
         print("✅ Original model output is finite:", [torch.isfinite(o).all().item() for o in output])
         break
    
    # Better stability: use HistogramObserver for activations
    activation_observer = MinMaxObserver.with_args(
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    quant_min=0,
    quant_max=255,
    eps=1e-6
    )

    # === Prepare FX QAT Config === #
    #qconfig = get_default_qat_qconfig("fbgemm")
    
    qconfig = QConfig(
    activation=FakeQuantize.with_args(observer=activation_observer),
    weight=default_weight_observer
    )
    #qconfig_dict = {"": qconfig}
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    
    # Exclude the problematic qkv conv layers from QAT
    qconfig_mapping = qconfig_mapping.set_module_name("encoder_level1.0.attn.qkv", None)
    qconfig_mapping = qconfig_mapping.set_module_name("encoder_level1.1.attn.qkv", None)
    qconfig_mapping = qconfig_mapping.set_module_name("encoder_level1.2.attn.qkv", None)
    qconfig_mapping = qconfig_mapping.set_module_name("encoder_level1.3.attn.qkv", None)

    example_input, _ = next(iter(train_data_loader))
    example_input = example_input.to(device)

    net_prepared = prepare_qat_fx(net, qconfig_mapping, example_input)

    
    # === Disable observers for warmup ===
    for m in net_prepared.modules():
        if isinstance(m, tq.FakeQuantize):
            m.disable_observer()

    # === Warmup Fake Quantizers ===
    net_prepared.train()
    with torch.no_grad():
        for i, (turb, gt) in enumerate(train_data_loader):
            turb = turb.to(device)
            turb = turb.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)
            _ = net_prepared(turb)
            if i > 10:
                break

    optimizer = torch.optim.Adam(net_prepared.parameters(), lr=lr)
  
    print("===> Training Start ...")

    for name, module in net_prepared.named_modules():
      if hasattr(module, 'weight_fake_quant'):
          fq = module.weight_fake_quant
          if hasattr(fq, 'activation_post_process'):
              obs = fq.activation_post_process
              if hasattr(obs, 'min_val') and hasattr(obs, 'max_val'):
                  print(f"[{name}] -> min: {obs.min_val}, max: {obs.max_val}")

    for name, submodule in net_prepared.named_modules():
      if isinstance(submodule, torch.nn.Conv2d) and submodule.in_channels == 3 and submodule.out_channels == 48:
          fq = getattr(submodule, 'weight_fake_quant', None)
          if isinstance(fq, torch.ao.quantization.FakeQuantize):
            fq.disable_observer()
            print(f"⛔ Disabled observer in {name}")
    

    
    def clamp_forward_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        return output

    conv_target = None
    for name, module in net_prepared.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3 and module.out_channels == 48:
            conv_target = module
            break

    if conv_target:
        conv_target.register_forward_hook(clamp_forward_hook)
        print("🩹 Clamping forward output of patch_embed.proj")

     

    
    def nan_guard_hook(module, input, output):
        if isinstance(output, torch.Tensor) and not torch.isfinite(output).all():
            print(f"❌ Detected NaNs in {module}")
            raise RuntimeError(f"NaNs in module: {module}")
        return torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

    # Apply globally (before training loop)
    for name, module in net_prepared.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.LayerNorm, nn.BatchNorm2d)):
            module.register_forward_hook(nan_guard_hook)


    warmup_batches = 100
    batch_counter = 0

    with torch.no_grad():
      for name, param in net_prepared.named_parameters():
          if torch.isnan(param).any() or torch.isinf(param).any():
              print(f"🚑 Fixing weights again in {name}")
              param.copy_(torch.nan_to_num(param, nan=0.0, posinf=1.0, neginf=-1.0))
    
    for name, module in net_prepared.named_modules():
      if hasattr(module, 'weight_fake_quant'):
          fq = module.weight_fake_quant
          if hasattr(fq, 'activation_post_process'):
              obs = fq.activation_post_process
              print(f"{name} -> min: {getattr(obs, 'min_val', None)}, max: {getattr(obs, 'max_val', None)}")


    for epoch in range(num_epochs):
        psnr_list = []
        start_time = time.time()

        torch.save(net_prepared.state_dict(), f'{save_dir}/turb_current{epoch}.pth')

        for batch_id, train_data in enumerate(train_data_loader):
            if batch_id > 5000:
                break

            step_num = batch_id + epoch * 5000 + 1
            lr = lr_schedule_cosdecay(step_num, all_T)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            turb, gt = train_data
            turb = turb.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            # NaN-safe input
            turb = turb.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)

            optimizer.zero_grad()
            net_prepared.train()

            try:
                try:
                    turb = turb.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)
                    outputs = net_prepared(turb)
                    for i, out in enumerate(outputs):
                        if not torch.isfinite(out).all():
                            print(f"❌ Output {i} has NaNs or Infs!")
                            raise RuntimeError("Non-finite value in model output")

                    _, J, T, I = outputs

                except Exception as e:
                    print("🚨 Crash during net_prepared forward pass!")
                    # Register hooks to trace the exact failing layer
                    for name, module in net_prepared.named_modules():
                        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
                            def register_hook(n, m):
                                def hook(mod, inp, outp):
                                    if not torch.isfinite(outp).all():
                                        print(f"❌ Module {n} produced non-finite output")
                                        raise RuntimeError(f"Non-finite output in {n}")
                                return m.register_forward_hook(hook)
                            register_hook(name, module)

                    _ = net_prepared(turb)  # Re-run to trigger hooks
                    raise e  # Re-raise

                # Loss and backward
                Rec_Loss1 = F.smooth_l1_loss(J, gt)
                Rec_Loss2 = F.smooth_l1_loss(I, turb)
                loss = alpha * Rec_Loss1 + (1 - alpha) * Rec_Loss2

                assert torch.isfinite(loss), "Loss is NaN or Inf"
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net_prepared.parameters(), max_norm=1.0)
                optimizer.step()

                psnr_list.extend(to_psnr(J, gt))

                if batch_id % 100 == 0:
                    print(f"Epoch: {epoch}, Iter: {batch_id}, Loss: {loss:.3f}, Rec_Loss1: {Rec_Loss1:.3f}, Rec_Loss2: {Rec_Loss2:.3f}")

                batch_counter += 1

                if batch_counter == warmup_batches:
                    print("🔓 Enabling observers after warmup...")
                    for m in net_prepared.modules():
                        if isinstance(m, tq.FakeQuantize):
                            m.enable_observer()
                    print("✅ Enabled observers for QAT")
                    force_reset_fake_quant_observers(net_prepared)

                    print("🔎 Verifying all observers for NaNs after enabling...")
                    for name, module in net_prepared.named_modules():
                        if isinstance(module, tq.FakeQuantize):
                            try:
                                test_input = torch.rand(1, 3, 64, 64, device=device)
                                with torch.no_grad():
                                    _ = module(test_input)
                                print(f"🟢 Observer OK: {name}")
                            except Exception as e:
                                print(f"🚨 Crashed observer: {name}")
                                print(f"❌ Disabling this observer to prevent failure.")
                                module.disable_observer()

            except AssertionError as e:
                print(f"⚠️ Skipping batch {batch_id} due to: {e}")
                continue

        train_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
        print(f"Train PSNR : {train_psnr:.3f}")


        #Optional evaluation
        #val_psnr, val_ssim = validation(net, test_data_loader, save_dir)
        #one_epoch_time = time.time() - start_time
        # print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, "train", save_dir)
        #print(f"Train PSNR : {train_psnr:.3f}, Val PSNR : {val_psnr:.3f}, Val SSIM : {val_ssim:.4f}")

        # torch.cuda.empty_cache()  # Clear memory at the end of the epoch
    # === Export quantized model === #
    net_prepared.eval()
    quantized_model = convert_fx(net_prepared)
    torch.save(quantized_model.state_dict(), os.path.join(save_dir, "quantized_turbnet_fx.pth"))
    torch.onnx.export(quantized_model, example_input, os.path.join(save_dir, "turbnet_qat_fx.onnx"),
                      input_names=['input'], output_names=['output'], opset_version=13)

    print("===> Quantized model saved and exported.")

