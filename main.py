import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from torchvision import models as torchvision_models
from numpy.random import randint
from torch.utils.data import Subset

# Import the modified dataset utilities
from datasets import load_dataset
from PED2 import Ped2Dataset
from datasets.datasets_utils import DataAugmentationSiT, GMML_replace_list
from datasets.ped2_data_augmentation import DataAugmentationPed2

import utils
import vision_transformer as vits
from vision_transformer import RECHead

def get_args_parser():
    parser = argparse.ArgumentParser('SiTv2', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'], help="Name of architecture to train.")
    parser.add_argument('--img_size', default=224, type=int, help="Input size to the Transformer.")
    
    # Reconstruction parameters
    parser.add_argument('--recons_blocks', default='6-8-10-12', type=str, help="""Reconstruct the input back from the 
                        given blocks, empty string means no reconstruction will be applied. (Default: '6-8-10-12') """)
                        
    parser.add_argument('--drop_perc', type=float, default=0.75, help='Drop X percentage of the input image')
    parser.add_argument('--drop_replace', type=float, default=0.3, help='Replace X percentage of the input image')
    
    parser.add_argument('--drop_align', type=int, default=1, help='Align drop with patches')
    parser.add_argument('--drop_type', type=str, default='noise', help='Drop Type.')
    parser.add_argument('--drop_only', type=int, default=1, help='Align drop with patches')

    # Dataset
    parser.add_argument('--data_set', default='PED2', type=str, 
                        choices=['MNIST', 'PED2', 'CIFAR10', 'CIFAR100', 'Flowers', 'Aircraft', 'Cars', 'ImageNet5p', 'ImageNet10p', 'ImageNet', 'TinyImageNet', 'PASCALVOC', 'MSCOCO', 'VGenome', 'Pets'], 
                        help='Name of the dataset.')
    parser.add_argument('--data_location', default='./data', type=str, help='Dataset location.')
    parser.add_argument('--val_split', type=float, default=0.1, help='Percentage of data to use for validation')

    # Hyper-parameters
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size per GPU.")
    parser.add_argument('--epochs', default=200, type=int, help="Number of epochs of training.")
    
    parser.add_argument('--weight_decay', type=float, default=0.04, help="weight decay")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="Final value of the weight decay.")
    
    parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Target LR at the end of optimization.")
    
    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="Whether or not to use half precision for training.")   
    parser.add_argument('--clip_grad', type=float, default=3.0, help="Maximal parameter gradient norm.")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument("--use_smooth_l1", type=utils.bool_flag, default=True, help="Use Smooth L1 Loss instead of L1 Loss")
    parser.add_argument("--loss_beta", type=float, default=0.1, help="Beta parameter for Smooth L1 Loss")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.25, 1.), help="Scale range of global crops")
    parser.add_argument('--local_crops_number', type=int, default=0, help="Number of local crops.")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4), help="Scale range of local crops")
    
    # Misc
    parser.add_argument('--output_dir', default="checkpoints/PED2", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    return parser


def train_SiTv2(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    args.epochs += 1

    # Preparing Dataset
    if args.data_set == 'PED2':
        transform = DataAugmentationPed2(args)
    else:
        transform = DataAugmentationSiT(args)
        
    full_dataset, _ = load_dataset.build_dataset(args, True, trnsfrm=transform)
    
    # Split dataset into train and validation
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(args.val_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    
    print(f"==> {args.data_set} training set is loaded.")
    print(f"-------> The training dataset consists of {len(train_dataset)} images.")
    print(f"-------> The validation dataset consists of {len(val_dataset)} images.")

    # Create Transformer with adjusted parameters
    SiT_model = vits.__dict__[args.model](
        img_size=[args.img_size],
        drop_path_rate=0.2,  # Increased stochastic depth
        attn_drop_rate=0.1,  # Add attention dropout
        drop_rate=0.1,       # Add regular dropout
        qkv_bias=True        # Enable bias in attention
    )
    n_params = sum(p.numel() for p in SiT_model.parameters() if p.requires_grad)
    embed_dim = SiT_model.embed_dim
    
    # Create reconstruction head with the correct img_size
    rec_head = RECHead(embed_dim, patch_size=SiT_model.patch_embed.patch_size, img_size=args.img_size)
    
    SiT_model = FullpiplineSiT(SiT_model, rec_head)
    SiT_model = SiT_model.cuda()
        
    SiT_model = nn.parallel.DistributedDataParallel(SiT_model, device_ids=[args.gpu])
    print(f"==> {args.model} model is created.")
    print(f"-------> The model has {n_params} parameters.")
    
    # Create Optimizer with improved parameters
    params_groups = utils.get_params_groups(SiT_model)
    optimizer = torch.optim.AdamW(
        params_groups,
        betas=(0.9, 0.999),
        eps=1e-8
    )  # to use with ViTs

    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # Initialize improved schedulers with longer warmup
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256.,  
        args.min_lr, 
        args.epochs, 
        len(train_loader), 
        warmup_epochs=min(int(args.epochs * 0.1), 15)  # Dynamic warmup period
    )
    wd_schedule = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, len(train_loader))

    # Resume Training if exist
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore, SiT_model=SiT_model,
        optimizer=optimizer, fp16_scaler=fp16_scaler)
    start_epoch = to_restore["epoch"]

    # Initialize tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    start_time = time.time()
    print(f"==> Start training from epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        # Train an epoch
        train_stats = train_one_epoch(SiT_model, train_loader, optimizer, lr_schedule, wd_schedule,
            epoch, fp16_scaler, args)
        
        # Validate the model
        val_stats = validate_one_epoch(SiT_model, val_loader, epoch, args)
        
        # Track losses
        train_losses.append(train_stats['loss'])
        val_losses.append(val_stats['loss'])
        
        # Save best model
        is_best = val_stats['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_stats['loss']
            utils.save_on_master({
                'SiT_model': SiT_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
                'val_loss': best_val_loss
            }, os.path.join(args.output_dir, 'best_model.pth'))

        # Regular checkpoint saving
        save_dict = {
            'SiT_model': SiT_model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1, 
            'args': args,
            'val_loss': val_stats['loss'],
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        
        # Log stats
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch
        }
        
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            # Plot loss curves
            if epoch % args.saveckp_freq == 0:
                plot_loss_curves(train_losses, val_losses, args.output_dir)
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(SiT_model, data_loader, optimizer, lr_schedule, wd_schedule, epoch, fp16_scaler, args):
    
    save_recon = os.path.join(args.output_dir, 'reconstruction_samples')
    Path(save_recon).mkdir(parents=True, exist_ok=True)
    bz = args.batch_size
    plot_ = True
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, ((clean_crops, corrupted_crops, masks_crops), _) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        clean_crops = [im.cuda(non_blocking=True) for im in clean_crops]
        corrupted_crops = [im.cuda(non_blocking=True) for im in corrupted_crops]
        masks_crops = [im.cuda(non_blocking=True) for im in masks_crops]
        
        if args.drop_replace > 0:
            corrupted_crops, masks_crops = GMML_replace_list(clean_crops, corrupted_crops, masks_crops, drop_type=args.drop_type,
                                                          max_replace=args.drop_replace, align=args.drop_align)
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            s_recons_g, s_recons_l = SiT_model(corrupted_crops, args.recons_blocks)
            
            #-------------------------------------------------
            # Use Smooth L1 Loss instead of L1 Loss for better convergence
            if args.use_smooth_l1:
                recloss = F.smooth_l1_loss(s_recons_g, torch.cat(clean_crops[0:2]), reduction='none', beta=args.loss_beta)
            else:
                recloss = F.l1_loss(s_recons_g, torch.cat(clean_crops[0:2]), reduction='none')
            
            # Apply focus loss to prioritize difficult regions
            if args.drop_only == 1:
                # Get the mask
                mask = torch.cat(masks_crops[0:2]) == 1
                # Calculate element-wise error
                error = recloss.detach()
                # Create focus weights - give more weight to higher errors
                focus_weights = torch.ones_like(error)
                focus_weights[mask] = 1.0 + 0.5 * (error[mask] / error[mask].mean())
                # Apply weights to loss
                weighted_loss = recloss * focus_weights
                loss = weighted_loss[mask].mean()
            else:
                loss = recloss.mean()
                
            if len(clean_crops) > 2:
                if args.use_smooth_l1:
                    recloss = F.smooth_l1_loss(s_recons_l, torch.cat(clean_crops[2:]), reduction='none', beta=args.loss_beta)
                else:
                    recloss = F.l1_loss(s_recons_l, torch.cat(clean_crops[2:]), reduction='none')
                
                if args.drop_only == 1:
                    mask = torch.cat(masks_crops[2:]) == 1
                    error = recloss.detach()
                    focus_weights = torch.ones_like(error)
                    focus_weights[mask] = 1.0 + 0.5 * (error[mask] / error[mask].mean())
                    weighted_loss = recloss * focus_weights
                    r_ = weighted_loss[mask].mean()
                else:
                    r_ = recloss.mean()
                
                loss += r_
                
            if plot_==True and utils.is_main_process():# and args.saveckp_freq and epoch % args.saveckp_freq == 0:
                plot_ = False
                #validating: check the reconstructed images
                print_out = save_recon + '/epoch_' + str(epoch).zfill(5)  + '.jpg' 
                imagesToPrint = torch.cat([clean_crops[0][0: min(15, bz)].cpu(),  corrupted_crops[0][0: min(15, bz)].cpu(),
                                      s_recons_g[0: min(15, bz)].cpu(), masks_crops[0][0: min(15, bz)].cpu()], dim=0)
                torchvision.utils.save_image(imagesToPrint, print_out, nrow=min(15, bz), normalize=True, value_range=(-1, 1))
                        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # model update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(SiT_model, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer) 
                param_norms = utils.clip_gradients(SiT_model, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate_one_epoch(SiT_model, data_loader, epoch, args):
    # Switch to evaluation mode
    SiT_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    with torch.no_grad():
        for it, ((clean_crops, corrupted_crops, masks_crops), _) in enumerate(metric_logger.log_every(data_loader, 50, header)):
            # move images to gpu
            clean_crops = [im.cuda(non_blocking=True) for im in clean_crops]
            corrupted_crops = [im.cuda(non_blocking=True) for im in corrupted_crops]
            masks_crops = [im.cuda(non_blocking=True) for im in masks_crops]
            
            if args.drop_replace > 0:
                corrupted_crops, masks_crops = GMML_replace_list(clean_crops, corrupted_crops, masks_crops, 
                                                              drop_type=args.drop_type,
                                                              max_replace=args.drop_replace, 
                                                              align=args.drop_align)
            
            # Forward pass
            s_recons_g, s_recons_l = SiT_model(corrupted_crops, args.recons_blocks)
            
            # Calculate L1 loss
            if args.use_smooth_l1:
                recloss = F.smooth_l1_loss(s_recons_g, torch.cat(clean_crops[0:2]), reduction='none', beta=args.loss_beta)
            else:
                recloss = F.l1_loss(s_recons_g, torch.cat(clean_crops[0:2]), reduction='none')
                
            loss = recloss[torch.cat(masks_crops[0:2])==1].mean() if (args.drop_only == 1) else recloss.mean()
                
            if len(clean_crops) > 2:
                if args.use_smooth_l1:
                    recloss = F.smooth_l1_loss(s_recons_l, torch.cat(clean_crops[2:]), reduction='none', beta=args.loss_beta)
                else:
                    recloss = F.l1_loss(s_recons_l, torch.cat(clean_crops[2:]), reduction='none')
                
                r_ = recloss[torch.cat(masks_crops[2:])==1].mean() if (args.drop_only == 1) else recloss.mean()
                loss += r_
                
            # Calculate PSNR (a better image quality metric)
            with torch.no_grad():
                psnr = calculate_psnr(s_recons_g, torch.cat(clean_crops[0:2]))
                ssim_value = calculate_ssim(s_recons_g, torch.cat(clean_crops[0:2]))
            
            # Update metrics
            metric_logger.update(loss=loss.item())
            metric_logger.update(psnr=psnr.item())
            metric_logger.update(ssim=ssim_value.item())
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Validation stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def calculate_psnr(x, y, max_val=2.0):  # Assuming images are in range [-1, 1]
    mse = F.mse_loss(x, y, reduction='mean')
    if mse == 0:
        return torch.tensor(100.0).to(x.device)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def calculate_ssim(x, y, window_size=11, sigma=1.5, max_val=2.0):
    # Simple SSIM calculation for PyTorch tensors
    # This is a simplified version of SSIM
    # Use gaussian window
    def gaussian_window(window_size, sigma):
        x = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        return gauss / gauss.sum()
    
    # Create 1D window
    window_1d = gaussian_window(window_size, sigma).to(x.device)
    window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
    window = window_2d.expand(1, 1, window_size, window_size).to(x.device)
    
    # Mean calculations
    mu_x = F.conv2d(x, window, padding=window_size//2, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=window_size//2, groups=y.shape[1])
    
    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y
    
    # Variance calculations
    sigma_x_sq = F.conv2d(x**2, window, padding=window_size//2, groups=x.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(y**2, window, padding=window_size//2, groups=y.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(x*y, window, padding=window_size//2, groups=x.shape[1]) - mu_xy
    
    # Constants
    C1 = (0.01 * max_val)**2
    C2 = (0.03 * max_val)**2
    
    # SSIM calculation
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / denominator
    
    return ssim_map.mean()


def plot_loss_curves(train_losses, val_losses, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curves: {e}")


class FullpiplineSiT(nn.Module):
    def __init__(self, backbone, head_recons):
        super(FullpiplineSiT, self).__init__()

        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head_recons = head_recons
        
        # Add Pixel-Shuffle upsampling for better reconstruction
        self.refine = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, recons_blocks='6-8-10-12'):  
        global_crops = min(2, len(x))  # Default to 2 global crops or less if not enough provided
        
        # Process global crops
        global_crops_tensor = torch.cat(x[0:global_crops])
        backbone_output = self.backbone(global_crops_tensor, recons_blocks=recons_blocks)
        output_recons_global = self.head_recons(backbone_output)
        
        # Additional refinement for better reconstruction
        if hasattr(self, 'refine'):
            # Downsample first to match original resolution
            output_recons_global = F.interpolate(output_recons_global, scale_factor=0.5, mode='bilinear')
            # Apply refinement
            output_recons_global = self.refine(output_recons_global)
            # Ensure values are in range [-1, 1]
            output_recons_global = torch.clamp(output_recons_global, -1, 1)
        
        # Process local crops if available
        output_recons_local = None  
        if len(x) > global_crops:
            local_crops_tensor = torch.cat(x[global_crops:])
            backbone_output_local = self.backbone(local_crops_tensor, recons_blocks=recons_blocks)
            output_recons_local = self.head_recons(backbone_output_local)
            
            # Apply same refinement to local crops
            if hasattr(self, 'refine'):
                output_recons_local = F.interpolate(output_recons_local, scale_factor=0.5, mode='bilinear')
                output_recons_local = self.refine(output_recons_local)
                output_recons_local = torch.clamp(output_recons_local, -1, 1)
        
        return output_recons_global, output_recons_local


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiTv2', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_SiTv2(args)