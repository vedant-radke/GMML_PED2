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

    # Hyper-parameters
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size per GPU.")
    parser.add_argument('--epochs', default=200, type=int, help="Number of epochs of training.")
    
    parser.add_argument('--weight_decay', type=float, default=0.04, help="weight decay")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="Final value of the weight decay.")
    
    parser.add_argument("--lr", default=0.005, type=float, help="Learning rate.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Target LR at the end of optimization.")
    
    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="Whether or not to use half precision for training.")   
    parser.add_argument('--clip_grad', type=float, default=3.0, help="Maximal parameter gradient norm.")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.25, 1.), help="Scale range of global crops")
    parser.add_argument('--local_crops_number', type=int, default=0, help="Number of local crops.")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4), help="Scale range of local crops")
    
    # Misc
    parser.add_argument('--output_dir', default="checkpoints/PED2", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to resume training checkpoint.')
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

    # Increment epochs by 1 to match the range 20, 202, 20
    start_epoch = 0
    total_epochs = 202

    # Preparing Dataset
    if args.data_set == 'PED2':
        transform = DataAugmentationPed2(args)
    else:
        transform = DataAugmentationSiT(args)
        
    dataset, _ = load_dataset.build_dataset(args, True, trnsfrm=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f"==> {args.data_set} training set is loaded.")
    print(f"-------> The dataset consists of {len(dataset)} images.")

    # Create Transformer
    SiT_model = vits.__dict__[args.model](img_size=[args.img_size])
    n_params = sum(p.numel() for p in SiT_model.parameters() if p.requires_grad)
    embed_dim = SiT_model.embed_dim
    
    # Create reconstruction head with the correct img_size
    rec_head = RECHead(embed_dim, patch_size=SiT_model.patch_embed.patch_size, img_size=args.img_size)
    
    SiT_model = FullpiplineSiT(SiT_model, rec_head)
    SiT_model = SiT_model.cuda()
        
    SiT_model = nn.parallel.DistributedDataParallel(SiT_model, device_ids=[args.gpu])
    print(f"==> {args.model} model is created.")
    print(f"-------> The model has {n_params} parameters.")
    
    # Create Optimizer
    params_groups = utils.get_params_groups(SiT_model)
    optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs

    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # Handle checkpoint resuming
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cuda')
        start_epoch = checkpoint['epoch']
        SiT_model.load_state_dict(checkpoint['SiT_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if fp16_scaler is not None and 'fp16_scaler' in checkpoint:
            fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])
        
        print(f"==> Resuming training from epoch {start_epoch}")
    
    # COMMENTED OUT ADAPTIVE LEARNING RATE
    # Instead, use a constant learning rate
    # lr_schedule = utils.cosine_scheduler(args.lr * (args.batch_size * utils.get_world_size()) / 256.,  
    #     args.min_lr, total_epochs, len(data_loader), warmup_epochs=args.warmup_epochs)
    # wd_schedule = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, total_epochs, len(data_loader))

    start_time = time.time()
    print(f"==> Start training from epoch {start_epoch}")
    for epoch in range(start_epoch, total_epochs):
        data_loader.sampler.set_epoch(epoch)

        # Skip epochs not in our desired ranges: 0-20, 20-202
        if (epoch < 20 or (epoch >= 20 and epoch < 202)):
            # Train an epoch
            train_stats = train_one_epoch(SiT_model, data_loader, optimizer, 
                epoch, fp16_scaler, args)

            save_dict = {'SiT_model': SiT_model.state_dict(), 'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1, 'args': args}
            
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
                
            # Always save the latest checkpoint
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
            
            # Additionally save checkpoints at specified frequency
            if args.saveckp_freq and epoch % args.saveckp_freq == 0:
                utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
            if utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

# Modify train_one_epoch to remove adaptive learning rate
def train_one_epoch(SiT_model, data_loader, optimizer, epoch, fp16_scaler, args):
    
    save_recon = os.path.join(args.output_dir, 'reconstruction_samples')
    Path(save_recon).mkdir(parents=True, exist_ok=True)
    bz = args.batch_size
    plot_ = True
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, ((clean_crops, corrupted_crops, masks_crops), _) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        # REMOVED ADAPTIVE LEARNING RATE UPDATE
        # Instead, keep the learning rate constant at the specified value
        
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
            recloss = F.l1_loss(s_recons_g, torch.cat(clean_crops[0:2]), reduction='none')
            loss = recloss[torch.cat(masks_crops[0:2])==1].mean() if (args.drop_only == 1) else recloss.mean()
                
            if len(clean_crops) > 2:
                recloss = F.l1_loss(s_recons_l, torch.cat(clean_crops[2:]), reduction='none') 
                r_ = recloss[torch.cat(masks_crops[2:])==1].mean() if (args.drop_only == 1) else recloss.mean()
                loss += r_
                
            if plot_==True and utils.is_main_process():
                plot_ = False
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
        # REMOVED LEARNING RATE AND WEIGHT DECAY UPDATES
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# Rest of the code remains the same
# (FullpiplineSiT class and main block)
class FullpiplineSiT(nn.Module):
    def __init__(self, backbone, head_recons):
        super(FullpiplineSiT, self).__init__()

        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head_recons = head_recons

    def forward(self, x, recons_blocks='6-8-10-12'):  
        global_crops = min(2, len(x))  # Default to 2 global crops or less if not enough provided
        
        # Process global crops
        global_crops_tensor = torch.cat(x[0:global_crops])
        backbone_output = self.backbone(global_crops_tensor, recons_blocks=recons_blocks)
        output_recons_global = self.head_recons(backbone_output)
        
        # Process local crops if available
        output_recons_local = None  
        if len(x) > global_crops:
            local_crops_tensor = torch.cat(x[global_crops:])
            backbone_output_local = self.backbone(local_crops_tensor, recons_blocks=recons_blocks)
            output_recons_local = self.head_recons(backbone_output_local)
        
        return output_recons_global, output_recons_local

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiTv2', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_SiTv2(args)