# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets

import resnet

from sage_transform import SageTransform
from sage_loader import SageFolder

# Vision Transformers
import sys
sys.path.append("../pytorch-image-models-sage")
from timm.models import vision_transformer as vit_models


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
                        help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
                        of input square patches - default 16 (for 16x16 patches). Using smaller
                        values leads to better performance but requires more memory. Applies only
                        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
                        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")


    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    torch.backends.cudnn.benchmark = True
    # init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

#     if args.rank == 0:
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    # transforms = aug.TrainTransform()
    # dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
    # note tha the data_dir should contain the rgb and thermal directories
    transform = SageTransform()
    dataset = SageFolder(args.data_dir / "train/pairs", transform=transform)

    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    sampler = torch.utils.data.RandomSampler(dataset)

    # assert args.batch_size % args.world_size == 0
    # per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    # ! need to have two instances of the same architecture to deal with
    # IR and RGB images separately
    model_rgb = VICReg(args, num_channels_lr=(3, 1)).cuda(gpu)
    model_IR = VICReg(args, num_channels_lr=(1,3)).cuda(gpu)
#     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer_rgb = LARS(
        model_rgb.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )
    optimizer_IR = LARS(
        model_IR.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    if (args.exp_dir / "model.pth").is_file():
        # if args.rank == 0:
        print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model_rgb.load_state_dict(ckpt["model_rgb"])
        model_IR.load_state_dict(ckpt["model_IR"])
        optimizer_rgb.load_state_dict(ckpt["optimizer_rgb"])
        optimizer_IR.load_state_dict(ckpt["optimizer_IR"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler_rgb = torch.cuda.amp.GradScaler()
    scaler_IR = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        # sampler.set_epoch(epoch)
        for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.type(torch.cuda.HalfTensor)
            y = y.cuda(gpu, non_blocking=True)
            
            lr_rgb = adjust_learning_rate(args, optimizer_rgb, loader, step)
            
            # rgb model
            optimizer_rgb.zero_grad()
            with torch.cuda.amp.autocast():
                loss_rgb = model_rgb.forward(x, y)
            scaler_rgb.scale(loss_rgb).backward()
            scaler_rgb.step(optimizer_rgb)
            scaler_rgb.update()
            
            # IR model
            lr_IR = adjust_learning_rate(args, optimizer_IR, loader, step)
            optimizer_IR.zero_grad()
            with torch.cuda.amp.autocast():
                loss_IR = model_IR.forward(y, x)
            scaler_IR.scale(loss_IR).backward()
            scaler_IR.step(optimizer_IR)
            scaler_IR.update()

            current_time = time.time()
            # if args.rank == 0 and current_time - last_logging > args.log_freq_time:
            stats = dict(
                epoch=epoch,
                step=step,
                loss_rgb=loss_rgb.item(),
                loss_IR=loss_IR.item(),
                time=int(current_time - start_time),
                lr_rgb=lr_rgb,
                lr_IR=lr_IR
            )
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)
            last_logging = current_time
            if step % 500 == 0:
                state = dict(
                    epoch=epoch,
                    model_rgb=model_rgb.state_dict(),
                    model_IR=model_IR.state_dict(),
                    optimizer_rgb=optimizer_rgb.state_dict(),
                    optimizer_IR=optimizer_IR.state_dict(),
                )
                torch.save(state, args.exp_dir / "model.pth")
        #if args.rank == 0:
        state = dict(
            epoch=epoch + 1,
            model_rgb=model_rgb.state_dict(),
            model_IR=model_IR.state_dict(),
            optimizer_rgb=optimizer_rgb.state_dict(),
            optimizer_IR=optimizer_IR.state_dict(),
        )
        torch.save(state, args.exp_dir / "model.pth")
    #if args.rank == 0:
    torch.save(model_rgb.module.backbone.state_dict(), args.exp_dir / "resnet50_rgb.pth")
    torch.save(model_IR.module.backbone.state_dict(), args.exp_dir / "resnet50_IR.pth")


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class VICReg(nn.Module):
    def __init__(self, args, num_channels_lr):
        # num_channels = (left branch channels, right branch channels)
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        # remember there is always a branch with 1xN input and another branch
        # 3xN input.
#         print(num_channels_lr)
        self.backbone_left, self.embedding_left = resnet.__dict__[args.arch](
            zero_init_residual=True,
            num_channels=num_channels_lr[0]
        )
        self.projector_left = Projector(args, self.embedding_left)
        
        self.backbone_right, self.embedding_right = resnet.__dict__[args.arch](
            zero_init_residual=True,
            num_channels=num_channels_lr[1]
        )
        self.projector_right = Projector(args, self.embedding_right)

    def forward(self, x, y):
        x_in = self.backbone_left(x)
        x = self.projector_left(x_in)
        # how does this line fix the type conversion thing?!
#         y = y.type(torch.cuda.HalfTensor)
        y_in = self.backbone_right(y)
        y = self.projector_right(y_in)

        # note this line might need to modified to adjust for different dimensions
        # of x and y
        repr_loss = F.mse_loss(x, y)

#         x = torch.cat(x, dim=0)
#         y = torch.cat(y, dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)