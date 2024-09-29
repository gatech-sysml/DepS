# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024

import os
import random
from os.path import exists
from statistics import mean

import horovod.torch as hvd
import numpy as np
import torch
import wandb
from torch import nn

from deps.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicBatchNorm2d,
    DynamicSeparableConv2d,
)
from deps.imagenet_classification.elastic_nn.training.progressive_shrinking import (
    load_models,
)
from deps.imagenet_classification.run_manager import DistributedImageNetRunConfig
from deps.imagenet_classification.run_manager.distributed_run_manager import (
    DistributedRunManager,
)
from deps.utils import ExponentialMovingAverage, MyRandomResizedCrop, download_url
from network_factory import fetch_supernet, fetch_teacher_net
from train_options import parse_cli_args


def seed_everything(seed=0, harsh=False):
    """
    Seeds all important random functions
    Args:
        seed (int, optional): seed value. Defaults to 0.
        harsh (bool, optional): torch backend deterministic. Defaults to False.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    if harsh:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)


def init_wandb(args) -> None:
    wandb.init(
        name=args.exp_id,
        project=args.wandb_project,
        dir=args.wandb_dir,
        entity=args.wandb_entity,
    )

    return True


## Only useful for MBV3 runs
def initialize_bn_with_bignas(run_manager):
    network = run_manager.network
    # Make last bn (i.e. point_linear) of each block initialized with 0
    # Order within a single block (inverted_bottleneck, depth_conv, and then point_linear)

    ## Skip the first block
    ## Blocks represent all residual blocks
    for block in network.blocks[1:]:
        if block.shortcut is not None:
            if isinstance(block.conv.point_linear.bn, nn.BatchNorm2d):
                block.conv.point_linear.bn.weight.data.fill_(0)
            elif isinstance(block.conv.point_linear.bn, DynamicBatchNorm2d):
                block.conv.point_linear.bn.bn.weight.data.fill_(0)
            else:
                raise ValueError("BN layer type not found")

    return


if __name__ == "__main__":
    args = parse_cli_args()

    os.makedirs(args.path, exist_ok=True)

    # Initialize Horovod
    hvd.init()

    if args.task == "kernel" or args.task == "depth" or args.task == "expand":
        args.teacher_path = download_url(
            "https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7",
            model_dir=".torch/ofa_checkpoints/%d" % hvd.rank(),
        )
    else:
        pass

    # Pin GPU to be used to process local rank (one GPU per process)
    print("========================================")
    print(hvd.local_rank())

    torch.cuda.set_device(hvd.local_rank())

    print(hvd.local_rank())
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    num_gpus = hvd.size()
    args.num_gpus = num_gpus
    print("Number of GPUS: ", num_gpus)

    seed_everything(args.manual_seed)

    # image size
    if isinstance(args.image_size, str):
        args.image_size = [int(img_size) for img_size in args.image_size.split(",")]
    if isinstance(args.image_size, int):
        pass

    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    args.opt_param = {
        "momentum": args.momentum,
        "nesterov": not args.no_nesterov,
    }

    # Linearly rescale the learning rate with GPUs
    args.init_lr = args.base_lr * num_gpus

    run_config = DistributedImageNetRunConfig(
        **args.__dict__, num_replicas=num_gpus, rank=hvd.rank()
    )

    if hvd.rank() == 0:
        if args.wandb:
            print(hvd.rank())
            assert init_wandb(args), "W&B initialization failed"

        print("Check run config on W&B dashboard")

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None

    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # Build network arguments from CLI args
    # Works only for MBV3
    if args.network_family in ["mbv3", "mbv3_32", "proxyless"]:
        args.width_mult_list = [
            float(width_mult) for width_mult in args.width_mult_list.split(",")
        ]
        args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
        args.expand_list = [int(e) for e in args.expand_list.split(",")]
        args.depth_list = [int(d) for d in args.depth_list.split(",")]
    elif args.network_family in ["resnet"]:
        args.width_mult_list = [
            float(width_mult) for width_mult in args.width_mult_list.split(",")
        ]
        args.expand_list = [float(e) for e in args.expand_list.split(",")]
        args.depth_list = [int(d) for d in args.depth_list.split(",")]
        args.ks_list = [int(k) for k in args.ks_list.split(",")]
    else:
        raise NotImplementedError

    args.width_mult_list = (
        args.width_mult_list[0]
        if len(args.width_mult_list) == 1
        else args.width_mult_list
    )

    ## Update all cli args onto wandb
    if args.wandb:
        if hvd.rank() == 0:
            wandb.config.update(args, allow_val_change=True)

    if args.task != "teacher":
        net = fetch_supernet(args, run_config)
        run_manager_init_flag = True
    else:
        net = fetch_teacher_net(args, run_config)
        run_manager_init_flag = False
    if args.task != "teacher" and args.kd_ratio > 0:
        raise NotImplementedError
    else:
        print("Not using a pre-trained teacher model")

    """ Distributed RunManager """
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = num_gpus * args.base_batch_size * args.model_ema_steps / args.n_epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(net, device="cuda", decay=1.0 - alpha)

    run_manager_init_flag = False  # TODO: Remove. Only added to make supernet init same as teacher for sanity check
    distributed_run_manager = DistributedRunManager(
        args.path,
        net,
        run_config,
        compression,
        backward_steps=args.dynamic_batch_size,
        is_root=(hvd.rank() == 0),
        init=run_manager_init_flag,
    )

    distributed_run_manager.model_ema = model_ema

    if args.resume:
        assert args.resume_ckpt is not None, "args.resume_ckpt is None"
        assert exists(
            args.resume_ckpt
        ), f"args.resume_ckpt does not exist: {args.resume_ckpt}"
        checkpoint = torch.load(args.resume_ckpt, map_location="cpu")
        distributed_run_manager.network.load_state_dict(checkpoint["state_dict"])
        distributed_run_manager.optimizer.load_state_dict(checkpoint["optimizer"])
        distributed_run_manager.start_epoch = checkpoint["epoch"] + 1
    elif args.init_ckpt is not None:
        print("Loading initialization weights from: ", args.init_ckpt)
        checkpoint = torch.load(args.init_ckpt, map_location="cpu")
        distributed_run_manager.network.load_state_dict(checkpoint["state_dict"])

    distributed_run_manager.save_config()
    distributed_run_manager.broadcast()

    if args.task not in ["teacher", "deps", "bignas"] and args.kd_ratio > 0:
        print("============    Loading teacher model: START    ============")
        load_models(
            distributed_run_manager, args.teacher_model, model_path=args.teacher_path
        )
        print("============    Loading teacher model: END    ============")

    from deps.imagenet_classification.elastic_nn.training.training import (
        validate,
        train,
    )

    validate_func_dict = {
        "image_size_list": {args.image_size}
        if isinstance(args.image_size, int)
        else sorted({160, 224}),
        "ks_list": sorted(args.ks_list),
        "expand_ratio_list": sorted(args.expand_list),
        "depth_list": sorted(args.depth_list),
    }
    if args.task == "teacher":
        train(
            distributed_run_manager,
            args,
            lambda _net, _run_manager, mode, epoch: validate(
                _net, _run_manager, mode, epoch, **validate_func_dict
            ),
        )
    elif args.task == "deps":
        train(
            distributed_run_manager, args,
        )
    elif args.task == "bignas":
        # BigNAS initialization related changes
        # if args.network_family in ["mbv3", "mbv3_32"]:
        #     initialize_bn_with_bignas(distributed_run_manager)
        train(
            distributed_run_manager, args,
        )
    else:
        raise NotImplementedError
