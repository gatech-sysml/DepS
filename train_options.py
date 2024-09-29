# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024

import argparse
import os

from deps.utils.sanity_utils import check_args_sanity
from modify_args import set_default_args

CKPT_ROOT = "/home/akhare39/aditya/wsn/wsn_public/ckpt_dump"


def parse_cli_args() -> argparse.Namespace:
    WANDB_DIR = os.environ["WANDB_DIR"]

    parser = argparse.ArgumentParser()

    # Weight Shared Algorithm choices
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["teacher", "bignas", "deps", "kernel", "depth", "expand"],
    )
    parser.add_argument("--validation_frequency", type=int, default=1)
    parser.add_argument("--checkpoint_frequency", type=int, default=5)
    parser.add_argument("--print_frequency", type=int, default=10)

    # Experiment resume and initialization
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to the checkpoint to restore",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Path Dataset")
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help="Checkpoint to initialize the network from",
    )
    ####################################

    # Progressive shrinking
    parser.add_argument("--teacher_path", type=str, default=None)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--ps_resume", action="store_true")
    parser.add_argument(
        "--kd_ratio",
        type=float,
        default=0,
        choices=[0.0, 1.0],
        help="KD Ratio 1 means it expects teacher model",
    )
    ####################################

    ######################## SUPERNETWORK CONFIGURATION ########################
    parser.add_argument(
        "--network_family",
        type=str,
        required=True,
        choices=["mbv3", "mbv3_32", "resnet", "proxyless"],
    )
    parser.add_argument("--dynamic_batch_size", type=int, required=True)
    parser.add_argument(
        "--sampling", type=str, default="random", choices=["random", "compofa"]
    )
    parser.add_argument("--ks_list", type=str, default=None)
    parser.add_argument("--expand_list", type=str, default=None)
    parser.add_argument("--depth_list", type=str, default=None)
    parser.add_argument("--width_mult_list", type=str, default="1.0")

    parser.add_argument("--inplace_distillation", action="store_true")
    parser.add_argument("--bignas_recipe", default="original")
    parser.add_argument("--bignas_lr_decay_step_size", type=int, default=None)

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["imagenet", "cifar", "imagenet-100"],
    )
    parser.add_argument(
        "--cifar_mode", type=str, default=None, choices=["cifar100", "cifar10", None]
    )
    #################################################################

    # Weights and Biases parameters
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_entity", default="gatech-sysml", type=str)
    parser.add_argument("--wandb_project", default="wsn", type=str)
    parser.add_argument("--wandb_dir", default=WANDB_DIR, type=str)
    #################################################################

    ######################## TRAINING RECIPE ########################
    # Optimizer parameters
    parser.add_argument(
        "--opt_type", type=str, required=True, choices=["adam", "sgd", "rmsprop"]
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--no_nesterov", action="store_true")
    parser.add_argument("--base_lr", type=float, default=None)
    parser.add_argument("--model_init", type=str, default="he_fout")
    ##################################

    # Learning rate schedule parameters
    parser.add_argument(
        "--lr_schedule_type",
        type=str,
        required=True,
        choices=[
            "bignas_exp",
            "cosine",
            "multistep",
            "multistep_periodic",
            "multistep_periodic_constant",
            None,
        ],
    )
    parser.add_argument("--lr_schedule_param", type=str, default=None)
    parser.add_argument("--lr_gamma", type=float, required=True)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--warmup_lr", type=float, default=0.1)
    ####################################

    # Regularization parameters
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    # Manual weight decay
    parser.add_argument("--manual_weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    ####################################

    # Other training parameters
    parser.add_argument("--base_batch_size", type=int, default=256)
    parser.add_argument("--bn_momentum", type=float, default=0.99)
    ## Proposed method parameters

    parser.add_argument("--teacher_warmup", type=int, default=0)
    parser.add_argument("--smallnet_warmup", type=int, default=5)
    parser.add_argument(
        "--gradient_aggregation",
        type=str,
        default="sum",
        choices=["avg", "sum", "avg_sqrt"],
    )
    parser.add_argument("--reorganize_weights", action="store_true")
    parser.add_argument("--min_lr", type=float, default=2.5e-3)
    ####################################

    # Performance bells and whistles
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--min_multiplier", type=float, default=1.0)
    parser.add_argument("--max_multiplier", type=float, default=1.0)
    parser.add_argument("--other_multiplier", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Other training parameters
    parser.add_argument("--bn_eps", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument(
        "--dy_conv_scaling_mode", type=int, default=-1, choices=[-1, 1]
    )  # This is 1 for OFA only. -1 for DepS and BigNAS
    parser.add_argument("--kd_type", type=str, default="ce", choices=["ce"])
    ################################

    ## Gradient Clipping
    parser.add_argument(
        "--clip_grad_norm",
        default=None,
        type=float,
        help="the maximum gradient norm (default None)",
    )
    #####################

    ## Exponential moving average
    parser.add_argument(
        "--model_ema",
        action="store_true",
        help="enable tracking Exponential Moving Average of model parameters",
    )
    parser.add_argument(
        "--model_ema_steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model_ema_decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    ####################################

    parser.add_argument(
        "--no_decay_keys", type=str, default="bn#bias", choices=[None, "bn#bias"]
    )
    parser.add_argument("--fp16_allreduce", action="store_true")

    # Basic data augmentations
    parser.add_argument("--resize_scale", type=float, default=0.08)
    parser.add_argument(
        "--distort_color", type=str, required=True, choices=["tf", "torch", "none"]
    )

    ## Additional ImageNet augmentations
    parser.add_argument("--random_erase_prob", type=float, required=True)
    parser.add_argument(
        "--auto_augment", type=str, default=None, choices=["imagenet", "ta_wide", None]
    )

    ## Additional ImageNet ResNet augmentations
    parser.add_argument("--rand_augment", type=str, default=None)
    parser.add_argument("--mixup_alpha", type=float, default=None)
    parser.add_argument("--cutmix_alpha", type=float, default=None)
    parser.add_argument("--cutmix_minmax", type=float, default=None)

    parser.add_argument(
        "--scaling", action="store_true"
    )  # Not used at all. Get rid of it
    ####################################

    # Parallelization parameters
    parser.add_argument("--n_worker", type=int, default=8)
    ####################################

    # Randomness parameters
    parser.add_argument("--manual_seed", type=int, default=8)
    ####################################

    args = parser.parse_args()

    check_args_sanity(args)
    set_default_args(CKPT_ROOT, args)

    if args.sampling == "compofa":
        args.compound = True
    else:
        args.compound = False

    args.continuous_size = True
    args.not_sync_distributed_image_size = False
    args.base_stage_width = "proxyless"

    args.independent_distributed_sampling = False

    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr

    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size

    # 1000 for ImageWoof, None for ImageNet
    if args.dataset == "imagewoof":
        args.valid_size = 1000
    elif args.dataset in ["imagenet", "imagenet-100", "cifar"]:
        args.valid_size = None
    else:
        raise NotImplementedError

    if args.dataset in ["imagewoof", "imagenet", "imagenet-100"]:
        args.image_size = 224
    elif args.dataset in ["cifar"]:
        args.image_size = 32
    else:
        raise NotImplementedError

    return args


### TODOs
## max_bignas_init was used in the main run! Bring it back
