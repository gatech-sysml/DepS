# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024


import os


def check_args_sanity(args):
    if args.dataset in ["cifar"]:
        assert (
            "_32" in args.network_family
        ), "Network family should support 32x32 for CIFAR datasets"

    if args.resume:
        assert (
            args.resume_ckpt is not None
        ), "If resume=True, resume_ckpt should point to path of checkpoint to restore"
        assert os.path.exists(
            args.resume_ckpt
        ), "If resume=True, file path pointed by resume_ckpt should exist"

    if args.task == "bignas":
        assert (
            args.bignas_lr_decay_step_size is not None
        ), "For BigNAS, bignas_lr_decay_step_size can't be None"
        print(f"bignas_lr_decay_step_size: {args.bignas_lr_decay_step_size}")

    assert args.weight_decay > 0

    if args.lr_schedule_param is not None:
        args.lr_schedule_param = (
            [int(x) for x in args.lr_schedule_param.split(",")]
            if args.lr_schedule_param is not None
            else None
        )

    assert args.wandb_dir is not None, "Please set the WANDB_DIR environment variable"
    assert os.path.exists(args.wandb_dir), "WANDB_DIR does not exist"

    if args.task in ["deps"]:
        if None in [
            args.n_epochs,
            args.base_lr,
            args.dynamic_batch_size,
            args.ks_list,
            args.expand_list,
            args.depth_list,
        ]:
            out = f"n_epochs: {args.n_epochs}\n  base_lr: {args.base_lr}\n  dynamic_batch_size: {args.dynamic_batch_size}\n  ks_list: {args.ks_list}\n  expand_list: {args.expand_list}\n  depth_list: {args.depth_list}\n  "
            print(out)
            raise ValueError("Missing required arguments for deps")

    elif args.task in ["teacher"]:
        if None in [
            args.n_epochs,
            args.base_lr,
            args.dynamic_batch_size,
            args.ks_list,
            args.expand_list,
            args.depth_list,
        ]:
            out = f"n_epochs: {args.n_epochs}\n  base_lr: {args.base_lr}\n  dynamic_batch_size: {args.dynamic_batch_size}\n  ks_list: {args.ks_list}\n  expand_list: {args.expand_list}\n  depth_list: {args.depth_list}\n"
            print(out)
            raise ValueError("Missing required arguments for teacher")
    else:
        if args.task != "bignas":
            assert (
                args.teacher_path is not None
            ), "Teacher path is required for non-(deps/teacher) tasks"
