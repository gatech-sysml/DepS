# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024

from os.path import join


def set_default_args(CKPT_ROOT, args):
    if args.task == "teacher":
        args.path = join(CKPT_ROOT, f"{args.exp_id}/teacher")
        args.name = f"teacher_{args.exp_id}"
        args.dynamic_batch_size = 1
        args.kd_ratio = 0.0  # no explicit teacher model for individual training
    elif args.task == "deps":
        args.path = join(CKPT_ROOT, f"{args.exp_id}/deps")
        args.name = f"deps_{args.exp_id}"
        args.kd_ratio = 0.0  # No explicit teacher model for deps
    elif args.task == "bignas":
        args.path = join(CKPT_ROOT, f"{args.exp_id}/bignas")
        args.name = f"bignas_{args.exp_id}"
        args.kd_ratio = 0.0  # No explicit teacher model for bignas
    elif args.task == "kernel":
        args.path = join(CKPT_ROOT, f"{args.exp_id}/normal2kernel")
        args.name = f"normal2kernel_{args.exp_id}"
        args.kd_ratio = 1.0
        args.dynamic_batch_size = 1
        args.n_epochs = 120
        args.base_lr = 3e-2
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3, 5, 7"
        args.expand_list = "6"
        args.depth_list = "4"
    elif args.task == "depth":
        args.path = join(
            CKPT_ROOT, f"{args.exp_id}/kernel2kernel_depth/phase%d" % args.phase
        )
        args.name = f"kernel2kernel_depth_phase={args.phase}_{args.exp_id}"
        args.dynamic_batch_size = 2
        args.kd_ratio = 1.0
        if args.phase == 1:
            args.n_epochs = 25
            args.base_lr = 2.5e-3
            args.warmup_epochs = 0
            args.warmup_lr = -1
            # TODO: If we choose to do kernel based elastic training, then this ks_list will be ks_list="3, 5, 7" but since we don't do that, let it be maximum i.e. "7"
            # args.ks_list = "3, 5, 7"
            args.ks_list = "7"
            args.expand_list = "6"
            args.depth_list = "3, 4"
        else:
            args.n_epochs = 120
            args.base_lr = 7.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            # TODO: If we choose to do kernel based elastic training, then this ks_list will be ks_list="3, 5, 7" but since we don't do that, let it be maximum i.e. "7"
            # args.ks_list = "3, 5, 7"
            args.ks_list = "7"
            args.expand_list = "6"
            args.depth_list = "2, 3, 4"

    elif args.task == "expand":
        args.path = join(
            CKPT_ROOT,
            f"{args.exp_id}/kernel_depth2kernel_depth_width/phase%d" % args.phase,
        )
        args.name = f"kernel_depth2kernel_depth_width_phase={args.phase}_{args.exp_id}"
        args.dynamic_batch_size = 4
        args.kd_ratio = 1.0
        if args.phase == 1:
            args.n_epochs = 25
            args.base_lr = 2.5e-3
            args.warmup_epochs = 0
            args.warmup_lr = -1
            # TODO: If we choose to do kernel based elastic training, then this ks_list will be ks_list="3, 5, 7" but since we don't do that, let it be maximum i.e. "7"
            # args.ks_list = "3, 5, 7"
            args.ks_list = "7"
            args.expand_list = "4,6"
            args.depth_list = "2, 3, 4"
        else:
            args.n_epochs = 120
            args.base_lr = 7.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            # TODO: If we choose to do kernel based elastic training, then this ks_list will be ks_list="3, 5, 7" but since we don't do that, let it be maximum i.e. "7"
            # args.ks_list = "3, 5, 7"
            args.ks_list = "7"
            args.expand_list = "3, 4, 6"
            args.depth_list = "2, 3, 4"
    else:
        raise NotImplementedError
