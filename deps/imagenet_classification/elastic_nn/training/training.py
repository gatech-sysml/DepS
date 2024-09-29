# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024

import copy
import random
import time

import horovod.torch as hvd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from deps.imagenet_classification.elastic_nn.training.bignas import (
    train_one_epoch_bignas,
)
from deps.imagenet_classification.elastic_nn.training.deps import train_one_epoch_deps
from deps.imagenet_classification.elastic_nn.training.teacher import (
    train_one_epoch_teacher,
)
from deps.imagenet_classification.run_manager import DistributedRunManager
from deps.utils import (
    AverageMeter,
    DistributedMetric,
    MyRandomResizedCrop,
    cross_entropy_loss_with_soft_target,
    list_mean,
    subset_mean,
    val2list,
)

__all__ = [
    "validate",
    "train",
]


def optimizer_step(run_manager, args, epoch):
    old_lr = 0
    for param_group in run_manager.optimizer.param_groups:
        old_lr += param_group["lr"]
    old_lr = old_lr / len(run_manager.optimizer.param_groups)

    run_manager.run_config.calc_and_adjust_lr(
        run_manager.optimizer, epoch, args.lr_gamma,
    )

    new_lr = 0
    for param_group in run_manager.optimizer.param_groups:
        new_lr += param_group["lr"]
    new_lr = new_lr / len(run_manager.optimizer.param_groups)

    return old_lr, new_lr


def validate(
    net,
    run_manager,
    mode,
    epoch=0,
    image_size_list=None,
    ks_list=None,
    expand_ratio_list=None,
    depth_list=None,
    width_mult_list=None,
    additional_setting=None,
):

    if isinstance(net, nn.DataParallel):
        net = net.module
    if mode == "test":
        data_loader = run_manager.run_config.test_loader
    else:
        raise NotImplementedError
    distributed = isinstance(run_manager, DistributedRunManager)
    width_mult_list = [1.0]
    if image_size_list is None:
        image_size_list = val2list(run_manager.run_config.data_provider.image_size, 1)

        image_size_list = [min(image_size_list), max(image_size_list)]
    if ks_list is None:
        ks_list = [min(net.ks_list), max(net.ks_list)]
    if expand_ratio_list is None:
        expand_ratio_list = [min(net.expand_ratio_list), max(net.expand_ratio_list)]
    if depth_list is None:
        depth_list = [min(net.depth_list), max(net.depth_list)]
    if width_mult_list is None:
        if "width_mult_list" in net.__dict__:
            width_mult_list = list(range(len(net.width_mult_list)))
        else:
            width_mult_list = [1.0]

    subnet_settings = []
    for d in depth_list:
        for e in expand_ratio_list:
            for k in ks_list:
                for w in width_mult_list:
                    for img_size in image_size_list:
                        subnet_settings.append(
                            [
                                {
                                    "image_size": img_size,
                                    "d": d,
                                    "e": e,
                                    "ks": k,
                                    "w": w,
                                },
                                "R%s-D%s-E%s-K%s-W%s" % (img_size, d, e, k, w),
                            ]
                        )

    subnet_settings = (
        subnet_settings[0:2] + subnet_settings[-2:]
    )  # Only run 4 subnets (top 2 and bottom 2)
    losses_of_subnets, top1_of_subnets, top5_of_subnets = [], [], []

    valid_log = ""
    valid_log_dict = {}
    for setting, name in subnet_settings:
        run_str = name
        run_manager.write_log(
            "-" * 30 + " Validate %s " % name + "-" * 30, "train", should_print=False
        )
        if run_manager.run_config.task not in ["teacher"]:
            print("Set active subnet and calibrate batch norm parameters...")
            net.set_active_subnet(**setting)
            run_manager.reset_running_statistics(net)

        losses = DistributedMetric(f"{mode}_loss") if distributed else AverageMeter()
        metric_dict = run_manager.get_metric_dict()

        with tqdm(
            total=len(data_loader),
            desc="Inference {} Epoch #{} {}".format(mode.upper(), epoch, run_str),
            disable=False or not run_manager.is_root,
        ) as t:
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.cuda(), labels.cuda()

                output = net(images)
                loss = run_manager.test_criterion(output, labels)

                losses.update(loss, images.size(0))
                run_manager.update_metric(metric_dict, output, labels)

                t.set_postfix(
                    {
                        "loss": losses.avg.item(),
                        **run_manager.get_metric_vals(metric_dict, return_dict=True),
                        "img_size": images.size(2),
                    }
                )
                t.update(1)

        loss = losses.avg.item()
        top1, top5 = run_manager.get_metric_vals(metric_dict)

        valid_log_dict[f"{name}_{mode}/top1"] = metric_dict["top1"].avg.item()
        valid_log_dict[f"{name}_{mode}/top5"] = metric_dict["top5"].avg.item()
        valid_log_dict[f"{name}_{mode}/loss"] = loss

        valid_log_dict[f"{name}_{mode}/top1"] = top1
        valid_log_dict[f"{name}_{mode}/top5"] = top5
        valid_log_dict[f"{name}_{mode}/loss"] = loss

        losses_of_subnets.append(loss)
        top1_of_subnets.append(top1)
        top5_of_subnets.append(top5)
        valid_log += "%s (%.3f), " % (name, top1)

    return (
        subnet_settings,
        losses_of_subnets,
        top1_of_subnets,
        top5_of_subnets,
        valid_log,
        valid_log_dict,
    )


def train_one_epoch_ps(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.network
    distributed = isinstance(run_manager, DistributedRunManager)

    # switch to train mode
    dynamic_net.train()
    if distributed:
        run_manager.run_config.train_loader.sampler.set_epoch(epoch)
    MyRandomResizedCrop.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()
    losses = DistributedMetric("train_loss") if distributed else AverageMeter()
    metric_dict = run_manager.get_metric_dict()
    subnet_metric_dict = {}

    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch),
        disable=distributed and not run_manager.is_root,
    ) as t:

        end = time.time()
        for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
            MyRandomResizedCrop.BATCH = i
            data_time.update(time.time() - end)

            if args.opt_type == "sgd":
                if epoch < warmup_epochs:
                    new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                        run_manager.optimizer,
                        warmup_epochs * nBatch,
                        nBatch,
                        epoch,
                        i,
                        warmup_lr,
                    )
                else:
                    print("SGDDD")
                    new_lr = run_manager.run_config.adjust_learning_rate(
                        run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                    )
            else:
                # Only call update lr every 3, 5, or generally, k epochs
                assert args.opt_type in ["adam", "rmsprop"]
                assert "multistep" in args.lr_schedule_type
                if epoch % args.lr_schedule_param[0] == 0:
                    run_manager.run_config.calc_and_adjust_lr(
                        run_manager.optimizer,
                        args.init_lr,
                        args.lr_gamma,
                        args.lr_schedule_type,
                        args.lr_schedule_param,
                    )

                new_lr = 0
                for param_group in run_manager.optimizer.param_groups:
                    new_lr += param_group["lr"]
                new_lr = new_lr / len(run_manager.optimizer.param_groups)
                # new_lr = run_manager.optimizer.param_groups[0]["lr"]

            images, labels = images.cuda(), labels.cuda()
            target = labels

            if args.kd_ratio > 0:
                args.teacher_model.train()
                with torch.no_grad():
                    soft_logits = args.teacher_model(images).detach()
                    soft_label = F.softmax(soft_logits, dim=1)

            # clean gradients
            dynamic_net.zero_grad()

            loss_of_subnets = []

            # compute output
            subnet_strs = ""
            for _ in range(args.dynamic_batch_size):
                # set random seed before sampling
                subnet_seed = int("%d%.3d%.3d" % (epoch * nBatch + i, _, 0))
                random.seed(subnet_seed)
                subnet_settings = dynamic_net.sample_active_subnet()

                subnet_str = (
                    "%d: " % _
                    + ",".join(
                        [
                            "%s_%s"
                            % (
                                key,
                                "%.1f" % subset_mean(val, 0)
                                if isinstance(val, list)
                                else val,
                            )
                            for key, val in subnet_settings.items()
                        ]
                    )
                    + " || "
                )

                subnet_strs += subnet_str
                subnet_name = ""
                for key, val in subnet_settings.items():
                    subnet_name += f"{key}={subset_mean(val, 0)}_"

                output = dynamic_net(images)

                if args.kd_ratio == 0:
                    loss = run_manager.train_criterion(output, labels)
                    loss_type = "ce"
                else:
                    if args.kd_type == "ce":
                        kd_loss = cross_entropy_loss_with_soft_target(
                            output, soft_label
                        )
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = args.kd_ratio * kd_loss + run_manager.train_criterion(
                        output, labels
                    )
                    loss_type = "%.1fkd-%s & ce" % (args.kd_ratio, args.kd_type)

                # measure accuracy and record loss
                loss_of_subnets.append(loss)
                run_manager.update_metric(metric_dict, output, target)
                loss.backward()

            run_manager.optimizer.step()
            losses.update(list_mean(loss_of_subnets), images.size(0))

            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "R": images.size(2),
                    "lr": new_lr,
                    "loss_type": loss_type,
                    "seed": str(subnet_seed),
                    "str": subnet_str,
                    # "data_time": data_time.avg,
                }
            )

            t.update(1)
            end = time.time()

    if args.wandb:
        if hvd.rank() == 0:
            subnet_metric_dict[f"{subnet_name}_train"] = {}
            subnet_metric_dict[f"{subnet_name}_train"]["loss"] = losses.avg.item()
            subnet_metric_dict[f"{subnet_name}_train"]["top1"] = metric_dict[
                "top1"
            ].avg.item()
            subnet_metric_dict[f"{subnet_name}_train"]["top5"] = metric_dict[
                "top5"
            ].avg.item()

            wandb.log(subnet_metric_dict, step=epoch)
            wandb.log({"lr": new_lr}, step=epoch)

    return losses.avg.item(), run_manager.get_metric_vals(metric_dict)


def train(run_manager, args, validate_func=None):
    distributed = isinstance(run_manager, DistributedRunManager)

    if validate_func is None:
        validate_func = validate

    for epoch in range(
        run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs
    ):

        if args.task in ["deps", "bignas"]:
            if args.lr_schedule_type != "cosine":
                old_lr, new_lr = optimizer_step(run_manager, args, epoch)

            if args.task == "deps":
                train_loss, (train_top1, train_top5) = train_one_epoch_deps(
                    run_manager, args, epoch, args.warmup_epochs, args.warmup_lr
                )
            elif args.task == "bignas":
                train_loss, (train_top1, train_top5) = train_one_epoch_bignas(
                    run_manager, args, epoch, args.warmup_epochs, args.warmup_lr
                )

            if args.wandb:
                if hvd.rank() == 0:
                    if args.lr_schedule_type != "cosine":
                        wandb.log({"lr": new_lr}, step=epoch)
                        wandb.log({"lr_ratio": old_lr / new_lr}, step=epoch)
        elif args.task == "teacher":
            if args.lr_schedule_type != "cosine":
                old_lr, new_lr = optimizer_step(run_manager, args, epoch)
            train_loss, (train_top1, train_top5) = train_one_epoch_teacher(
                run_manager, args, epoch, args.warmup_epochs, args.warmup_lr
            )

        else:
            train_loss, (train_top1, train_top5) = train_one_epoch_ps(
                run_manager, args, epoch, args.warmup_epochs, args.warmup_lr
            )

        if epoch % args.validation_frequency == 0:
            # net = copy.deepcopy(run_manager.net)
            (
                test_subnets,
                test_losses,
                test_accs,
                test_accs5,
                _test_log,
                _test_log_dict,
            ) = validate_func(run_manager.net, run_manager, epoch=epoch, mode="test")

            test_loss = list_mean(test_losses)
            test_acc = list_mean(test_accs)
            test_acc5 = list_mean(test_accs5)

            if args.wandb:
                if hvd.rank() == 0:
                    print("Test Syncing W&B")
                    wandb.log(_test_log_dict, step=epoch)

            is_best = test_acc > run_manager.best_acc
            run_manager.best_acc = max(run_manager.best_acc, test_acc)

            if not distributed or run_manager.is_root:
                test_log = "Test [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})".format(
                    epoch - args.warmup_epochs,
                    run_manager.run_config.n_epochs,
                    test_loss,
                    test_acc,
                    run_manager.best_acc,
                )
                run_manager.write_log(test_log, "Test", should_print=False)

                checkpoint_dict = {
                    "epoch": epoch,
                    "best_acc": run_manager.best_acc,
                    "optimizer": run_manager.optimizer.state_dict(),
                    "state_dict": run_manager.network.state_dict(),
                    "state_dict_ema": run_manager.model_ema.state_dict()
                    if args.model_ema
                    else None,
                }

                run_manager.save_model(
                    checkpoint_dict, is_best=is_best, model_name="checkpoint.pth.tar"
                )

                if epoch % args.checkpoint_frequency == 0:
                    run_manager.save_model(
                        checkpoint_dict,
                        is_best=False,
                        model_name=f"checkpoint_{epoch}.pth.tar",
                    )
