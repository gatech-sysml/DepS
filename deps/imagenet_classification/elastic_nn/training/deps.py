# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024

import copy
import math
import random
import time
from collections import defaultdict
from functools import reduce
from statistics import mean

import horovod.torch as hvd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from deps.imagenet_classification.run_manager import DistributedRunManager
from deps.utils import (
    AverageMeter,
    DistributedMetric,
    MyRandomResizedCrop,
    cross_entropy_loss_with_soft_target,
    list_mean,
    subset_mean,
)

__all__ = [
    "train_one_epoch_deps",
]

SUBNET_NAMES = ["deps", "minnet", "medium1net", "medium2net"]


def smallnet_warmup_adjust_learning_rate(
    T_total, nBatch, epoch, batch=0, warmup_lr=0, desired_lr=None
):
    if desired_lr is None:
        raise ValueError()

    T_cur = epoch * nBatch + batch + 1  # from IN 150/270 branch

    new_lr = T_cur / T_total * (desired_lr - warmup_lr) + warmup_lr
    return new_lr


def adjust_beta_value(args, epoch, multiplier, lr, teacher_warmup, nBatch, batch):
    from deps.imagenet_classification.elastic_nn.training.lrs import (
        mbv3_imagenet100_lrs_180,
        mbv3_imagenet1k_lrs_180,
        mbv3_cifar_lrs_200,
        mbv3_imagenet1k_lrs_270,
        proxyless_imagenet1k_lrs_300,
    )

    lr_dict = None
    if args.network_family in ["mbv3", "mbv3_32"]:
        if args.n_epochs == 180:
            if args.dataset == "imagenet-100":
                lr_dict = {}
                for key, value in mbv3_imagenet100_lrs_180.items():
                    lr_dict[key] = value * args.num_gpus
            elif args.dataset == "imagenet":
                lr_dict = mbv3_imagenet1k_lrs_180

        elif args.n_epochs == 200:
            assert args.dataset == "cifar"
            lr_dict = mbv3_cifar_lrs_200
        elif args.n_epochs == 270:
            assert args.dataset == "imagenet"
            lr_dict = mbv3_imagenet1k_lrs_270
        else:
            raise NotImplementedError()

    elif args.network_family in ["proxyless"]:
        if args.n_epochs == 300:
            lr_dict = proxyless_imagenet1k_lrs_300
        else:
            raise NotImplementedError

    assert lr_dict is not None
    smallnet_warmup = args.smallnet_warmup
    desired_lr = multiplier * lr_dict[teacher_warmup]
    final_lr = multiplier * lr

    if epoch < teacher_warmup - smallnet_warmup:
        new_beta = 0
    elif epoch >= teacher_warmup - smallnet_warmup and epoch < teacher_warmup:
        T_total = smallnet_warmup * nBatch
        warmup_epoch = epoch - (teacher_warmup - smallnet_warmup)
        warmup_lr = smallnet_warmup_adjust_learning_rate(
            T_total, nBatch, warmup_epoch, batch, desired_lr=desired_lr
        )
        new_beta = warmup_lr / lr
    else:
        new_beta = final_lr / lr

    return new_beta


def train_one_epoch_deps(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.network
    distributed = isinstance(run_manager, DistributedRunManager)

    # switch to train mode
    dynamic_net.train()
    teacher_warmup = run_manager.run_config.teacher_warmup
    teacher_mode = False
    # reorganize_flag = False
    print(type(teacher_warmup), teacher_warmup)
    print(type(args.smallnet_warmup), args.smallnet_warmup)

    if epoch < teacher_warmup - args.smallnet_warmup:
        print("Teacher MODE")
        teacher_mode = True
    else:
        print("Supernet MODE")

    if not teacher_mode:
        if not run_manager.run_config.reorganize_flag:
            if args.reorganize_weights:
                # Only reorganize if explicitly mentioned in CLI
                expand_stage_list = dynamic_net.expand_ratio_list.copy()
                expand_stage_list.sort(reverse=True)
                n_stages = len(expand_stage_list) - 1
                current_stage = n_stages - 1
                dynamic_net.re_organize_middle_weights(expand_ratio_stage=current_stage)
                run_manager.run_config.reorganize_flag = True

    gradient_aggregation = args.gradient_aggregation
    if distributed:
        run_manager.run_config.train_loader.sampler.set_epoch(epoch)

    MyRandomResizedCrop.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()
    losses = DistributedMetric("train_loss") if distributed else AverageMeter()
    metric_dict = run_manager.get_metric_dict()
    subnet_metric_dict = {}

    num_training_steps = nBatch * epoch
    new_lr = 0

    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch),
        disable=distributed and not run_manager.is_root,
    ) as t:
        end = time.time()
        if teacher_mode:
            beta = 0
            new_beta = 0
        else:
            pass
        for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
            MyRandomResizedCrop.BATCH = i
            data_time.update(time.time() - end)
            num_training_steps += 1

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
                    new_lr = run_manager.run_config.calc_and_adjust_lr(
                        run_manager.optimizer,
                        epoch=epoch - warmup_epochs,
                        gamma=args.lr_gamma,
                        batch=i,
                        nBatch=nBatch,
                    )

            images, labels = images.cuda(), labels.cuda()
            target = labels

            # clean gradients
            dynamic_net.zero_grad()
            loss_of_subnets = []

            subnet_strs = ""

            gradient_store = defaultdict(int)

            subnet_seed = 0
            subnet_mask = 0
            subnet_mask_sums = {}

            if teacher_mode:  # Train only teacher in teacher mode!
                dynamic_net.set_max_net()
                subnet_settings = {
                    "ks": [max(dynamic_net.ks_list)],
                    "e": [max(dynamic_net.expand_ratio_list)],
                    "d": [max(dynamic_net.depth_list)],
                }
                subnet_str = (
                    "%d: " % 0
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
                loss = run_manager.train_criterion(output, labels)
                loss_type = "ce"
                loss_of_subnets.append(loss)
                run_manager.update_metric(metric_dict, output, target)

                loss.backward()
                losses.update(loss, images.size(0))
            else:
                max_network_soft_labels = None
                for subnet_no in range(args.dynamic_batch_size):
                    if subnet_no == 0:  # Sample deps
                        dynamic_net.set_max_net()
                        subnet_settings = {
                            "ks": [max(dynamic_net.ks_list)],
                            "e": [max(dynamic_net.expand_ratio_list)],
                            "d": [max(dynamic_net.depth_list)],
                        }

                        weight = args.max_multiplier
                        subnet_mask = dynamic_net.get_active_subnet_mask()
                        output = dynamic_net(images, max=True)

                        max_network_logits = output.clone().detach()

                        ## Generate soft labels with respect to temperature parameter
                        max_network_soft_labels = F.softmax(
                            (max_network_logits / args.temperature), dim=1
                        )

                        loss = run_manager.train_criterion(output, labels)

                    elif subnet_no == 1:  # Sample minnet

                        new_beta = adjust_beta_value(
                            args,
                            epoch,
                            multiplier=args.min_multiplier,
                            lr=new_lr,
                            teacher_warmup=teacher_warmup,
                            nBatch=nBatch,
                            batch=i,
                        )

                        dynamic_net.set_min_net()
                        subnet_settings = {
                            "ks": [min(dynamic_net.ks_list)],
                            "e": [min(dynamic_net.expand_ratio_list)],
                            "d": [min(dynamic_net.depth_list)],
                        }
                        weight = new_beta
                        subnet_mask = dynamic_net.get_active_subnet_mask()
                        output = dynamic_net(images, max=False)

                        if args.inplace_distillation:
                            # Distillation loss
                            assert max_network_soft_labels is not None
                            loss = (
                                args.temperature ** 2
                            ) * cross_entropy_loss_with_soft_target(
                                output, max_network_soft_labels, args.temperature
                            )
                        else:
                            # Normal loss
                            loss = run_manager.train_criterion(output, labels)

                    elif subnet_no > 1:
                        new_beta = adjust_beta_value(
                            args,
                            epoch,
                            multiplier=args.other_multiplier,
                            lr=new_lr,
                            teacher_warmup=teacher_warmup,
                            nBatch=nBatch,
                            batch=i,
                        )

                        subnet_seed = int(
                            "%d%.3d%.3d" % (epoch * nBatch + i, subnet_no, 0)
                        )
                        random.seed(subnet_seed)
                        if args.sampling == "random":
                            subnet_settings = dynamic_net.sample_active_subnet()
                            subnet_mask = dynamic_net.get_active_subnet_mask()
                        elif args.sampling == "compofa":
                            subnet_settings = dynamic_net.sample_compound_subnet()
                            subnet_mask = dynamic_net.get_active_subnet_mask()
                        else:
                            raise NotImplementedError

                        weight = new_beta
                        output = dynamic_net(images, max=False)

                        if args.inplace_distillation:
                            # Distillation loss
                            assert max_network_soft_labels is not None
                            loss = (
                                args.temperature ** 2
                            ) * cross_entropy_loss_with_soft_target(
                                output, max_network_soft_labels, args.temperature
                            )

                        else:
                            # Normal loss
                            loss = run_manager.train_criterion(output, labels)

                    else:
                        raise ValueError("Invalid subnet_no")

                    subnet_str = (
                        "%d: " % subnet_no
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

                    loss_type = "ce"
                    loss_of_subnets.append(loss)
                    run_manager.update_metric(metric_dict, output, target)

                    loss.backward()
                    run_manager.optimizer.synchronize()  # Explicit synchronize

                    # new_beta takes care of the adjustment
                    for name, param in dynamic_net.named_parameters():
                        if param.requires_grad:
                            # https://discuss.pytorch.org/t/please-help-how-can-copy-the-gradient-from-net-a-to-net-b/41226/6
                            # Clone is not necessary
                            mask = subnet_mask[name].cuda()
                            if name not in subnet_mask_sums.keys():
                                subnet_mask_sums[name] = weight * mask
                            else:
                                subnet_mask_sums[name] += weight * mask

                            if name not in gradient_store.keys():
                                gradient_store[name] = weight * param.grad.data * mask
                            else:
                                gradient_store[name] += weight * param.grad.data * mask
                        else:
                            import ipdb

                            ipdb.set_trace()

                    dynamic_net.zero_grad()
                    losses.update(loss, images.size(0))

                for name, param in dynamic_net.named_parameters():
                    # Remove averaging. Just do summation.
                    # # print(gradient_store[name].shape, subnet_mask_sums[name].shape)
                    # assert gradient_store[name].size() == subnet_mask_sums[name].size()
                    if gradient_aggregation == "avg":
                        gradient_store[name] = (
                            gradient_store[name] / subnet_mask_sums[name]
                        )
                    elif gradient_aggregation == "avg_sqrt":
                        gradient_store[name] = gradient_store[name] / torch.sqrt(
                            subnet_mask_sums[name]
                        )
                    elif gradient_aggregation == "sum":
                        # Sum already happened in line `gradient_store[name] += weight * param.grad.data * mask`
                        pass
                    else:
                        raise NotImplementedError(
                            f"Gradient Aggregation: {gradient_aggregation} not supported"
                        )

                    # tensor copy
                    param.grad.data.copy_(gradient_store[name])

            run_manager.optimizer.step()
            gradient_store = {}
            subnet_mask_sums = {}

            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "R": images.size(2),
                    "loss_type": "ce",
                    "seed": str(subnet_seed),
                    "str": subnet_str,
                }
            )
            t.update(1)
            end = time.time()

    if args.wandb:
        if hvd.rank() == 0:
            subnet_metric_dict["average_train"] = {}
            subnet_metric_dict["average_train"]["loss"] = losses.avg.item()
            subnet_metric_dict["average_train"]["top1"] = metric_dict["top1"].avg.item()
            subnet_metric_dict["average_train"]["top5"] = metric_dict["top5"].avg.item()

            wandb.log(subnet_metric_dict, step=epoch)

            wandb.log({"beta": new_beta}, step=epoch)

            beta_dict = {}
            beta_dict["beta/max"] = args.max_multiplier
            beta_dict["beta/min"] = new_beta
            beta_dict["beta/beta"] = new_beta

            wandb.log({"lr": new_lr}, step=epoch)

            lr_dict = {}
            lr_dict["effective_lr/min"] = new_beta * new_lr
            lr_dict["effective_lr/max"] = 1 * new_lr

            wandb.log(lr_dict, step=epoch)
            wandb.log(beta_dict, step=epoch)

    return losses.avg.item(), run_manager.get_metric_vals(metric_dict)
