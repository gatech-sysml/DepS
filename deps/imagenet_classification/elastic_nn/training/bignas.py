# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024

import copy
import random
import time
from collections import defaultdict

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
    "train_one_epoch_bignas",
]


def train_one_epoch_bignas(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
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
    num_training_steps = 0
    new_lr = 0

    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch),
        disable=distributed and not run_manager.is_root,
    ) as t:
        end = time.time()
        for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
            num_training_steps += 1
            MyRandomResizedCrop.BATCH = i
            data_time.update(time.time() - end)

            if args.lr_schedule_type == "cosine":
                if args.opt_type == "sgd":
                    new_lr = run_manager.run_config.calc_and_adjust_lr(
                        run_manager.optimizer,
                        epoch=epoch - warmup_epochs,
                        gamma=args.lr_gamma,
                        batch=i,
                        nBatch=nBatch,
                    )

            # if args.bignas_recipe == "original":
            #     assert args.bignas_lr_decay_step_size != 0
            #     assert args.bignas_lr_decay_step_size != 1

            #     if num_training_steps % args.bignas_lr_decay_step_size == 0:
            #         run_manager.run_config.calc_and_adjust_lr(run_manager.optimizer, args.init_lr, args.lr_gamma, args.lr_schedule_type, args.lr_schedule_param)
            # else:
            #     pass
            # TEACHER BIGNAS RECIPE
            # Nothing, it's handled in training.py

            images, labels = images.cuda(), labels.cuda()
            target = labels

            # clean gradients
            dynamic_net.zero_grad()
            loss_of_subnets = []

            # compute output
            subnet_strs = ""
            # args.dynamic_batch_size represents the number of subnets to sample in each epoch
            max_network_soft_labels = None
            for subnet_no in range(args.dynamic_batch_size):
                if subnet_no == 0:  # Sample Maximum Network
                    dynamic_net.set_max_net()
                    subnet_settings = {
                        "ks": [max(dynamic_net.ks_list)],
                        "e": [max(dynamic_net.expand_ratio_list)],
                        "d": [max(dynamic_net.depth_list)],
                    }
                    output = run_manager.net(images, max=True)

                    loss = run_manager.train_criterion(output, labels)

                    if args.inplace_distillation:
                        max_network_logits = output.clone().detach()
                        max_network_soft_labels = F.softmax(max_network_logits, dim=1)
                elif subnet_no == 1:  # Sample Minimum Network
                    dynamic_net.set_min_net()
                    subnet_settings = {
                        "ks": [min(dynamic_net.ks_list)],
                        "e": [min(dynamic_net.expand_ratio_list)],
                        "d": [min(dynamic_net.depth_list)],
                    }
                    output = run_manager.net(images, max=False)

                    # Inplace distillation
                    if args.inplace_distillation:
                        assert max_network_soft_labels is not None
                        loss = cross_entropy_loss_with_soft_target(
                            output, max_network_soft_labels
                        )
                    else:
                        raise NotImplementedError("Inplace distillation is required")
                elif subnet_no > 1:
                    # set random seed before sampling. This is vvv important to make sure that same subnets are sampled within an epoch
                    subnet_seed = int("%d%.3d%.3d" % (epoch * nBatch + i, subnet_no, 0))
                    random.seed(subnet_seed)
                    subnet_settings = dynamic_net.sample_active_subnet()
                    output = run_manager.net(images, max=False)

                    if args.inplace_distillation:
                        assert max_network_soft_labels is not None
                        loss = cross_entropy_loss_with_soft_target(
                            output, max_network_soft_labels
                        )
                    else:
                        raise NotImplementedError("Inplace distillation is required")
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

                # measure accuracy and record loss
                loss_type = "ce"
                loss_of_subnets.append(loss)
                run_manager.update_metric(metric_dict, output, target)
                loss.backward()

            # Explicit synchronize is not required because we're not doing anything funky with gradients before .step()
            # .step() does a synchronize by default
            run_manager.optimizer.step()
            losses.update(list_mean(loss_of_subnets), images.size(0))
            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "R": images.size(2),
                    "loss_type": loss_type,
                    "str": subnet_str,
                }
            )
            t.update(1)
            end = time.time()

    if args.wandb:
        if hvd.rank() == 0:
            subnet_metric_dict[f"average_train"] = {}
            subnet_metric_dict[f"average_train"]["loss"] = losses.avg.item()
            subnet_metric_dict[f"average_train"]["top1"] = metric_dict[
                "top1"
            ].avg.item()
            subnet_metric_dict[f"average_train"]["top5"] = metric_dict[
                "top5"
            ].avg.item()

            wandb.log(subnet_metric_dict, step=epoch)

    return losses.avg.item(), run_manager.get_metric_vals(metric_dict)
