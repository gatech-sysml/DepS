# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024

import random
import time
from statistics import mean

import horovod.torch as hvd
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from deps.imagenet_classification.run_manager import DistributedRunManager
from deps.utils import (
    AverageMeter,
    DistributedMetric,
    MyRandomResizedCrop,
    list_mean,
    subset_mean,
)

__all__ = [
    "train_one_epoch_teacher",
]


def train_one_epoch_teacher(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.net
    distributed = isinstance(run_manager, DistributedRunManager)
    dynamic_net.train()

    model_ema = run_manager.model_ema
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

    outputs = []

    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch),
        disable=distributed and not run_manager.is_root,
    ) as t:

        end = time.time()
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

            # zero gradients
            dynamic_net.zero_grad()

            loss_of_subnets = []
            subnet_settings = {
                "ks": args.ks_list,
                "d": args.depth_list,
                "e": args.expand_list,
            }

            subnet_name = ""
            for key, val in subnet_settings.items():
                subnet_name += f"{key}={subset_mean(val, 0)}_"

            # compute output
            output = dynamic_net(images)
            outputs.append(output.detach().cpu().numpy())
            loss = run_manager.train_criterion(output, labels)
            loss_type = "ce"
            loss_of_subnets.append(loss)
            run_manager.update_metric(metric_dict, output, target)
            loss.backward()
            run_manager.optimizer.synchronize()  # Explicit synchronize

            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    run_manager.net.parameters(), args.clip_grad_norm
                )

            with run_manager.optimizer.skip_synchronize():
                run_manager.optimizer.step()

            if model_ema is not None and args.model_ema:
                if i % args.model_ema_steps == 0:
                    model_ema.update_parameters(run_manager.net)
                    if epoch < args.warmup_epochs:
                        # Reset ema buffer to keep copying weights during warmup period
                        model_ema.n_averaged.fill_(0)

            losses.update(list_mean(loss_of_subnets), images.size(0))

            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "R": images.size(2),
                    "loss_type": loss_type,
                }
            )

            t.update(1)
            end = time.time()

    if args.wandb:
        if hvd.rank() == 0:
            print("Train Syncing W&B")
            subnet_metric_dict[f"average_train"] = {}

            subnet_metric_dict[f"average_train"]["loss"] = losses.avg.item()
            subnet_metric_dict[f"average_train"]["top1"] = metric_dict[
                "top1"
            ].avg.item()
            subnet_metric_dict[f"average_train"]["top5"] = metric_dict[
                "top5"
            ].avg.item()

            wandb.log(subnet_metric_dict, step=epoch)
            wandb.log({"lr": new_lr}, step=epoch)

    return losses.avg.item(), run_manager.get_metric_vals(metric_dict)
