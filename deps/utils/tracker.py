# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024


from statistics import mean

import numpy as np
import torch

from deps.utils import AverageMeter, DistributedMetric

__all__ = ["Tracker"]


class Tracker:
    def __init__(self, distributed):
        self.subnet_keys = ["deps", "minnet", "medium1net", "medium2net"]
        self.gradient_keys = [
            "first_conv",
            "blocks_2_pl_conv",
            "blocks_2_pl_bn",
            "blocks_10_pl_conv",
            "blocks_10_pl_bn",
            "blocks_18_pl_conv",
            "blocks_18_pl_bn",
        ]

        self.distributed = distributed

        self.gradients = {}
        self.gradient_aggregates = {}
        self.gradient_lists = {}

        # Difference between before and after is stored in self.gradient_lists
        self.gradient_before = {}
        self.gradient_after = {}

        self.losses = {}
        self.mean_losses = {}

        # Initialize all variables
        for subnet in self.subnet_keys:
            self.losses[subnet] = (
                DistributedMetric(f"train_{subnet}_loss")
                if self.distributed
                else AverageMeter()
            )
            self.mean_losses[subnet] = 0

            self.gradients[subnet] = {}
            self.gradient_lists[subnet] = {}
            self.gradient_before[subnet] = {}
            self.gradient_after[subnet] = {}
            self.gradient_aggregates[subnet] = {}

            for gradient_key in self.gradient_keys:
                self.gradients[subnet][gradient_key] = DistributedMetric(
                    f"{gradient_key}_{subnet}_gradient"
                )
                self.gradient_lists[subnet][gradient_key] = []

                # Stores a single value at a time. NOT TO BE USED FOR ACCUMULATION
                self.gradient_before[subnet][gradient_key] = 0
                self.gradient_after[subnet][gradient_key] = 0
                self.gradient_aggregates[subnet][gradient_key] = 0

    def track_before_gradient(self, subnet_name, run_manager):
        first_conv = run_manager.net.first_conv.conv.weight.grad.cpu().detach()

        blocks_2_pl_conv = (
            run_manager.net.blocks[2]
            .conv.point_linear.conv.conv.weight.grad.cpu()
            .detach()
        )
        blocks_10_pl_conv = (
            run_manager.net.blocks[10]
            .conv.point_linear.conv.conv.weight.grad.cpu()
            .detach()
        )
        blocks_18_pl_conv = (
            run_manager.net.blocks[18]
            .conv.point_linear.conv.conv.weight.grad.cpu()
            .detach()
        )

        blocks_2_pl_bn = (
            run_manager.net.blocks[2].conv.point_linear.bn.bn.weight.grad.cpu().detach()
        )
        blocks_10_pl_bn = (
            run_manager.net.blocks[10]
            .conv.point_linear.bn.bn.weight.grad.cpu()
            .detach()
        )
        blocks_18_pl_bn = (
            run_manager.net.blocks[18]
            .conv.point_linear.bn.bn.weight.grad.cpu()
            .detach()
        )

        for gradient_key in self.gradient_keys:
            self.gradient_before[subnet_name][gradient_key] = eval(gradient_key)

    def track_after_gradient(self, subnet_name, run_manager):
        first_conv = run_manager.net.first_conv.conv.weight.grad.cpu().detach()

        blocks_2_pl_conv = (
            run_manager.net.blocks[2]
            .conv.point_linear.conv.conv.weight.grad.cpu()
            .detach()
        )
        blocks_10_pl_conv = (
            run_manager.net.blocks[10]
            .conv.point_linear.conv.conv.weight.grad.cpu()
            .detach()
        )
        blocks_18_pl_conv = (
            run_manager.net.blocks[18]
            .conv.point_linear.conv.conv.weight.grad.cpu()
            .detach()
        )

        blocks_2_pl_bn = (
            run_manager.net.blocks[2].conv.point_linear.bn.bn.weight.grad.cpu().detach()
        )
        blocks_10_pl_bn = (
            run_manager.net.blocks[10]
            .conv.point_linear.bn.bn.weight.grad.cpu()
            .detach()
        )
        blocks_18_pl_bn = (
            run_manager.net.blocks[18]
            .conv.point_linear.bn.bn.weight.grad.cpu()
            .detach()
        )

        for gradient_key in self.gradient_keys:
            self.gradient_after[subnet_name][gradient_key] = eval(gradient_key)

    def update_gradient_list(self):
        for subnet in self.subnet_keys:
            for gradient_key in self.gradient_keys:
                self.gradient_lists[subnet][gradient_key].append(
                    self.gradient_after[subnet][gradient_key]
                    - self.gradient_before[subnet][gradient_key]
                )

    def track_loss(self, loss, subnet_name, size):
        self.losses[subnet_name].update(loss, size)

    def aggregate_losses(self):
        for subnet_key in self.subnet_keys:
            self.mean_losses[subnet_key] = self.losses[subnet_key].avg.item()

    def update_gradients(self):
        for subnet_key in self.subnet_keys:
            for gradient_key in self.gradient_keys:
                # Stack list into one large tensor
                self.gradient_lists[subnet_key][gradient_key] = torch.stack(
                    self.gradient_lists[subnet_key][gradient_key]
                )

                # Take L1 norm across all dimensions except dim 0 (leave out the batch size dimension i.e. dim=0)
                if "_bn" in gradient_key:
                    self.gradient_lists[subnet_key][gradient_key] = torch.norm(
                        self.gradient_lists[subnet_key][gradient_key], p=1, dim=1
                    )
                else:
                    self.gradient_lists[subnet_key][gradient_key] = (
                        torch.norm(
                            self.gradient_lists[subnet_key][gradient_key],
                            p=1,
                            dim=(1, 2, 3, 4),
                        )
                        / self.gradient_lists[subnet_key][gradient_key]
                        .shape[1:]
                        .numel()
                    )

                # Take mean across the batch_size dimension dim=0
                self.gradient_lists[subnet_key][gradient_key] = torch.mean(
                    self.gradient_lists[subnet_key][gradient_key]
                )

                self.gradients[subnet_key][gradient_key].update(
                    self.gradient_lists[subnet_key][gradient_key]
                )

    def aggregate_gradients(self):
        for subnet_key in self.subnet_keys:
            for gradient_key in self.gradient_keys:
                self.gradient_aggregates[subnet_key][gradient_key] = self.gradients[
                    subnet_key
                ][gradient_key].avg.item()

    def __del__(self):
        self.gradients = None
        self.gradient_lists = None
        # Stores a single value at a time. NOT TO BE USED FOR ACCUMULATION
        self.gradient_before = None
        self.gradient_after = None
        self.gradient_aggregates = None

        self.losses = None
        self.loss_list = None
