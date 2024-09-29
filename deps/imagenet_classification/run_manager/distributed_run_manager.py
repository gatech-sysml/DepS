# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024


import json
import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from deps.utils import (
    AverageMeter,
    DistributedMetric,
    MyRandomResizedCrop,
    accuracy,
    cross_entropy_loss_with_soft_target,
    cross_entropy_with_label_smoothing,
    get_net_info,
    init_models,
    list_mean,
    mix_images,
    mix_labels,
    write_log,
)

__all__ = ["DistributedRunManager"]


class DistributedRunManager:
    def __init__(
        self,
        path,
        net,
        run_config,
        hvd_compression,
        backward_steps=1,
        is_root=False,
        init=True,
    ):
        import horovod.torch as hvd

        self.path = path
        self.net = net
        self.run_config = run_config
        self.is_root = is_root

        self.best_acc = 0.0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        cudnn.benchmark = True
        if init and self.is_root:
            init_models(self.net, self.run_config.model_init)
        if self.is_root:
            net_info = get_net_info(self.net, self.run_config.data_provider.data_shape)
            with open("%s/net_info.txt" % self.path, "w") as fout:
                fout.write(json.dumps(net_info, indent=4) + "\n")
                try:
                    fout.write(self.net.module_str + "\n")
                except Exception:
                    fout.write("%s do not support `module_str`" % type(self.net))
                fout.write(
                    "%s\n" % self.run_config.data_provider.train.dataset.transform
                )
                fout.write(
                    "%s\n" % self.run_config.data_provider.test.dataset.transform
                )
                fout.write("%s\n" % self.net)

        self.net.cuda()

        # criterion
        # if isinstance(self.run_config.mixup_alpha, float):
        #     self.train_criterion = cross_entropy_loss_with_soft_target
        # elif self.run_config.label_smoothing > 0:
        #     self.train_criterion = lambda pred, target: cross_entropy_with_label_smoothing(
        #         pred, target, self.run_config.label_smoothing
        #     )
        # else:
        self.train_criterion = nn.CrossEntropyLoss(
            label_smoothing=self.run_config.label_smoothing
        )

        self.test_criterion = nn.CrossEntropyLoss()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split("#")
            net_params = [
                self.net.get_parameters(
                    keys, mode="exclude"
                ),  # parameters with weight decay
                self.net.get_parameters(
                    keys, mode="include"
                ),  # parameters without weight decay
            ]
        else:
            # noinspection PyBroadException
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = []
                for param in self.network.parameters():
                    if param.requires_grad:
                        net_params.append(param)
        self.optimizer = self.run_config.build_optimizer(net_params)

        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer,
            named_parameters=self.net.named_parameters(),
            compression=hvd_compression,
            backward_passes_per_step=backward_steps,
        )

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get("_save_path", None) is None:
            save_path = os.path.join(self.path, "checkpoint")
            os.makedirs(save_path, exist_ok=True)
            self.__dict__["_save_path"] = save_path
        return self.__dict__["_save_path"]

    @property
    def logs_path(self):
        if self.__dict__.get("_logs_path", None) is None:
            logs_path = os.path.join(self.path, "logs")
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__["_logs_path"] = logs_path
        return self.__dict__["_logs_path"]

    @property
    def network(self):
        return self.net

    @network.setter
    def network(self, new_val):
        self.net = new_val

    def write_log(self, log_str, prefix="valid", should_print=True, mode="a"):
        if self.is_root:
            write_log(self.logs_path, log_str, prefix, should_print, mode)

    """ save & load model & save_config & broadcast """

    def save_config(self, extra_run_config=None, extra_net_config=None):
        if self.is_root:
            run_save_path = os.path.join(self.path, "run.config")
            if not os.path.isfile(run_save_path):
                run_config = self.run_config.config
                if extra_run_config is not None:
                    run_config.update(extra_run_config)
                json.dump(run_config, open(run_save_path, "w"), indent=4)
                print("Run configs dump to %s" % run_save_path)

            try:
                net_save_path = os.path.join(self.path, "net.config")
                net_config = self.net.config
                if extra_net_config is not None:
                    net_config.update(extra_net_config)
                json.dump(net_config, open(net_save_path, "w"), indent=4)
                print("Network configs dump to %s" % net_save_path)
            except Exception:
                print("%s do not support net config" % type(self.net))

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if self.is_root:
            assert checkpoint is not None, "checkpoint can't be None. Send a dictionary"
            assert (
                model_name is not None
            ), "model_name can't be None. Send a valid string"

            latest_fname = os.path.join(self.save_path, "latest.txt")
            model_path = os.path.join(self.save_path, model_name)

            with open(latest_fname, "w") as _fout:
                _fout.write(model_path + "\n")

            torch.save(checkpoint, model_path)

            if is_best:
                best_path = os.path.join(self.save_path, "model_best.pth.tar")
                torch.save(checkpoint, best_path)

    def load_model(self, model_fname=None):
        if self.is_root:
            latest_fname = os.path.join(self.save_path, "latest.txt")
            if model_fname is None and os.path.exists(latest_fname):
                with open(latest_fname, "r") as fin:
                    model_fname = fin.readline()
                    if model_fname[-1] == "\n":
                        model_fname = model_fname[:-1]
            # noinspection PyBroadException
            try:
                if model_fname is None or not os.path.exists(model_fname):
                    model_fname = "%s/checkpoint.pth.tar" % self.save_path
                    with open(latest_fname, "w") as fout:
                        fout.write(model_fname + "\n")
                print("=> loading checkpoint '{}'".format(model_fname))
                checkpoint = torch.load(model_fname, map_location="cpu")
            except Exception:
                self.write_log(
                    "fail to load checkpoint from %s" % self.save_path, "valid"
                )
                return

            self.net.load_state_dict(checkpoint["state_dict"])
            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"] + 1
            if "best_acc" in checkpoint:
                self.best_acc = checkpoint["best_acc"]
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            self.write_log("=> loaded checkpoint '{}'".format(model_fname), "valid")

    # noinspection PyArgumentList
    def broadcast(self):
        import horovod.torch as hvd

        self.start_epoch = hvd.broadcast(
            torch.LongTensor(1).fill_(self.start_epoch)[0], 0, name="start_epoch"
        ).item()
        self.best_acc = hvd.broadcast(
            torch.Tensor(1).fill_(self.best_acc)[0], 0, name="best_acc"
        ).item()
        hvd.broadcast_parameters(self.net.state_dict(), 0)
        hvd.broadcast_optimizer_state(self.optimizer, 0)

    """ metric related """

    def get_metric_dict(self):
        return {
            "top1": DistributedMetric("top1"),
            "top5": DistributedMetric("top5"),
        }

    def update_metric(self, metric_dict, output, labels):
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        metric_dict["top1"].update(acc1[0], output.size(0))
        metric_dict["top5"].update(acc5[0], output.size(0))

    def get_metric_vals(self, metric_dict, return_dict=False):
        if return_dict:
            return {key: metric_dict[key].avg.item() for key in metric_dict}
        else:
            return [metric_dict[key].avg.item() for key in metric_dict]

    def get_metric_names(self):
        return "top1", "top5"

    """ train & validate """

    def validate(
        self, mode, epoch=0, run_str="", net=None, data_loader=None, no_logs=False,
    ):

        assert net is not None, "net cannot be none"

        if data_loader is None:
            if mode == "test":
                data_loader = self.run_config.test_loader
            else:
                raise NotImplementedError("mode %s not supported" % mode)

        net.eval()

        losses = DistributedMetric("{}_loss".format(mode))
        metric_dict = self.get_metric_dict()

        with torch.no_grad():
            with tqdm(
                total=len(data_loader),
                desc="Inference {} Epoch #{} {}".format(
                    mode.upper(), epoch + 1, run_str
                ),
                disable=no_logs or not self.is_root,
            ) as t:
                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.cuda(), labels.cuda()

                    output = net(images)
                    loss = self.test_criterion(output, labels)

                    losses.update(loss, images.size(0))
                    self.update_metric(metric_dict, output, labels)

                    t.set_postfix(
                        {
                            "loss": losses.avg.item(),
                            **self.get_metric_vals(metric_dict, return_dict=True),
                            "img_size": images.size(2),
                        }
                    )
                    t.update(1)

        return losses.avg.item(), self.get_metric_vals(metric_dict)

    def validate_all_resolution(self, epoch=0, is_test=False, net=None):
        if net is None:
            net = self.net
        if isinstance(self.run_config.data_provider.image_size, list):
            img_size_list, loss_list, top1_list, top5_list = [], [], [], []
            for img_size in self.run_config.data_provider.image_size:
                img_size_list.append(img_size)
                self.run_config.data_provider.assign_active_img_size(img_size)
                self.reset_running_statistics(net=net)
                loss, (top1, top5) = self.validate(epoch, is_test, net=net)
                loss_list.append(loss)
                top1_list.append(top1)
                top5_list.append(top5)
            return img_size_list, loss_list, top1_list, top5_list
        else:
            loss, (top1, top5) = self.validate(epoch, is_test, net=net)
            return (
                [self.run_config.data_provider.active_img_size],
                [loss],
                [top1],
                [top5],
            )

    def reset_running_statistics(
        self,
        net=None,
        subset_size=2000,
        subset_batch_size=200,
        data_loader=None,
        distributed=False,
    ):
        from deps.imagenet_classification.elastic_nn.utils import set_running_statistics

        if net is None:
            net = self.net
        if data_loader is None:
            data_loader = self.run_config.random_sub_train_loader(
                subset_size, subset_batch_size
            )
        set_running_statistics(net, data_loader, distributed=distributed)
