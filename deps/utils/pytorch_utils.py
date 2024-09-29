# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024

# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "mix_images",
    "mix_labels",
    "label_smooth",
    "cross_entropy_loss_with_soft_target",
    "cross_entropy_with_label_smoothing",
    "clean_num_batch_tracked",
    "rm_bn_from_net",
    "get_net_device",
    "count_parameters",
    "count_net_flops",
    "measure_net_latency",
    "get_net_info",
    "build_optimizer",
    "calculate_and_adjust_learning_rate",
    "ExponentialMovingAverage",
]

""" Mixup """


def mix_images(images, lam):
    flipped_images = torch.flip(images, dims=[0])  # flip along the batch dimension
    return lam * images + (1 - lam) * flipped_images


def mix_labels(target, lam, n_classes, label_smoothing=0.1):
    onehot_target = label_smooth(target, n_classes, label_smoothing)
    flipped_target = torch.flip(onehot_target, dims=[0])
    return lam * onehot_target + (1 - lam) * flipped_target


""" Label smooth """


def label_smooth(target, n_classes: int, label_smoothing=0.1):
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target


def cross_entropy_loss_with_soft_target(pred, soft_target, temperature=1.0):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred / temperature), 1))


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    soft_target = label_smooth(target, pred.size(1), label_smoothing)
    return cross_entropy_loss_with_soft_target(pred, soft_target)


""" BN related """


def clean_num_batch_tracked(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.num_batches_tracked is not None:
                m.num_batches_tracked.zero_()


def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x


""" Network profiling """


def get_net_device(net):
    return net.parameters().__next__().device


def count_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params


def count_net_flops(net, data_shape=(1, 3, 224, 224)):
    from .flops_counter import profile

    if isinstance(net, nn.DataParallel):
        net = net.module

    flop, _ = profile(copy.deepcopy(net), data_shape)
    return flop


def measure_net_latency(
    net, l_type="gpu8", fast=True, input_shape=(3, 224, 224), clean=False
):
    if isinstance(net, nn.DataParallel):
        net = net.module

    # remove bn from graph
    rm_bn_from_net(net)

    # return `ms`
    if "gpu" in l_type:
        l_type, batch_size = l_type[:3], int(l_type[3:])
    else:
        batch_size = 1

    data_shape = [batch_size] + list(input_shape)
    if l_type == "cpu":
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
        if get_net_device(net) != torch.device("cpu"):
            if not clean:
                print("move net to cpu for measuring cpu latency")
            net = copy.deepcopy(net).cpu()
    elif l_type == "gpu":
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
    else:
        raise NotImplementedError
    images = torch.zeros(data_shape, device=get_net_device(net))

    measured_latency = {"warmup": [], "sample": []}
    net.eval()
    with torch.no_grad():
        for i in range(n_warmup):
            inner_start_time = time.time()
            net(images)
            used_time = (time.time() - inner_start_time) * 1e3  # ms
            measured_latency["warmup"].append(used_time)
            if not clean:
                print("Warmup %d: %.3f" % (i, used_time))
        outer_start_time = time.time()
        for i in range(n_sample):
            net(images)
        total_time = (time.time() - outer_start_time) * 1e3  # ms
        measured_latency["sample"].append((total_time, n_sample))

    return total_time / n_sample, measured_latency


def get_net_info(
    net, input_shape=(3, 224, 224), measure_latency=None, print_info=False
):
    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module

    # parameters
    net_info["params"] = count_parameters(net) / 1e6

    # flops
    net_info["flops"] = count_net_flops(net, [1] + list(input_shape)) / 1e6

    # latencies
    latency_types = [] if measure_latency is None else measure_latency.split("#")
    for l_type in latency_types:
        latency, measured_latency = measure_net_latency(
            net, l_type, fast=False, input_shape=input_shape
        )
        net_info["%s latency" % l_type] = {"val": latency, "hist": measured_latency}

    if print_info:
        print(net)
        print("Total training params: %.2fM" % (net_info["params"]))
        print("Total FLOPs: %.2fM" % (net_info["flops"]))
        for l_type in latency_types:
            print(
                "Estimated %s latency: %.3fms"
                % (l_type, net_info["%s latency" % l_type]["val"])
            )

    return net_info


""" optimizer """


def build_optimizer(
    net_params, opt_type, opt_param, init_lr, weight_decay, no_decay_keys
):
    if no_decay_keys is not None:
        assert isinstance(net_params, list) and len(net_params) == 2
        net_params = [
            {"params": net_params[0], "weight_decay": weight_decay},
            {"params": net_params[1], "weight_decay": 0},
        ]
    else:
        net_params = [{"params": net_params, "weight_decay": weight_decay}]

    if opt_type == "sgd":
        opt_param = {} if opt_param is None else opt_param
        momentum, nesterov = (
            opt_param.get("momentum", 0.9),
            opt_param.get("nesterov", True),
        )
        optimizer = torch.optim.SGD(
            net_params, init_lr, momentum=momentum, nesterov=nesterov
        )
        print(f"SGD w/ lr: {init_lr} and wd: {weight_decay} and momentum: {momentum}")
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(net_params, init_lr)
    elif opt_type == "rmsprop":
        momentum = opt_param.get("momentum", 0.9)
        print(
            f"RMSProp w/ lr: {init_lr} and wd: {weight_decay} and momentum: {momentum}"
        )
        optimizer = torch.optim.RMSprop(
            net_params,
            lr=init_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=0.0316227766,
            alpha=0.9,
        )

    else:
        raise NotImplementedError

    return optimizer


def calculate_and_adjust_learning_rate(
    optimizer,
    optimizer_name,
    init_lr,
    gamma,
    epoch,
    n_epochs,
    batch=0,
    nBatch=None,
    lr_schedule_type=None,
    lr_schedule_param=None,
    min_lr=0,
):
    lr = None
    if optimizer_name == "sgd":
        if lr_schedule_type == "cosine":
            t_total = n_epochs * nBatch
            t_cur = epoch * nBatch + batch
            lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
        elif lr_schedule_type == "multistep":
            if lr_schedule_param is None:
                lr = init_lr
            else:
                lr = init_lr * (gamma ** (len(lr_schedule_param) - 1))
                for idx, param in enumerate(lr_schedule_param):
                    if epoch < param:
                        lr = init_lr * (gamma ** idx)
                        break
        elif lr_schedule_type == "multistep_periodic":
            if lr_schedule_param is None:
                raise NotImplementedError
            else:
                assert len(lr_schedule_param) == 1
                if epoch >= 150:
                    init_lr = init_lr * 0.01
                    lr = init_lr * (gamma ** ((epoch - 150) // lr_schedule_param[0]))
                elif epoch >= 100:
                    init_lr = init_lr * 0.1
                    lr = init_lr * (gamma ** ((epoch - 100) // lr_schedule_param[0]))
                else:
                    lr = init_lr * (gamma ** (epoch // lr_schedule_param[0]))
        else:
            raise NotImplementedError

        if lr is not None:
            lr = max(min_lr, lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        return lr
    elif optimizer_name == "rmsprop":
        for param_group in optimizer.param_groups:
            if lr_schedule_type == "multistep_periodic":
                if lr_schedule_param is None:
                    raise NotImplementedError
                else:
                    assert len(lr_schedule_param) == 1
                    param_group["lr"] = gamma * param_group["lr"]
            elif lr_schedule_type == "multistep_periodic_constant":
                if lr_schedule_param is None:
                    raise NotImplementedError
                else:
                    assert len(lr_schedule_param) == 1
                    param_group["lr"] = gamma * param_group["lr"]
                    param_group["lr"] = max(init_lr * 0.05, param_group["lr"])
            else:
                raise ValueError("do not support: %s" % lr_schedule_type)
    else:
        raise NotImplementedError


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        # super().__init__(model, device, ema_avg, use_buffers=True)
        super().__init__(model, device, ema_avg)
