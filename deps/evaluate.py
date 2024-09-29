import pickle
from os.path import exists, join

import horovod.torch as hvd
import torch

from deps.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from deps.imagenet_classification.networks import MobileNetV3Large
from deps.imagenet_classification.run_manager import DistributedImageNetRunConfig
from deps.imagenet_classification.run_manager.distributed_run_manager import (
    DistributedRunManager,
)

hvd.init()
torch.cuda.set_device(hvd.local_rank())
num_gpus = hvd.size()

prefix = "/nethome/sannavajjala6/projects/wsn/.torch/ofa_checkpoints/0/"
ofa_teacher_path = join(prefix, "ofa_D4_E6_K7")
ofa_weight_shared_path = join(prefix, "ofa_mbv3_d234_e346_k357_w1.0")

assert exists(ofa_teacher_path)
assert exists(ofa_weight_shared_path)


ofa_weight_shared = OFAMobileNetV3(
    n_classes=1000,
    bn_param=(0.99, 1e-5),
    dropout_rate=0,
    width_mult=1.0,
    ks_list=[3, 5, 7],
    expand_ratio_list=[3, 4, 6],
    depth_list=[2, 3, 4],
)

ofa_weight_shared.load_state_dict(
    torch.load(ofa_weight_shared_path, map_location="cpu")["state_dict"]
)


run_config = DistributedImageNetRunConfig(
    "ps",
    dataset="imagenet",
    test_batch_size=256,
    valid_size=None,
    image_size=224,
    num_replicas=1,
    rank=0,
    cifar_mode=None,
)

run_config.n_epochs = 10
run_config.num_networks_per_batch = 4

run_manager = DistributedRunManager(
    "./",
    ofa_weight_shared,
    run_config,
    hvd_compression=hvd.Compression.none,
    backward_steps=1,
    is_root=(hvd.rank() == 0),
    init=False,
)


def validate(
    run_manager, img_size, d, e, k, w=1.0, epoch=0, bn_calib=False,
):

    net = run_manager.net
    net.eval()

    name = ("R%s-D%s-E%s-K%s-W%s" % (img_size, d, e, k, w),)
    setting = {
        "image_size": img_size,
        "d": d,
        "e": e,
        "ks": k,
        "w": w,
    }

    print("Set active subnet")
    net.set_active_subnet(**setting)
    if bn_calib:
        print("calibrate batch norm parameters...")
        run_manager.reset_running_statistics(net)

    mode = "test"
    _, (top1, top5) = run_manager.validate(
        epoch=epoch, mode=mode, run_str=name, net=net
    )

    return top1, top5


ofa_performance_bn_calib = {}
ofa_performance_nobn_calib = {}

depth = [2, 3, 4]
expand = [3, 4, 6]
kernel = [3, 5, 7]

for d in depth:
    for e in expand:
        for k in kernel:
            name = f"D{d}-E{e}-K{k}"
            calib_top1, _ = validate(run_manager, 224, d, e, k, bn_calib=True)
            ofa_performance_bn_calib[name] = calib_top1

            # nocalib_top1, _ = validate(run_manager, 224, d, e, k, bn_calib=False)
            # ofa_performance_nobn_calib[name] = nocalib_top1
            print(f"Model: {name} Calib: {calib_top1}")

pickle.dump(ofa_performance_bn_calib, open("ofa_performance_bn_calib.pkl", "wb"))
# pickle.dump(ofa_performance_nobn_calib, open('ofa_performance_nobn_calib.pkl', 'wb'))
