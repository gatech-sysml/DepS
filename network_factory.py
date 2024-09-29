# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024

from deps.imagenet_classification.elastic_nn.networks import (
    OFAMobileNetV3,
    OFAMobileNetV332,
    OFAProxylessNASNets,
    OFAResNets,
)
from deps.imagenet_classification.networks import (
    MobileNetV2,
    MobileNetV3Large,
    MobileNetV3Large32,
    ResNet50,
)


def fetch_teacher_net(args, run_config):
    assert len(args.ks_list) == 1
    assert len(args.expand_list) == 1
    assert len(args.depth_list) == 1
    print(
        f"ks: {args.ks_list[0]} expand: {args.expand_list[0]} depth: {args.depth_list[0]} width: {args.width_mult_list}"
    )
    if args.network_family == "mbv3":
        print("Using MobileNetV3Large for teacher")
        net = MobileNetV3Large(
            n_classes=run_config.data_provider.n_classes,
            bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=args.dropout,
            width_mult=args.width_mult_list,
            ks=args.ks_list[0],
            expand_ratio=args.expand_list[0],
            depth_param=args.depth_list[0],
        )
    elif args.network_family == "mbv3_32":
        print("Using MobileNetV3Large32 for teacher")
        net = MobileNetV3Large32(
            n_classes=run_config.data_provider.n_classes,
            bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=args.dropout,
            width_mult=args.width_mult_list,
            ks=args.ks_list[0],
            expand_ratio=args.expand_list[0],
            depth_param=args.depth_list[0],
        )
    elif args.network_family == "proxyless":
        print("Using OFAProxylessNASNets for teacher")
        net = OFAProxylessNASNets(
            n_classes=run_config.data_provider.n_classes,
            bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=args.dropout,
            base_stage_width=args.base_stage_width,
            width_mult=args.width_mult_list,
            ks_list=args.ks_list,
            expand_ratio_list=args.expand_list,
            depth_list=args.depth_list,
            model_init=args.model_init,
            scaling=args.scaling,
            compound=args.compound,
        )
    elif args.network_family == "resnet":

        # from torchvision.models import resnet
        # Overwrite the URL of the previous weights
        # resnet.model_urls["resnet50"] = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"

        # Initialize the model using the legacy API
        # net = resnet.resnet50(pretrained=False)
        net = OFAResNets(
            n_classes=run_config.data_provider.n_classes,
            bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=args.dropout,
            depth_list=args.depth_list,
            expand_ratio_list=args.expand_list,
            width_mult_list=args.width_mult_list,
        )

    return net


def fetch_supernet(args, run_config):
    if args.network_family == "mbv3":
        print("Using OFAMobileNetV3")
        net = OFAMobileNetV3(
            n_classes=run_config.data_provider.n_classes,
            bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=args.dropout,
            base_stage_width=args.base_stage_width,
            width_mult=args.width_mult_list,
            ks_list=args.ks_list,
            expand_ratio_list=args.expand_list,
            depth_list=args.depth_list,
            model_init=args.model_init,
            scaling=args.scaling,
            compound=args.compound,
        )

    elif args.network_family == "mbv3_32":
        print("Using OFAMobileNetV332")
        net = OFAMobileNetV332(
            n_classes=run_config.data_provider.n_classes,
            bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=args.dropout,
            base_stage_width=args.base_stage_width,
            width_mult=args.width_mult_list,
            ks_list=args.ks_list,
            expand_ratio_list=args.expand_list,
            depth_list=args.depth_list,
            model_init=args.model_init,
            scaling=args.scaling,
            compound=args.compound,
        )

    elif args.network_family == "proxyless":
        print("Using OFAProxylessNASNets")
        net = OFAProxylessNASNets(
            n_classes=run_config.data_provider.n_classes,
            bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=args.dropout,
            base_stage_width=args.base_stage_width,
            width_mult=args.width_mult_list,
            ks_list=args.ks_list,
            expand_ratio_list=args.expand_list,
            depth_list=args.depth_list,
            model_init=args.model_init,
            scaling=args.scaling,
            compound=args.compound,
        )
    elif args.network_family == "resnet":
        net = OFAResNets(
            n_classes=run_config.data_provider.n_classes,
            bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=args.dropout,
            depth_list=args.depth_list,
            expand_ratio_list=args.expand_list,
            width_mult_list=args.width_mult_list,
        )

    else:
        raise NotImplementedError(
            f"Network family: {args.network_family} not supported"
        )

    if args.task == "deps":
        net.initialize_mask()

    return net
