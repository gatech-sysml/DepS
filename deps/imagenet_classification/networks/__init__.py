# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024

from .mobilenet_v3 import *
from .mobilenet_v3_32 import MobileNetV3Large32
from .proxyless_nets import *
from .resnets import *


def get_net_by_name(name):
    if name == ProxylessNASNets.__name__:
        return ProxylessNASNets
    elif name == MobileNetV3.__name__:
        return MobileNetV3
    elif name == ResNets.__name__:
        return ResNets
    else:
        raise ValueError("unrecognized type of network: %s" % name)
