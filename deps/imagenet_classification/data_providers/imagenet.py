# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024


import math
import os
import warnings

import numpy as np
import torch.utils.data
import torchvision.datasets as datasets
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

from deps.imagenet_classification.data_providers.custom_transforms import (
    RandomCutmix,
    RandomMixup,
)
from deps.utils.my_dataloader import MyDistributedSampler, MyRandomResizedCrop

from .base_provider import DataProvider

__all__ = ["ImagenetDataProvider"]


class ImagenetDataProvider(DataProvider):
    # PATH to IMAGENET FOLDER
    DEFAULT_PATH = "./data/imagenet"
    IS_IMAGENET_100 = False

    def __init__(
        self,
        save_path=None,
        train_batch_size=128,
        test_batch_size=64,
        valid_size=None,
        n_worker=32,
        resize_scale=0.08,
        distort_color=None,
        image_size=224,
        num_replicas=None,
        rank=None,
        random_erase_prob=0.2,
        auto_augment=None,
        mixup_alpha=0,
        cutmix_alpha=0,
        cutmix_minmax=0,
        rand_augment=None,
        **kwargs,
    ):

        assert mixup_alpha is not None
        assert cutmix_alpha is not None

        warnings.filterwarnings("ignore")
        self._save_path = save_path
        self.kwargs = kwargs
        self.image_size = image_size  # int or list of int
        self.distort_color = "None" if distort_color is None else distort_color
        self.resize_scale = resize_scale
        self.random_erase_prob = random_erase_prob
        self.auto_augment = auto_augment
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        self.rand_augment = rand_augment

        mixup_transforms = []
        if self.mixup_alpha > 0.0:
            mixup_transforms.append(
                RandomMixup(self.n_classes, p=1.0, alpha=self.mixup_alpha)
            )

        if self.cutmix_alpha > 0.0:
            mixup_transforms.append(
                RandomCutmix(self.n_classes, p=1.0, alpha=self.cutmix_alpha)
            )

        if mixup_transforms:
            mixupcutmix = transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            from deps.utils.my_dataloader import MyDataLoader

            assert isinstance(self.image_size, list)
            self.image_size.sort()  # e.g., 160 -> 224
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(
                    img_size
                )
            self.active_img_size = max(self.image_size)  # active resolution for test
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader

        train_dataset = self.train_dataset(self.build_train_transform())

        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset) * valid_size)

            valid_dataset = self.train_dataset(valid_transforms)
            train_indices, valid_indices = self.random_sample_valid_set(
                len(train_dataset), valid_size
            )
            print("Num replicas: ", num_replicas)
            if num_replicas is not None:
                train_sampler = MyDistributedSampler(
                    train_dataset, num_replicas, rank, True, np.array(train_indices)
                )
                valid_sampler = MyDistributedSampler(
                    valid_dataset, num_replicas, rank, True, np.array(valid_indices)
                )
            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    train_indices
                )
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    valid_indices
                )

            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                sampler=train_sampler,
                num_workers=n_worker,
                pin_memory=False,
                collate_fn=collate_fn,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                sampler=valid_sampler,
                num_workers=n_worker,
                pin_memory=False,
            )
        else:
            if num_replicas is not None:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas, rank
                )
                self.train = train_loader_class(
                    train_dataset,
                    batch_size=train_batch_size,
                    sampler=train_sampler,
                    num_workers=n_worker,
                    pin_memory=False,
                )
            else:
                self.train = train_loader_class(
                    train_dataset,
                    batch_size=train_batch_size,
                    shuffle=True,
                    num_workers=n_worker,
                    pin_memory=False,
                )
            self.valid = None

        test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas, rank
            )
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                sampler=test_sampler,
                num_workers=n_worker,
                pin_memory=False,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=False,
            )

        if self.IS_IMAGENET_100:
            print(
                f"****** Using {ImagenetDataProvider.name()} with classes {self.n_classes}"
            )

        if self.valid is None:
            print(
                "===> Validation dataset is None. Setting Val and Test as the same dataset"
            )
            print("===> Temporary fix only")
            self.valid = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=False,
            )
            # raise ValueError

    @staticmethod
    def name():
        if ImagenetDataProvider.IS_IMAGENET_100:
            return "imagenet-100"
        return "imagenet"

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        if ImagenetDataProvider.IS_IMAGENET_100:
            return 100
        return 1000

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser("~/data/ImageNet/")

        return self._save_path

    @property
    def data_url(self):
        raise ValueError("unable to download %s" % self.name())

    def train_dataset(self, _transforms):
        return datasets.ImageFolder(self.train_path, _transforms)

    def test_dataset(self, _transforms):
        return datasets.ImageFolder(self.valid_path, _transforms)

    @property
    def train_path(self):
        return os.path.join(self.save_path, "train")

    @property
    def valid_path(self):
        return os.path.join(self.save_path, "val")

    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print(
                "Color jitter: %s, resize_scale: %s, img_size: %s"
                % (self.distort_color, self.resize_scale, image_size)
            )

        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop
            print(
                "Use MyRandomResizedCrop: %s, \t %s"
                % MyRandomResizedCrop.get_candidate_image_size(),
                "sync=%s, continuous=%s"
                % (
                    MyRandomResizedCrop.SYNC_DISTRIBUTED,
                    MyRandomResizedCrop.CONTINUOUS,
                ),
            )
        else:
            resize_transform_class = transforms.RandomResizedCrop

        train_transforms = [
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]

        if self.auto_augment is not None:
            auto_augment_policy = self.auto_augment
            if self.auto_augment == "ta_wide":
                train_transforms.append(
                    autoaugment.TrivialAugmentWide(
                        interpolation=InterpolationMode.BILINEAR
                    )
                )
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                train_transforms.append(
                    autoaugment.AutoAugment(
                        policy=aa_policy, interpolation=InterpolationMode.BILINEAR
                    )
                )

        color_transform = None
        if self.distort_color == "torch":
            color_transform = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        elif self.distort_color == "tf":
            color_transform = transforms.ColorJitter(
                brightness=32.0 / 255.0, saturation=0.5
            )

        if color_transform is not None:
            train_transforms.append(color_transform)

        train_transforms.extend(
            [transforms.ToTensor(), self.normalize,]
        )

        if self.random_erase_prob > 0:
            train_transforms.append(transforms.RandomErasing(p=self.random_erase_prob))

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose(
            [
                transforms.Resize(int(math.ceil(image_size / 0.875))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[
                self.active_img_size
            ] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_sub_train_loader(
        self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        # used for resetting BN running statistics
        if self.__dict__.get("sub_train_%d" % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indices = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = self.train_dataset(
                self.build_train_transform(
                    image_size=self.active_img_size, print_log=False
                )
            )
            chosen_indices = rand_indices[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(
                    new_train_dataset,
                    num_replicas,
                    rank,
                    True,
                    np.array(chosen_indices),
                )
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    chosen_indices
                )
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset,
                batch_size=batch_size,
                sampler=sub_sampler,
                num_workers=num_worker,
                pin_memory=False,
            )
            self.__dict__["sub_train_%d" % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__["sub_train_%d" % self.active_img_size].append(
                    (images, labels)
                )
        return self.__dict__["sub_train_%d" % self.active_img_size]
