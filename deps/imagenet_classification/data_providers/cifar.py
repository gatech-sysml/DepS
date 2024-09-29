# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024


import os
import warnings

import numpy as np
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from deps.utils.my_dataloader import MyDistributedSampler

from .base_provider import DataProvider

__all__ = ["CIFARDataProvider"]


class CIFARDataProvider(DataProvider):
    DEFAULT_PATH = "/nethome/sannavajjala6/data/cifar/"

    def __init__(
        self,
        save_path=None,
        train_batch_size=128,
        test_batch_size=64,
        valid_size=None,
        n_worker=32,
        image_size=32,
        num_replicas=None,
        cifar_mode="cifar100",
        rank=None,
        **kwargs,
    ):

        warnings.filterwarnings("ignore")
        self._save_path = save_path
        self.kwargs = kwargs
        self.image_size = image_size  # int or list of int
        self.cifar_mode = cifar_mode

        self._valid_transform_dict = {}
        assert isinstance(self.image_size, int)
        if not isinstance(self.image_size, int):
            raise NotImplementedError
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

    @staticmethod
    def name():
        return "cifar"

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        if self.cifar_mode == "cifar100":
            return 100
        elif self.cifar_mode == "cifar10":
            return 10
        else:
            raise NotImplementedError

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = "/nethome/sannavajjala6/data/cifar/"
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser(f"~/data/{self.cifar_mode}/")

        return self._save_path

    def train_dataset(self, _transforms):
        if self.cifar_mode == "cifar100":
            print("USING CIFAR 100")
            return datasets.CIFAR100(
                "./data", train=True, transform=_transforms, download=True
            )
        elif self.cifar_mode == "cifar10":
            print("USING CIFAR 10")
            return datasets.CIFAR10(
                "./data", train=True, transform=_transforms, download=True
            )
        else:
            raise NotImplementedError

    def test_dataset(self, _transforms):
        if self.cifar_mode == "cifar100":
            print("USING CIFAR 100")
            return datasets.CIFAR100(
                "./data", train=False, transform=_transforms, download=True
            )
        elif self.cifar_mode == "cifar10":
            print("USING CIFAR 10")
            return datasets.CIFAR10("./data", train=False, transform=_transforms)
        else:
            raise NotImplementedError

    @property
    def normalize(self):
        if self.cifar_mode == "cifar100":
            mean = [0.5071, 0.4867, 0.4408]
            std = [0.2675, 0.2565, 0.2761]
        elif self.cifar_mode == "cifar10":
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        else:
            raise NotImplementedError

        return transforms.Normalize(mean=mean, std=std)

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print("img_size: %s" % (image_size))

        if isinstance(image_size, list):
            raise NotImplementedError
        else:
            resize_transform_class = transforms.RandomCrop

        train_transforms = [
            transforms.RandomHorizontalFlip(),
            resize_transform_class(image_size, padding=4, padding_mode="reflect"),
        ]

        train_transforms += [
            transforms.ToTensor(),
            self.normalize,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size

        # Don't do any transforms on valid dataset. Just normalization
        return transforms.Compose([transforms.ToTensor(), self.normalize])

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
