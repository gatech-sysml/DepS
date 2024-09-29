# DϵpS: Delayed ϵ-Shrinking for Faster Once-For-All Training
# Alind Khare, Aditya Annavajjala, Animesh Agarwal, Igor Fedorov, Hugo Latapie, Myungjin Lee, Alexey Tumanov
# Systems for Artificial Intelligence Lab (SAIL), Georgia Institute of Technology, Atlanta, USA
# European Conference on Computer Vision (ECCV) 2024

from deps.imagenet_classification.data_providers import (
    CIFARDataProvider,
    ImagenetDataProvider,
    ImageWoofDataProvider,
)
from deps.utils import build_optimizer, calculate_and_adjust_learning_rate

__all__ = ["RunConfig", "ImagenetRunConfig", "DistributedImageNetRunConfig"]


class RunConfig:
    def __init__(
        self,
        task,
        n_epochs,
        init_lr,
        lr_schedule_type,
        lr_schedule_param,
        dataset,
        train_batch_size,
        test_batch_size,
        valid_size,
        opt_type,
        opt_param,
        weight_decay,
        label_smoothing,
        no_decay_keys,
        mixup_alpha,
        model_init,
        validation_frequency,
        print_frequency,
        lr_gamma,
        min_lr,
    ):
        self.n_epochs = n_epochs
        self.task = task
        self.init_lr = init_lr
        self.lr_gamma = lr_gamma
        self.min_lr = min_lr

        print("LR Gamma: ", self.lr_gamma)

        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys
        self.mixup_alpha = mixup_alpha

        self.model_init = model_init
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith("_"):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def calc_and_adjust_lr(self, optimizer, epoch, gamma=0.1, batch=0, nBatch=None):
        return calculate_and_adjust_learning_rate(
            optimizer,
            self.opt_type,
            init_lr=self.init_lr,
            gamma=self.lr_gamma,
            epoch=epoch,
            n_epochs=self.n_epochs,
            batch=batch,
            nBatch=nBatch,
            lr_schedule_type=self.lr_schedule_type,
            lr_schedule_param=self.lr_schedule_param,
            min_lr=self.min_lr,
        )

    def warmup_adjust_learning_rate(
        self, optimizer, T_total, nBatch, epoch, batch=0, warmup_lr=0
    ):
        T_cur = epoch * nBatch + batch + 1
        new_lr = T_cur / T_total * (self.init_lr - warmup_lr) + warmup_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        return new_lr

    """ data provider """

    @property
    def data_provider(self):
        raise NotImplementedError

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    def random_sub_train_loader(
        self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        return self.data_provider.build_sub_train_loader(
            n_images, batch_size, num_worker, num_replicas, rank
        )

    """ optimizer """

    def build_optimizer(self, net_params):
        return build_optimizer(
            net_params,
            self.opt_type,
            self.opt_param,
            self.init_lr,
            self.weight_decay,
            self.no_decay_keys,
        )


class ImagenetRunConfig(RunConfig):
    def __init__(
        self,
        task,
        n_epochs=150,
        init_lr=0.05,
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        dataset="imagewoof",
        train_batch_size=256,
        test_batch_size=500,
        valid_size=None,
        opt_type="sgd",
        opt_param=None,
        weight_decay=4e-5,
        label_smoothing=0.1,
        no_decay_keys=None,
        model_init="he_fout",
        validation_frequency=1,
        print_frequency=10,
        n_worker=32,
        resize_scale=0.08,
        distort_color="tf",
        image_size=224,
        dynamic_batch_size=None,
        cifar_mode=None,
        lr_gamma=1,
        teacher_warmup=0,
        data_path=None,
        min_lr=0,
        random_erase_prob=0,
        auto_augment=None,
        rand_augment=None,
        mixup_alpha=0,
        cutmix_alpha=0,
        cutmix_minmax=0,
        **kwargs
    ):
        super(ImagenetRunConfig, self).__init__(
            task=task,
            n_epochs=n_epochs,
            init_lr=init_lr,
            lr_schedule_type=lr_schedule_type,
            lr_schedule_param=lr_schedule_param,
            dataset=dataset,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            valid_size=valid_size,
            opt_type=opt_type,
            opt_param=opt_param,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            no_decay_keys=no_decay_keys,
            mixup_alpha=mixup_alpha,
            model_init=model_init,
            validation_frequency=validation_frequency,
            print_frequency=print_frequency,
            lr_gamma=lr_gamma,
            min_lr=min_lr,
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.data_path = data_path
        self.num_networks_per_batch = dynamic_batch_size
        self.cifar_mode = cifar_mode

        self.teacher_warmup = teacher_warmup
        self.reorganize_flag = False  # Set reorganize_flag False UNTIL re_organize actually happens during training.

        self.random_erase_prob = random_erase_prob
        self.auto_augment = auto_augment

        self.rand_augment = rand_augment
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax

    @property
    def data_provider(self):
        if self.__dict__.get("_data_provider", None) is None:
            if (
                self.dataset == ImagenetDataProvider.name()
                or self.dataset == "imagenet-100"
            ):
                DataProviderClass = ImagenetDataProvider
                if self.dataset == "imagenet-100":
                    DataProviderClass.IS_IMAGENET_100 = True

            elif self.dataset == ImageWoofDataProvider.name():
                DataProviderClass = ImageWoofDataProvider
            elif self.dataset == CIFARDataProvider.name():
                DataProviderClass = CIFARDataProvider
            else:
                raise NotImplementedError
            if self.data_path is not None:
                DataProviderClass.DEFAULT_PATH = self.data_path

            self.__dict__["_data_provider"] = DataProviderClass(
                train_batch_size=self.train_batch_size,
                test_batch_size=self.test_batch_size,
                valid_size=self.valid_size,
                n_worker=self.n_worker,
                resize_scale=self.resize_scale,
                distort_color=self.distort_color,
                image_size=self.image_size,
                random_erase_prob=self.random_erase_prob,
                auto_augment=self.auto_augment,
                rand_augment=self.rand_augment,
                mixup_alpha=self.mixup_alpha,
                cutmix_alpha=self.cutmix_alpha,
                cutmix_minmax=self.cutmix_minmax,
            )

        return self.__dict__["_data_provider"]


class DistributedImageNetRunConfig(ImagenetRunConfig):
    def __init__(
        self,
        task,
        n_epochs=150,
        init_lr=0.05,
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        dataset="imagewoof",
        train_batch_size=64,
        test_batch_size=64,
        valid_size=None,
        opt_type="sgd",
        opt_param=None,
        weight_decay=4e-5,
        label_smoothing=0.1,
        no_decay_keys=None,
        model_init="he_fout",
        validation_frequency=1,
        print_frequency=10,
        n_worker=8,
        resize_scale=0.08,
        distort_color="tf",
        image_size=224,
        dynamic_batch_size=None,
        cifar_mode=None,
        lr_gamma=1,
        data_path=None,
        teacher_warmup=0,
        min_lr=0,
        random_erase_prob=0,
        auto_augment=None,
        rand_augment=None,
        mixup_alpha=0,
        cutmix_alpha=0,
        cutmix_minmax=0,
        **kwargs
    ):
        super(DistributedImageNetRunConfig, self).__init__(
            task=task,
            n_epochs=n_epochs,
            init_lr=init_lr,
            lr_schedule_type=lr_schedule_type,
            lr_schedule_param=lr_schedule_param,
            dataset=dataset,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            valid_size=valid_size,
            opt_type=opt_type,
            opt_param=opt_param,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            no_decay_keys=no_decay_keys,
            model_init=model_init,
            validation_frequency=validation_frequency,
            print_frequency=print_frequency,
            n_worker=n_worker,
            resize_scale=resize_scale,
            distort_color=distort_color,
            image_size=image_size,
            dynamic_batch_size=dynamic_batch_size,
            cifar_mode=cifar_mode,
            lr_gamma=lr_gamma,
            data_path=data_path,
            teacher_warmup=teacher_warmup,
            min_lr=min_lr,
            random_erase_prob=random_erase_prob,
            auto_augment=auto_augment,
            rand_augment=rand_augment,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            cutmix_minmax=cutmix_minmax,
            **kwargs
        )

        self._num_replicas = kwargs["num_replicas"]
        self._rank = kwargs["rank"]

    @property
    def data_provider(self):
        if self.__dict__.get("_data_provider", None) is None:
            if (
                self.dataset == ImagenetDataProvider.name()
                or self.dataset == "imagenet-100"
            ):
                DataProviderClass = ImagenetDataProvider
                if self.dataset == "imagenet-100":
                    DataProviderClass.IS_IMAGENET_100 = True
            elif self.dataset == ImageWoofDataProvider.name():
                DataProviderClass = ImageWoofDataProvider
            elif self.dataset == CIFARDataProvider.name():
                DataProviderClass = CIFARDataProvider
            else:
                raise NotImplementedError

            if self.data_path is not None:
                DataProviderClass.DEFAULT_PATH = self.data_path

            self.__dict__["_data_provider"] = DataProviderClass(
                train_batch_size=self.train_batch_size,
                test_batch_size=self.test_batch_size,
                valid_size=self.valid_size,
                n_worker=self.n_worker,
                resize_scale=self.resize_scale,
                distort_color=self.distort_color,
                image_size=self.image_size,
                num_replicas=self._num_replicas,
                rank=self._rank,
                random_erase_prob=self.random_erase_prob,
                auto_augment=self.auto_augment,
                rand_augment=self.rand_augment,
                mixup_alpha=self.mixup_alpha,
                cutmix_alpha=self.cutmix_alpha,
                cutmix_minmax=self.cutmix_minmax,
                cifar_mode=self.cifar_mode,
            )

        return self.__dict__["_data_provider"]
