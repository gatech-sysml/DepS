# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch
import torch.nn as nn

from deps.imagenet_classification.elastic_nn.training.training import validate
from deps.utils import list_mean

__all__ = [
    "load_models",
    "train_elastic_depth",
    "train_elastic_expand",
    "train_elastic_width_mult",
]


def load_models(run_manager, dynamic_net, model_path=None):
    assert model_path is not None
    init = torch.load(model_path, map_location="cpu")["state_dict"]
    dynamic_net.load_state_dict(init)
    run_manager.write_log("Loaded init from %s" % model_path, "valid")


def train_elastic_depth(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    depth_stage_list = dynamic_net.depth_list.copy()
    depth_stage_list.sort(reverse=True)
    n_stages = len(depth_stage_list) - 1
    current_stage = n_stages - 1

    # load pretrained models
    if run_manager.start_epoch == 0 and not args.ps_resume:
        validate_func_dict["depth_list"] = sorted(dynamic_net.depth_list)
        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)

        (
            val_subnets,
            val_losses,
            val_accs,
            val_accs5,
            _val_log,
            _val_log_dict,
        ) = validate(run_manager, "valid", 0, **validate_func_dict)
        (
            test_subnets,
            test_losses,
            test_accs,
            test_accs5,
            _test_log,
            _test_log_dict,
        ) = validate(run_manager, "test", 0, **validate_func_dict)
    else:
        assert args.ps_resume

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Depth: %s -> %s"
        % (depth_stage_list[: current_stage + 1], depth_stage_list[: current_stage + 2])
        + "-" * 30,
        "valid",
    )
    # add depth list constraints
    if (
        len(set(dynamic_net.ks_list)) == 1
        and len(set(dynamic_net.expand_ratio_list)) == 1
    ):
        validate_func_dict["depth_list"] = depth_stage_list
    else:
        validate_func_dict["depth_list"] = sorted(
            {min(depth_stage_list), max(depth_stage_list)}
        )

    val_subnets, val_losses, val_accs, val_accs5, _val_log, _val_log_dict = validate(
        run_manager, "valid", 0, **validate_func_dict
    )
    (
        test_subnets,
        test_losses,
        test_accs,
        test_accs5,
        _test_log,
        _test_log_dict,
    ) = validate(run_manager, "test", 0, **validate_func_dict)

    val_loss = list_mean(val_losses)
    val_acc = list_mean(val_accs)
    val_acc5 = list_mean(val_accs5)
    test_loss = list_mean(test_losses)
    test_acc = list_mean(test_accs)
    test_acc5 = list_mean(test_accs5)

    val_log = "Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})".format(
        0 + 1 - args.warmup_epochs,
        run_manager.run_config.n_epochs,
        val_loss,
        val_acc,
        run_manager.best_acc,
    )
    val_log += _val_log
    run_manager.write_log(val_log, "valid", should_print=False)

    test_log = "Test [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})".format(
        0 + 1 - args.warmup_epochs,
        run_manager.run_config.n_epochs,
        test_loss,
        test_acc,
        run_manager.best_acc,
    )
    test_log += _test_log
    run_manager.write_log(test_log, "Test", should_print=False)

    train_func(
        run_manager,
        args,
        lambda _run_manager, mode, epoch: validate(
            _run_manager, mode, epoch, **validate_func_dict
        ),
    )


def train_elastic_expand(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    expand_stage_list = dynamic_net.expand_ratio_list.copy()
    expand_stage_list.sort(reverse=True)
    n_stages = len(expand_stage_list) - 1
    current_stage = n_stages - 1

    # load pretrained models
    if run_manager.start_epoch == 0 and not args.ps_resume:
        validate_func_dict["expand_ratio_list"] = sorted(dynamic_net.expand_ratio_list)

        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        dynamic_net.re_organize_middle_weights(expand_ratio_stage=current_stage)
        (
            val_subnets,
            val_losses,
            val_accs,
            val_accs5,
            _val_log,
            _val_log_dict,
        ) = validate(run_manager, "valid", 0, **validate_func_dict)
        (
            test_subnets,
            test_losses,
            test_accs,
            test_accs5,
            _test_log,
            _test_log_dict,
        ) = validate(run_manager, "test", 0, **validate_func_dict)

    #        # run_manager.write_log(
    #        #     "%.3f\t%.3f\t%.3f\t%s"
    #        #     % validate(run_manager, "test", 0, **validate_func_dict),
    #     "valid",
    # )
    else:
        assert args.ps_resume

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Expand Ratio: %s -> %s"
        % (
            expand_stage_list[: current_stage + 1],
            expand_stage_list[: current_stage + 2],
        )
        + "-" * 30,
        "valid",
    )
    if len(set(dynamic_net.ks_list)) == 1 and len(set(dynamic_net.depth_list)) == 1:
        validate_func_dict["expand_ratio_list"] = expand_stage_list
    else:
        validate_func_dict["expand_ratio_list"] = sorted(
            {min(expand_stage_list), max(expand_stage_list)}
        )

    # train
    validate(run_manager, "valid", 0, **validate_func_dict)
    validate(run_manager, "test", 0, **validate_func_dict)

    train_func(
        run_manager,
        args,
        lambda _run_manager, mode, epoch: validate(
            _run_manager, mode, epoch, **validate_func_dict
        ),
    )


def train_elastic_width_mult(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    width_stage_list = dynamic_net.width_mult_list.copy()
    width_stage_list.sort(reverse=True)
    n_stages = len(width_stage_list) - 1
    current_stage = n_stages - 1

    if run_manager.start_epoch == 0 and not args.ps_resume:
        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        if current_stage == 0:
            dynamic_net.re_organize_middle_weights(
                expand_ratio_stage=len(dynamic_net.expand_ratio_list) - 1
            )
            run_manager.write_log(
                "reorganize_middle_weights (expand_ratio_stage=%d)"
                % (len(dynamic_net.expand_ratio_list) - 1),
                "valid",
            )
            try:
                dynamic_net.re_organize_outer_weights()
                run_manager.write_log("reorganize_outer_weights", "valid")
            except Exception:
                pass
        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, mode="valid", **validate_func_dict),
            "valid",
        )
    else:
        assert args.ps_resume

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Width Mult: %s -> %s"
        % (width_stage_list[: current_stage + 1], width_stage_list[: current_stage + 2])
        + "-" * 30,
        "valid",
    )
    validate_func_dict["width_mult_list"] = sorted({0, len(width_stage_list) - 1})

    # train
    validate(run_manager, "valid", 0, **validate_func_dict)
    validate(run_manager, "test", 0, **validate_func_dict)

    train_func(
        run_manager,
        args,
        lambda _run_manager, mode, epoch: validate(
            _run_manager, mode, epoch, **validate_func_dict
        ),
    )
