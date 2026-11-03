"""Exp1/Exp2 逐格 HPO：复用训练核心，固定选模指标，持久化 study 与可直接训练的最佳配置。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optuna
import torch

from Experiment.train_sfr_unet import (DataBundle, add_training_arguments, atomic_save,
                                      fingerprint, invariant_config, parse_with_config, train)
from Experiment.run_train_sfr_unet import resolve_folds


SEARCH_VERSION = 1
DEFAULT_SEARCH = ("lr", "weight_decay", "batch_size", "base_ch", "p_drop", "bottleneck_depth")
CATEGORICAL = {"batch_size": [32, 64, 128, 256], "base_ch": [16, 32, 64],
               "p_drop": [0., .1, .2, .3], "bottleneck_depth": [1, 2],
               "pos_num_bands": [2, 4, 8], "pos_max_freq": [2., 4., 8., 16.],
               "se_reduction": [4, 8, 16], "total_weight": [0., .3, 1., 3.], "negative_weight": [.3, 1., 3.]}
LOG_RANGES = {"lr": [1e-5, 3e-3], "weight_decay": [1e-7, 1e-3]}


def parse_search(values):
    values = [part.strip() for token in values for part in token.split(",") if part.strip()]
    allowed = set(CATEGORICAL) | set(LOG_RANGES)
    if not values or len(values) != len(set(values)) or set(values) - allowed:
        raise ValueError(f"搜索参数必须非空、不重复，可选 {sorted(allowed)}。")
    return tuple(sorted(values))


def study_contract(config, bundle, search, sampler, sampler_seed, pruning):
    fixed = invariant_config(config)
    for name in search:
        fixed.pop(name)
    return {"schema": "macnet_sfr_unet_hpo_v1", "search_version": SEARCH_VERSION,
            "data": bundle.contract, "fixed_config": fixed,
            "search": {name: {"categorical": CATEGORICAL[name]} if name in CATEGORICAL
                       else {"log_range": LOG_RANGES[name]} for name in search},
            "sampler": sampler, "sampler_seed": sampler_seed, "pruning": pruning,
            "runtime": {"device": config.device, "cpu_threads": config.cpu_threads},
            "software": {"torch": str(torch.__version__), "optuna": optuna.__version__},
            "selection": "run_macro(cell+total+negative), fixed coefficients=1"}


def optimize(config, *, search=DEFAULT_SEARCH, n_trials=50, timeout=28800,
             sampler="tpe", sampler_seed=42, pruning=True, held_out=None):
    """每折独立 study；n_trials 是本次新增尝试次数，timeout 是每折的秒数。"""
    search = parse_search(search)
    if n_trials <= 0 or timeout <= 0:
        raise ValueError("n_trials/timeout 必须为正。")
    if not 0 <= sampler_seed < 2**32:
        raise ValueError("sampler_seed 必须在 [0,2**32) 内。")
    if config.resume or config.stop_after_epochs:
        raise ValueError("HPO 不接受训练 resume/stop_after_epochs；study 自动续接已完成 trial。")
    if sampler not in {"tpe", "grid"}:
        raise ValueError("sampler 仅支持 tpe/grid。")
    if sampler == "grid" and any(k in LOG_RANGES for k in search):
        raise ValueError("grid 只接受离散参数；lr/weight_decay 请通过普通训练参数固定。")
    reports = []
    for job in resolve_folds(config, held_out):
        torch.set_num_threads(job.cpu_threads)
        bundle = DataBundle(job, evaluation=False)
        contract = study_contract(job, bundle, search, sampler, sampler_seed, pruning)
        identity = fingerprint(contract)
        name = f"{job.data_exp}SFRUNet_HPO_v{SEARCH_VERSION}_{identity[:12]}"
        database = (Path(job.cache_dir) / f"{name}.db").resolve()
        sampler_object = (optuna.samplers.TPESampler(seed=sampler_seed) if sampler == "tpe"
                          else optuna.samplers.GridSampler({k: CATEGORICAL[k] for k in search}, seed=sampler_seed))
        pruner = (optuna.pruners.MedianPruner(n_startup_trials=5,
                  n_warmup_steps=max(10, math.ceil(job.epochs*job.warmup_ratio))) if pruning
                  else optuna.pruners.NopPruner())
        storage = optuna.storages.RDBStorage(f"sqlite:///{database.as_posix()}")
        study = optuna.create_study(study_name=name, storage=storage,
                                    direction="minimize", load_if_exists=True, sampler=sampler_object, pruner=pruner)
        previous = study.user_attrs.get("contract")
        if previous is not None and fingerprint(previous) != identity:
            storage.remove_session(); storage.engine.dispose()
            raise ValueError("已有 study 契约不一致。")
        if previous is None and study.trials:
            storage.remove_session(); storage.engine.dispose()
            raise ValueError("拒绝接管无契约的已有 study。")
        study.set_user_attr("contract", contract)
        export_dir = database.parent / name
        export_dir.mkdir(exist_ok=True)
        sampler_path = export_dir / "sampler.pt"
        if sampler_path.is_file():
            saved_sampler = torch.load(sampler_path, map_location="cpu", weights_only=False)
            if saved_sampler.get("contract_sha256") != identity:
                storage.remove_session(); storage.engine.dispose()
                raise ValueError("sampler 状态与 study 不匹配。")
            study.sampler = saved_sampler["sampler"]
        print(f"Study={name}\nDatabase={database}\nSearch={search}; seed={job.seed}; fixed metric=macro joint", flush=True)

        def objective(trial):
            params = {key: trial.suggest_categorical(key, CATEGORICAL[key]) if key in CATEGORICAL
                      else trial.suggest_float(key, *LOG_RANGES[key], log=True) for key in search}
            trial_config = replace(job, **params, output_root=str(Path(job.output_root)/"HPO_SFRUNet"),
                                   run_name=f"{name}_Trial{trial.number}", tensorboard=False)
            trial.set_user_attr("resolved_config", asdict(trial_config))
            def callback(epoch, score):
                trial.report(score, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"epoch={epoch}, fixed validation score={score}")
            try:
                result = train(trial_config, bundle=bundle, hpo=True, epoch_callback=callback)
                trial.set_user_attr("output_dir", result["output_dir"])
                trial.set_user_attr("best_epoch", result["best_epoch"])
                return result["best_score"]
            except (FloatingPointError, torch.cuda.OutOfMemoryError) as exc:
                trial.set_user_attr("failure_reason", str(exc))
                raise optuna.TrialPruned(str(exc)) from exc
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        def export_best(study, trial):
            # 保存 sampler 的随机状态，避免每次重启都从同一随机序列重新建议。
            atomic_save(sampler_path, {"contract_sha256": identity, "sampler": study.sampler})
            complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not complete:
                return
            best = study.best_trial
            best_config = replace(job, **best.params, run_name="", resume="", stop_after_epochs=0)
            atomic_save(export_dir / "best_config.json", asdict(best_config), json_format=True)
            atomic_save(export_dir / "best_trial.json", {"trial": best.number, "value": best.value,
                        "params": best.params, "user_attrs": best.user_attrs, "contract_sha256": identity}, json_format=True)

        try:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[export_best])
            export_best(study, None)
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            result = {"study": name, "database": str(database), "completed_trials": len(completed),
                      "best_config": str(export_dir / "best_config.json") if completed else None}
            reports.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            # 释放 SQLAlchemy 连接池，Windows 上结束后可移动/清理数据库。
            storage.remove_session()
            storage.engine.dispose()
    return reports


def main(argv=None):
    parser = add_training_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--held-out-galaxy", action="append")
    parser.add_argument("--search-params", nargs="+", default=DEFAULT_SEARCH)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=28800)
    parser.add_argument("--sampler", choices=("tpe", "grid"), default="tpe")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--pruning", action=argparse.BooleanOptionalAction, default=True)
    args, config = parse_with_config(parser, argv)
    return optimize(config, search=args.search_params, n_trials=args.n_trials, timeout=args.timeout,
                    sampler=args.sampler, sampler_seed=args.sampler_seed, pruning=args.pruning,
                    held_out=args.held_out_galaxy)


if __name__ == "__main__":
    main()
