"""逐格正式训练启动器：支持 Exp1、Exp2 单折/多折与多 seed，不改变原标量入口。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Experiment.train_sfr_unet import DataBundle, add_training_arguments, parse_with_config, train


def resolve_folds(config, held_out=None):
    """优先识别单折 metadata，否则展开 prepare_data 生成的 folds.json。"""
    root = Path(config.cache_dir)
    metadata_path = root / f"{config.data_exp}metadata.json"
    requested = set(held_out or [])
    def selected(groups):
        return not requested or any(name in requested or name.removeprefix("disk_galaxy_") in requested for name in groups)
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        groups = metadata.get("held_out_groups", [])
        if requested and (metadata.get("experiment") != "exp2" or not selected(groups)):
            raise ValueError("--held-out-galaxy 与单折 metadata 不匹配。")
        if requested and not requested.issubset(set(groups) | {g.removeprefix("disk_galaxy_") for g in groups}):
            raise ValueError("存在未知 held-out galaxy。")
        return [config]
    path = root / f"{config.data_exp}folds.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "exp2" or manifest.get("split_strategy") != "leave_one_galaxy_out":
        raise ValueError("仅接受 Exp2 leave-one-galaxy-out manifest。")
    jobs, seen, available = [], set(), set()
    for fold in manifest["folds"]:
        prefix, groups = fold["cache_prefix"], fold["held_out_groups"]
        if prefix in seen or len(groups) != 1:
            raise ValueError("重复 fold 或非法留出组。")
        seen.add(prefix)
        available.update(groups)
        available.update(g.removeprefix("disk_galaxy_") for g in groups)
        if selected(groups):
            if not prefix.startswith(config.data_exp):
                raise ValueError("fold cache_prefix 未以 manifest 基础前缀开头。")
            cell_exp = config.cell_exp + prefix[len(config.data_exp):] if config.cell_exp else ""
            job = replace(config, data_exp=prefix, cell_exp=cell_exp)
            job.validate()
            actual = json.loads((root / f"{prefix}metadata.json").read_text(encoding="utf-8"))
            if actual.get("held_out_groups") != groups or set(actual.get("train_groups", [])) != set(fold["train_groups"]):
                raise ValueError("manifest 与单折 metadata 分组不一致。")
            jobs.append(job)
    if requested - available or not jobs:
        raise ValueError(f"未找到指定留出组：{sorted(requested - available)}")
    return jobs


def main(argv=None):
    parser = add_training_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--held-out-galaxy", action="append", help="完整运行名或去掉 disk_galaxy_ 的简称，可重复")
    parser.add_argument("--seeds", type=int, nargs="+", help="正式重复训练的随机种子，不参与参数选择")
    args, config = parse_with_config(parser, argv)
    torch.set_num_threads(config.cpu_threads)
    jobs = resolve_folds(config, args.held_out_galaxy)
    seeds = args.seeds if args.seeds is not None else [config.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("--seeds 不允许重复。")
    if config.resume and (len(jobs) != 1 or len(seeds) != 1):
        parser.error("断点恢复只支持单折、单 seed。")
    for job in jobs:
        for seed in seeds:
            replace(job, seed=seed).validate()
    reports = []
    for job in jobs:
        # 每折的数据和尺度只拟合一次，多个 seed 复用；不同折严格独立。
        bundle = DataBundle(job, evaluation=True)
        for seed in seeds:
            reports.append(train(replace(job, seed=seed), bundle=bundle))
    print(json.dumps([{k: v for k, v in report.items() if k != "evaluation"}
                      for report in reports], ensure_ascii=False, indent=2))
    return reports


if __name__ == "__main__":
    main()
