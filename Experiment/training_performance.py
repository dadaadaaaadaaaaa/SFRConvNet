"""小网格训练的可选数据路径；只改变取样和数据驻留方式，不改变监督契约。"""

import math
import copy
from pathlib import Path
from collections import OrderedDict
import torch
from torch.utils.data import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)


class DatasetCache:
    """在同一个 HPO 进程中复用已校验的 CPU dataset，文件变化时失效。

    缓存只保存无增强的对象；返回浅拷贝以隔离逐 trial 的增强器设置。
    tensor 视为只读，训练路径不得对 dataset 字段进行原地修改。
    """

    def __init__(self, max_entries=4):
        if max_entries < 1:
            raise ValueError("max_entries 必须为正。")
        self.max_entries = max_entries
        self.entries = OrderedDict()

    def load(self, dataset_type, path, *, objective_mode, augmentor=None):
        path = Path(path).resolve()
        prefix = str(path)
        for split in ("train.pt", "val.pt", "test.pt", "out.pt"):
            if prefix.endswith(split):
                prefix = prefix[:-len(split)]
                break
        metadata = Path(prefix + "metadata.json")
        def signature(file):
            stat = file.stat()
            return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        key = (dataset_type, str(path), objective_mode, signature(path),
               signature(metadata) if metadata.exists() else None)
        if key not in self.entries:
            # 清除同一路径的旧版本，避免保留过期的大 tensor。
            for old in list(self.entries):
                if old[1] == str(path) and old[2] == objective_mode:
                    del self.entries[old]
            self.entries[key] = dataset_type(path, objective_mode=objective_mode)
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        result = copy.copy(self.entries[key])
        result.augmentor = augmentor
        return result


def compute_run_time_sampling_weights(time, groups):
    """构造运行等权、运行内按梯形时间区间加权的抽样权重。"""

    time = torch.as_tensor(time, dtype=torch.float64).view(-1).cpu()
    groups = torch.as_tensor(groups, dtype=torch.long).view(-1).cpu()
    if time.numel() != groups.numel() or time.numel() == 0:
        raise ValueError("time/groups 必须是长度一致的非空一维数组。")
    if not torch.isfinite(time).all():
        raise ValueError("run_time_balanced sampling 要求所有物理时间有限。")

    weights = torch.zeros_like(time, dtype=torch.float64)
    unique_groups = torch.unique(groups, sorted=True)
    for group in unique_groups.tolist():
        indices = torch.nonzero(groups == group, as_tuple=False).view(-1)
        group_time = time.index_select(0, indices)
        order = torch.argsort(group_time)
        sorted_time = group_time.index_select(0, order)
        if len(sorted_time) == 1:
            temporal = torch.ones(1, dtype=torch.float64)
        else:
            delta = torch.diff(sorted_time)
            if torch.any(delta <= 0):
                raise ValueError(
                    "run_time_balanced sampling 要求同一运行内时间严格递增；"
                    f"group={group} 存在重复或倒序时间。"
                )
            temporal = torch.empty_like(sorted_time)
            temporal[0] = 0.5 * delta[0]
            temporal[-1] = 0.5 * delta[-1]
            if len(sorted_time) > 2:
                temporal[1:-1] = 0.5 * (delta[:-1] + delta[1:])
        temporal = temporal / temporal.sum()
        sorted_indices = indices.index_select(0, order)
        weights.index_copy_(0, sorted_indices, temporal)

    # 每个运行归一化为相同总权重；WeightedRandomSampler 会再做全局归一化。
    if torch.any(weights <= 0) or not torch.isfinite(weights).all():
        raise ValueError("run_time_balanced sampling 产生了非正或非有限权重。")
    return weights


def make_training_loader(dataset, args, device, *, shuffle):
    """统一常规和整批数据路径；worker 数使用用户指定值，不再隐式除二。"""
    sampling_mode = getattr(args, "sampling_mode", "uniform")
    if sampling_mode not in {"uniform", "run_time_balanced"}:
        raise ValueError(f"未知 sampling_mode：{sampling_mode}")
    sampler = None
    if shuffle and sampling_mode == "run_time_balanced":
        weights = compute_run_time_sampling_weights(dataset.time, dataset.groups)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
        )
    backend = getattr(args, "batch_backend", "standard")
    if backend != "standard":
        if args.num_workers != 0:
            raise ValueError("tensor 批量路径不使用 worker，请设置 num_workers=0。")
        if backend != "tensor_cpu":
            raise ValueError(f"未知 batch_backend：{backend}")
        return TensorBatchLoader(
            dataset,
            args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            device="cpu",
        )
    from torch.utils.data import DataLoader
    import os
    options = {}
    if args.num_workers > 0:
        options.update(persistent_workers=args.persistent_workers or os.name == "nt",
                       prefetch_factor=max(1, args.prefetch_factor))
    return DataLoader(dataset, batch_size=args.batch_size,
                      shuffle=shuffle if sampler is None else False,
                      sampler=sampler,
                      num_workers=args.num_workers, pin_memory=device.type == "cuda",
                      drop_last=False, **options)


class TensorBatchLoader:
    """对内存 tensor 整批索引；可将所需字段一次性复制到 GPU。

    与默认 DataLoader 使用相同的 CPU sampler，并消费相同的迭代器 seed。
    不支持在线增强，所有字段使用同一 batch 索引。不会修改原始 dataset。
    """

    def __init__(
        self,
        dataset,
        batch_size,
        *,
        shuffle,
        sampler=None,
        device="cpu",
        batches=None,
    ):
        if getattr(dataset, "augmentor", None) is not None:
            raise ValueError("整批 tensor 路径暂不支持在线增强。")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正。")
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.batches = batches
        self.sampler = sampler if sampler is not None else (
            RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        )
        names = ["x", "coord", "conditioning_scalar", "y"]
        if dataset.objective_mode == "signed_closure":
            names += ["macer_sfr_baseline", "target_sfr_physical"]
        names += ["groups"]
        self.fields = tuple(getattr(dataset, name).to(self.device) for name in names)

    def __len__(self):
        return len(self.batches) if self.batches is not None else math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        # 匹配 DataLoader 初始化迭代器的 RNG 消耗，保持同种子批次顺序。
        torch.empty((), dtype=torch.int64).random_()
        batches = self.batches if self.batches is not None else BatchSampler(
            self.sampler, self.batch_size, drop_last=False)
        for indices in batches:
            index = torch.as_tensor(indices, dtype=torch.long, device=self.device)
            values = tuple(field.index_select(0, index) for field in self.fields)
            yield (*values, self.dataset.n_groups)
