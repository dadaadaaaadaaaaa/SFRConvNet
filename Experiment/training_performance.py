"""小网格训练的可选数据路径；只改变取样和数据驻留方式，不改变监督契约。"""

import math
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler


class TensorBatchLoader:
    """对内存 tensor 整批索引；可将所需字段一次性复制到 GPU。

    与默认 DataLoader 使用相同的 CPU sampler，并消费相同的迭代器 seed。
    不支持在线增强，所有字段使用同一 batch 索引。不会修改原始 dataset。
    """

    def __init__(self, dataset, batch_size, *, shuffle, device="cpu", batches=None):
        if getattr(dataset, "augmentor", None) is not None:
            raise ValueError("整批 tensor 路径暂不支持在线增强。")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正。")
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.batches = batches
        self.sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
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


def configure_position_cache(model, datasets, *, augmentor):
    """仅对跨 split 一致的固定网格预计算；前向仍检查输入，变网格自动回退。"""
    if augmentor is not None:
        raise ValueError("固定位置编码缓存不能与坐标增强同时启用。")
    reference = datasets[0].coord[:1]
    for dataset in datasets:
        if not torch.equal(dataset.coord, reference.expand_as(dataset.coord)):
            raise ValueError("位置编码缓存需要所有输入样本共享同一坐标网格。")
    encoding = model.posenc
    coord = reference.to(next(model.parameters()).device)
    with torch.no_grad():
        encoding._fixed_features = None
        features = encoding(coord[:, :1], coord[:, 1:2], normalize=True)
        encoding._fixed_r = coord[:, :1].clone()
        encoding._fixed_theta = coord[:, 1:2].clone()
        encoding._fixed_features = features
