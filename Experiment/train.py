import math
import os
import random
import sys
import json
from typing import Optional, Tuple

# Make project root importable
cwd = os.getcwd()
project_root = os.path.abspath(os.path.join(cwd, ".."))
for path in {cwd, project_root}:
    if path not in sys.path:
        sys.path.append(path)

import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from Experiment.models import AccretionTransformer, SFRConvNet, MACNetRes_mbh, MACNetFiLM


MBH_CONDITIONED_MODELS = (MACNetRes_mbh, MACNetFiLM)


def predict_batch(model, x: torch.Tensor, coord: torch.Tensor, conditioning_scalar: torch.Tensor) -> torch.Tensor:
    """Run one batch with the conditioning scalar required by the model family.

    Current SFR caches store transformed KS-law SFR. Legacy ``*_mbh`` models
    require an explicit BH-mass cache and must not silently consume KS SFR.
    """
    if isinstance(model, MBH_CONDITIONED_MODELS):
        conditioning_variable = getattr(model, "conditioning_variable", "unknown")
        if conditioning_variable != "mbh":
            raise RuntimeError(
                f"{type(model).__name__} requires conditioning_variable='mbh', "
                f"but the loaded cache provides {conditioning_variable!r}."
            )
        return model(x, conditioning_scalar)
    r = coord[:, 0:1]
    theta = coord[:, 1:2]
    # For the current SFR pipeline this scalar is transformed KS-law SFR.
    # SFRConvNet ignores it when use_film=False.
    return model(x, r, theta, conditioning_scalar)


# =============================================
# 0) Augmentations
# =============================================
class RadialCropAugmentor: # 在针对SFR的预测中，不再需要该功能，触发概率设置为0
    """
    Implements radial cropping from outer to inner.
    Data format: (C, H, W), Coord format: (2, H, W) where coord[0] is r, coord[1] is theta.

    The augmentation randomly selects a cutoff radius 'b' between (r_max * min_ratio) and r_max.
    All data points where r > b are masked (set to 0).
    
    数据增强。r较小的地方是核心，物理梯度极大；r较大的地方则可能受到初始条件和人为边界的干扰，对真实吸积率影响较小甚至可能包含噪音
    如何实现？有50%的概率触发裁剪，在[r_max * 0.8, r_max] 之间随机抽取一个半径 b，然后对所有半径大于b的网格强行设置为0
    """

    def __init__(self, min_crop_ratio: float = 0.8, p_crop: float = 0.5):
        self.min_crop_ratio = max(0.0, min(1.0, min_crop_ratio))
        self.p_crop = max(0.0, min(1.0, p_crop))

    def __call__(self, x: torch.Tensor, coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() >= self.p_crop:
            return x, coord

        # coord shape: (2, H, W), coord[0] is r
        r_map = coord[0]
        r_max = float(r_map.max())
        r_min = float(r_map.min())  # Usually close to 0

        # Determine cutoff b
        # We want to keep range (0, b).
        # b should be randomly chosen in [r_max * min_ratio, r_max]
        # But closer to r_min logic: technically user said "0 < b < a".
        # We ensure b is at least retaining a significant portion of the core.

        lower_bound = r_max * self.min_crop_ratio
        if lower_bound <= r_min:
            lower_bound = r_min + (r_max - r_min) * 0.1  # Safety guard

        b = random.uniform(lower_bound, r_max)

        # Create mask: True where r > b (the outer part to be removed/masked)
        mask = r_map > b

        if not mask.any():
            return x, coord

        x = x.clone()
        # Apply mask to all channels
        # x is (C, H, W), mask is (H, W)
        x[:, mask] = 0.0

        return x, coord

class SFRSpatialAugmentor:
    """
    针对恒星形成率(SFR)的数据增强。
    保留全空间信息，利用物理系统的旋转对称性和镜像对称性进行数据增强。
    假设输入图像空间维度为 (H, W)，根据网格特性进行翻转或平移。
    """
    def __init__(self, p_flip: float = 0.5, p_rot: float = 0.5):
        self.p_flip = p_flip
        self.p_rot = p_rot

    def __call__(self, x: torch.Tensor, coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_aug, coord_aug = x.clone(), coord.clone()

        # 1. 随机镜像翻转 (假设 W 维度为某空间轴，或者 H,W 都是直角坐标轴)
        if random.random() < self.p_flip:
            # 水平翻转特征图
            x_aug = torch.flip(x_aug, dims=[-1])
            # 注意：如果 coord 包含 r 和 theta，翻转需要对应修改 theta
            # 假设 coord[1] 是 theta，范围是 [0, 2pi]，水平翻转等价于 theta -> pi - theta
            # 如果是笛卡尔坐标的图像，可以直接 flip coord 的对应轴
            coord_aug[1] = math.pi - coord_aug[1] 
            # 规范化到 [0, 2pi]或其他您的设定范围
            # coord_aug[1] = torch.remainder(coord_aug[1], 2 * math.pi)

        # 2. 若数据是极坐标网格 (r 在 H 轴，theta 在 W 轴)
        # 通过在 W 轴上循环平移(roll) 来实现物理上的随机旋转
        if random.random() < self.p_rot:
            w_size = x_aug.shape[-1]
            shift = random.randint(0, w_size - 1)
            x_aug = torch.roll(x_aug, shifts=shift, dims=-1)
            coord_aug = torch.roll(coord_aug, shifts=shift, dims=-1)
            #平移后，更新 theta 坐标数值（视您的坐标系范围而定）

        return x_aug, coord_aug
    
# =============================================
# 1) Dataset: current SFR PT format
#    [x, coord, target_sfr, transformed_ks_sfr, groups, n_groups, time]
# =============================================
def _load_cache_channel_metadata(pt_path):
    pt_path = os.fspath(pt_path)
    prefix = None
    for suffix in ("train.pt", "val.pt", "test.pt", "out.pt"):
        if pt_path.endswith(suffix):
            prefix = pt_path[: -len(suffix)]
            break
    if prefix is None:
        return {}

    metadata_path = prefix + "metadata.json"
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read cache metadata '{metadata_path}'.") from exc


class GalPTDataset(Dataset):
    def __init__(self, pt_path, *, augmentor: Optional[SFRSpatialAugmentor] = None):
        super().__init__()
        try:
            loaded = torch.load(pt_path, weights_only=False) # 从prepare_data输出的pt文件中读取
        except Exception as exc:
            raise RuntimeError(f"Failed to load dataset cache '{pt_path}'. ") from exc

        if isinstance(loaded, (list, tuple)):
            if len(loaded) >= 6:
                self.x, self.coord, self.y, self.conditioning_scalar, self.groups, self.n_groups = loaded[:6]
                self.time = loaded[6] if len(loaded) > 6 else None
                if len(loaded) > 7:
                    print(
                        f"[Warning] Dataset '{pt_path}' has {len(loaded)} items; "
                        "fields after time are ignored."
                    )
            else:
                raise RuntimeError(f"Unexpected number of items in cached dataset '{pt_path}': {len(loaded)}")
        else:
            raise RuntimeError(f"Cached dataset '{pt_path}' should be a tuple/list but got {type(loaded)!r}")

        def _ensure_tensor(arr, *, name: str):
            if isinstance(arr, torch.Tensor):
                return arr.to(dtype=torch.float32)

            # Special handling for potentially string-based 'groups'
            if name == "groups":
                if isinstance(arr, list):
                    arr = np.array(arr)
                if isinstance(arr, np.ndarray) and arr.dtype.kind in "SU":
                    return arr  # Keep as numpy string array for later processing

            try:
                return torch.as_tensor(arr, dtype=torch.float32)
            except Exception as inner_exc:
                raise TypeError(
                    f"Cached field '{name}' in '{pt_path}' cannot be converted " "to a float32 tensor."
                ) from inner_exc

        self.x = _ensure_tensor(self.x, name="x").contiguous() # 强制把特征图（x，y）和坐标coord在内存中连续排列，可以大幅提升GPU读取数据的速度
        self.coord = _ensure_tensor(self.coord, name="coord").contiguous()
        self.y = _ensure_tensor(self.y, name="y").view(-1, 1) # 将目标标量强行变形为[样本数，1]的二维张量，为了和神经网络的输出形状对齐
        self.conditioning_scalar = _ensure_tensor(
            self.conditioning_scalar, name="conditioning_scalar"
        ).view(-1, 1)

        channel_metadata = _load_cache_channel_metadata(pt_path)
        self.cache_metadata = channel_metadata
        self.pt_path = os.fspath(pt_path)
        self.conditioning_variable = channel_metadata.get("conditioning_variable")
        if self.conditioning_variable is None:
            # Compatibility with instantaneous-SFR caches created before the
            # conditioning_variable metadata field was introduced.
            if channel_metadata.get("target_quantity") == "instantaneous_sfr":
                self.conditioning_variable = "sfr_ks"
            else:
                self.conditioning_variable = "unknown"
        if self.conditioning_variable == "sfr_ks":
            self.sfr_ks_transformed = self.conditioning_scalar
        # Read-only compatibility alias for older analysis/training helpers.
        self.y_baseline = self.conditioning_scalar
        self.input_channels = tuple(channel_metadata.get("input_channels", ()))
        self.excluded_channels = tuple(channel_metadata.get("excluded_channels", ()))
        self.target_metadata = {
            key: channel_metadata[key]
            for key in (
                "target_quantity",
                "target_name",
                "target_dataset",
                "target_reduction",
                "target_units",
                "target_transform",
                "target_scale",
                "target_floor",
                "includes_zero_targets",
            )
            if key in channel_metadata
        }
        if self.input_channels and len(self.input_channels) != self.x.shape[1]:
            raise RuntimeError(
                f"Cache '{pt_path}' has {self.x.shape[1]} tensor channels but metadata lists "
                f"{len(self.input_channels)} channels: {self.input_channels}"
            )

        groups_raw = _ensure_tensor(self.groups, name="groups")
        if isinstance(groups_raw, np.ndarray) and groups_raw.dtype.kind in "SU": # 如果groups是字符串类型（如星系类型），则保持为numpy字符串数组，稍后在内存中转换为整数标签；否则直接转换为float32张量
            # Handle string groups
            raw_groups = groups_raw.reshape(-1, 1)
            self.group_names = tuple(sorted(np.unique(raw_groups.astype(str)).tolist()))
            unique_groups, inv = np.unique(raw_groups, return_inverse=True)
            self.n_groups = len(unique_groups)
            self.groups = torch.from_numpy(inv).long().view(-1, 1)
        else:
            # Handle numeric groups (already a tensor from _ensure_tensor)
            if groups_raw.ndim == 2 and groups_raw.size(1) > 1:
                # Keep only the last column if multiple columns exist (e.g., [type, subtype])
                # to ensure one label per sample.
                raw_groups = groups_raw[:, -1:]
            else:
                raw_groups = groups_raw.view(-1, 1)

            # Remap groups to 0..n_groups-1 for robust indexing in eval_epoch
            unique_groups, inv = torch.unique(raw_groups, return_inverse=True)
            self.n_groups = len(unique_groups)
            self.groups = inv.long().view(-1, 1)
            self.group_names = tuple(str(value.item()) for value in unique_groups.reshape(-1))

        self.augmentor = augmentor

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        x = self.x[idx]
        coord = self.coord[idx]
        if self.augmentor is not None:
            x, coord = self.augmentor(x, coord) # 调用augmentor对特征图和坐标进行数据增强
        return x, coord, self.conditioning_scalar[idx], self.y[idx], self.groups[idx], self.n_groups


def _validate_leave_one_out_cache_contract(cache_dir, data_exp, datasets):
    """Validate new Exp2 caches against both metadata and actual PT group labels."""
    metadata = _load_cache_channel_metadata(os.path.join(cache_dir, data_exp + "train.pt"))
    if metadata.get("split_strategy") != "leave_one_galaxy_out":
        return metadata

    train_groups = set(metadata.get("train_groups", ()))
    held_out_groups = set(metadata.get("held_out_groups", ()))
    if not train_groups or len(held_out_groups) != 1:
        raise RuntimeError(
            "Invalid leave-one-galaxy-out metadata: expected non-empty train_groups "
            f"and exactly one held_out_group, got {metadata}."
        )
    overlap = train_groups.intersection(held_out_groups)
    if overlap:
        raise RuntimeError(
            f"Data leakage detected in cache metadata: train/out groups overlap: {sorted(overlap)}"
        )

    for split, dataset in datasets.items():
        actual = set(dataset.group_names)
        if actual != train_groups:
            raise RuntimeError(
                f"{split}.pt contains groups {sorted(actual)}, expected in-domain groups "
                f"{sorted(train_groups)} from metadata."
            )

    out_path = os.path.join(cache_dir, data_exp + "out.pt")
    if not os.path.exists(out_path):
        raise RuntimeError(
            f"Leave-one-galaxy-out cache is incomplete: required held-out cache not found: {out_path}"
        )
    out_dataset = GalPTDataset(out_path)
    actual_out = set(out_dataset.group_names)
    if actual_out != held_out_groups:
        raise RuntimeError(
            f"out.pt contains groups {sorted(actual_out)}, expected held-out groups "
            f"{sorted(held_out_groups)} from metadata."
        )
    if actual_out.intersection(train_groups):
        raise RuntimeError(
            "Data leakage detected in actual PT data: out.pt contains a training group."
        )
    print(
        "[PASS] Leave-one-galaxy-out cache contract verified: "
        f"train/val/test={sorted(train_groups)}, out={sorted(actual_out)}"
    )
    del out_dataset
    return metadata


# =============================================
# 2) Args
# =============================================
def parse_args(argv=None) -> argparse.Namespace: # 能够输入哪些训练参数
    parser = argparse.ArgumentParser(description="Train MACNetRes with MBH input")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Directory containing cached .pt datasets") # 去哪拿.pt数据
    parser.add_argument("--exp_name", type=str, default="Exp5_", help="Experiment name prefix used for saving results") # 这次做哪个实验
    parser.add_argument( # 从哪个实验的数据中加载.pt文件，默认和当前实验名相同
        "--data_exp", type=str, default=None, help="Experiment name prefix for loading data (default: same as exp_name)"
    )

    # ↓ 训练的超参数
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size") # 每次训练送入模型的样本数量
    parser.add_argument("--epochs", type=int, default=500, help="Maximum number of training epochs") # 最大训练轮数
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate") # 初始学习率
    parser.add_argument("--weight_decay", type=float, default=0, help="AdamW weight decay (L2 regularization)") # 权重衰减，AdamW优化器的L2正则化强度
    parser.add_argument("--lr_min", type=float, default=0.0, help="Minimum learning rate once cosine decay completes") # 余弦退火完成后的最大学习率，通常设置为0或一个很小的值，以允许学习率在训练后期非常小，从而微调模型参数
    parser.add_argument( # 学习率预热的参数，预热阶段学习率从lr_warmup_start_factor * lr线性增加到lr，持续lr_warmup_steps步；如果lr_warmup_steps=0，则禁用预热并直接使用lr
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of linear warmup steps before cosine decay (0 to disable)",
    )
    parser.add_argument( # 如果lr_warmup_steps=0，则使用lr_warmup_ratio指定预热占总训练步数的比例（如0.2表示20%），否则忽略lr_warmup_ratio
        "--lr_warmup_ratio",
        type=float,
        default=0.2,
        help="If warmup steps is 0, use this fraction of total steps for warmup (0 disables)",
    )
    parser.add_argument( # 预热阶段学习率的起始相对因子，0.0表示从0开始预热，1.0表示从lr开始预热（即没有预热效果），通常设置为0.0以实现完整的预热效果
        "--lr_warmup_start_factor",
        type=float,
        default=0.0,
        help="Relative learning rate factor to start warmup from (0.0 starts from zero)",
    )
    default_workers = 0 if os.name == "nt" else min(16, (os.cpu_count() or 2) // 2) 
    parser.add_argument("--num_workers", type=int, default=default_workers, help="Number of dataloader workers") # 加载数据的子进程数量，Windows系统默认0（主进程加载），Unix系统默认CPU核心数的一半（但不超过16）
    parser.add_argument(
        "--persistent_workers", action="store_true", help="Keep dataloader workers persistent between epochs" # 是否在epoch之间保持数据加载器的子进程持续运行，启用后可以减少每个epoch开始时的启动开销，但会占用更多系统资源；通常在num_workers > 0且系统资源允许的情况下启用
    )
    parser.add_argument("--prefetch_factor", type=int, default=3, help="Prefetch factor for dataloader (per worker)") # 每个数据加载器子进程预取的批次数量，默认3表示每个子进程在当前批次处理时预取3个批次的数据，以提高数据加载效率；如果num_workers=0，则此参数被忽略
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility") # 随机种子，用于保证实验的可重现性
    parser.add_argument("--print_group_counts", action="store_true", help="Print per-group sample counts for loaders") # 打印每个组的样本数量，有助于检查数据分布是否平衡
    
    # ↓ cpu/gpu相关的参数
    parser.add_argument( # 选择cpu还是cuda训练
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use: 'auto', 'cpu' or 'cuda'",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs without improvement)") # 早停耐心值，如果验证集指标在连续patience个epoch内没有改善，则提前停止训练
    parser.add_argument( # 是否禁用自动混合精度（AMP），即使CUDA可用。启用AMP可以在支持的GPU上加速训练并减少显存使用，但在某些情况下可能导致数值不稳定或性能下降；如果遇到问题，可以尝试添加--no_amp参数禁用AMP
        "--no_amp", action="store_true", help="Disable automatic mixed precision (AMP) even if CUDA is available"
    )
    parser.add_argument("--grad_clip", type=float, default=0.3, help="Max norm for gradient clipping (0 to disable)") # 梯度裁剪的最大范数，默认0.3表示如果启用梯度裁剪，当梯度的总范数超过0.3时，将其缩放回0.3以防止梯度爆炸；如果设置为0或负值，则禁用梯度裁剪
    parser.add_argument( # 是否启用cuDNN benchmark模式，启用后cuDNN会在训练开始时进行一些性能测试，以选择最适合当前模型和输入大小的卷积算法，从而可能加速训练；但在某些情况下（如输入大小变化较大时）可能导致性能不稳定或增加启动时间；如果遇到问题，可以尝试添加--cudnn_benchmark参数启用该模式
        "--cudnn_benchmark", action="store_true", help="Enable cuDNN benchmark mode for potential speedup"
    )

    # ↓ 损失函数的参数
    parser.add_argument("--loss", type=str, choices=["mse", "l1", "huber"], default="mse", help="Loss function to use") # 损失函数类型，支持均方误差（MSE）、平均绝对误差（L1）和Huber损失，默认使用MSE损失；Huber损失在误差较小时表现为MSE，在误差较大时表现为L1，可以在存在异常值时提供更稳健的训练；huber_delta参数控制Huber损失的转折点，默认为1
    parser.add_argument("--huber_delta", type=float, default=1, help="Delta parameter for Huber loss") # Huber损失的delta参数，控制损失函数从MSE模式切换到L1模式的阈值，默认值为1；较小的delta会使模型更快地切换到L1模式，从而对异常值更鲁棒；较大的delta会使模型在更大的误差范围内保持MSE模式，从而可能更好地拟合数据但对异常值更敏感
    
    # ↓ 目标加权模式的参数
    parser.add_argument( # 目标加权模式，支持基于指数函数的权重（exp_clamp），基于组内z-score的指数权重（group_zscore_exp）和基于组内分位数的线性权重（group_quantile）；由于SFR分布相对平滑，默认改为 none
        "--weight_mode",
        type=str,
        choices=["none", "exp_clamp", "group_zscore_exp", "group_quantile"],
        default="none",
        help="Weighting mode for targets",
    )
    parser.add_argument("--weight_alpha", type=float, default=0.5, help="Alpha for target weighting") # 权重函数的alpha参数，控制权重调整的强度，默认0.5表示在基础权重（1.0）的基础上增加最多50%的权重；较大的alpha会使权重调整更显著，可能有助于强调某些样本，但也可能导致训练不稳定；较小的alpha会使权重调整更温和，可能更稳定但效果有限
    parser.add_argument("--weight_tau", type=float, default=3.0, help="Temperature for z-score exp weights") # z-score指数权重的温度参数，控制权重对z-score的敏感度，默认3.0表示权重对z-score的变化较为平滑；较小的tau会使权重对z-score的变化更敏感，从而可能更强调极端样本；较大的tau会使权重对z-score的变化更平滑，从而可能更稳定但效果有限
    parser.add_argument("--weight_power", type=float, default=1.0, help="Power for quantile weights") # 分位数权重的幂参数，控制权重对样本在组内相对位置的敏感度，默认1.0表示线性调整；较小的power会使权重对相对位置的变化更温和；较大的power会使权重对相对位置的变化更显著，从而可能更强调位于分位数边界附近的样本
    parser.add_argument("--weight_q_low", type=float, default=0.1, help="Lower quantile for group quantile weights") # 分位数权重的下分位数参数，控制计算分位数权重时使用的下分位数，默认0.1表示使用组内第10百分位数作为下分位数；较小的weight_q_low会使更多样本被视为高权重样本；较大的weight_q_low会使只有相对较高的样本被视为高权重样本
    parser.add_argument("--weight_q_high", type=float, default=0.9, help="Upper quantile for group quantile weights") # 分位数权重的上分位数参数，控制计算分位数权重时使用的上分位数，默认0.9表示使用组内第90百分位数作为上分位数；较小的weight_q_high会使更多样本被视为低权重样本；较大的weight_q_high会使只有相对较低的样本被视为低权重样本
    
    # ↓ 各种模型的超参数
    parser.add_argument( # 选取哪种模型进行架构，参考model.py
        "--model_type",
        type=str,
        choices=["transformer", "resnet", "resfilm", "convnet"],
        default="transformer",
        help="Type of model architecture: 'transformer', 'resnet', 'resfilm', or 'convnet'",
    )
    parser.add_argument("--d_model", type=int, default=256, help="Model embedding dimension (d_model)")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward network hidden dimension")
    parser.add_argument("--pos_num_bands", type=int, default=32, help="Number of positional encoding frequency bands") 
    parser.add_argument("--pos_max_freq", type=float, default=101, help="Maximum positional encoding frequency") # 前6个都是transformer架构的超参数
    parser.add_argument("--p_drop", type=float, default=0.3, help="Dropout probability") # p_drop，base_ch，se_reduction是用于神经网络的超参数，需要控制
    parser.add_argument("--base_ch", type=int, default=64, help="ConvNet base channels (convnet only)")
    parser.add_argument(
        "--stage_depths",
        type=str,
        default="2,2,2,2",
        help="ConvNet stage depths as comma list (e.g., 2,2,2,2)",
    )
    parser.add_argument("--se_reduction", type=int, default=4, help="ConvNet SE reduction ratio (convnet only)") #
    parser.add_argument(
        "--use_film",
        action="store_true",
        help="Condition SFRConvNet feature maps on transformed KS-law SFR via FiLM.",
    )
    parser.add_argument(
        "--hpo_params",
        type=str,
        default=None,
        help="Path to Optuna/JSON HPO params file. Matching CLI args will be overridden",
    )

    # ↓ 数据增强（augmentation）的参数
    parser.add_argument("--aug_prob", type=float, default=0.0, help="Probability to apply radial crop") # 设置触发RadialCropAugmentor的概率为0，从而移除这个功能
    parser.add_argument("--aug_min_crop_ratio", type=float, default=0.6, help="Minimum ratio r_cut/r_max for cropping")
    parser.add_argument("--sfr_aug_prob", type=float, default=0.5, help="Probability to apply SFR spatial augmentation")

    parser.add_argument("--log_dir", type=str, default="runs", help="Directory to write TensorBoard logs")
    parser.add_argument("--no_save", action="store_true", help="Disable saving results to disk (useful for HPO)")

    return parser.parse_args(argv)


def _parse_stage_depths(value: str) -> Tuple[int, ...]: # 字符串转化为pytorch能够识别的元组的解析器
    raw = value.strip()
    if not raw:
        return (2, 2, 2, 2)

    if raw.startswith("[") or raw.startswith("("):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return tuple(int(x) for x in parsed)

    try:
        return tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid stage_depths value: {value!r}") from exc


def _apply_hpo_overrides(args: argparse.Namespace) -> argparse.Namespace: # 在每一轮训练中加载.json文件，转化为正确的数据类型，然后覆盖手动输入的预设参数。这样能够实现自动化的多次训练，只要给多个.json文件即可
    path = getattr(args, "hpo_params", None)
    if not path:
        return args
    if not os.path.isfile(path):
        print(f"[Warning] HPO params file not found: {path}")
        return args
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[Warning] Failed to read HPO params '{path}': {exc}")
        return args

    converters = {
        "lr": float,
        "weight_decay": float,
        "lr_min": float,
        "lr_warmup_steps": int,
        "lr_warmup_ratio": float,
        "lr_warmup_start_factor": float,
        "batch_size": int,
        "grad_clip": float,
        "patience": int,
        "loss": str,
        "huber_delta": float,
        "weight_mode": str,
        "weight_alpha": float,
        "weight_tau": float,
        "weight_power": float,
        "weight_q_low": float,
        "weight_q_high": float,
        "d_model": int,
        "n_layers": int,
        "n_heads": int,
        "d_ff": int,
        "pos_num_bands": int,
        "pos_max_freq": float,
        "p_drop": float,
        "seed": int,
        "aug_prob": float,
        "aug_min_crop_ratio": float,
        "use_film": bool,
    }

    applied = []
    for key, convert in converters.items():
        if key not in payload:
            continue
        try:
            value = convert(payload[key])
        except (TypeError, ValueError):
            print(f"[Warning] Could not cast HPO param '{key}' (value={payload[key]!r})")
            continue
        if key == "loss":
            value = value.lower()
        setattr(args, key, value)
        applied.append(key)

    if applied:
        print(f"Loaded HPO params from {path}; applied keys: {', '.join(applied)}")
    else:
        print(f"[Warning] No matching keys found in HPO params: {path}")

    return args


def set_seed(seed: int): # 固定随机数种子，保证实验的可复现性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism users can also set CUDNN flags; keep perf by default.


# =============================================
# 3) Train and evaluate
# =============================================
'''
由于标准的神经网络损失函数（如MSE）对所有样本一视同仁，不适用于像吸积率这样瞬间爆发的长尾分布。
因此先计算基础误差loss，然后再乘以一个动态权重矩阵weights。这样预测错平庸样本问题不大，但是预测错罕见的吸积爆发则会被放大权重惩罚。
'''
class WeightedLoss(nn.Module):
    def __init__(self, base_criterion: nn.Module, weights_fn: callable = None):
        """
        A loss wrapper that applies per-sample weights to the base criterion.
         Args:
             base_criterion (nn.Module  ): The base loss function (e.g., nn.MSELoss).
             weights_fn (callable, optional): A function that takes the target tensor `y` and returns
                 a tensor of weights with the same shape as `y`. If None, uniform weights are used.
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.weights_fn = weights_fn if weights_fn is not None else (lambda y, groups=None: torch.ones_like(y))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, groups: torch.Tensor | None = None) -> torch.Tensor:
        loss = self.base_criterion(pred, target)
        weights = self.weights_fn(target, groups)

        weighted_loss = loss * weights
        return weighted_loss.mean()


def _compute_group_stats( # 不同星系的 SFR或其它预测目标 分布不一样，这个函数遍历训练集，计算每一种星系目标的均值mean和标准差std，以及10%分位数p10和90%分位数p90
    y: torch.Tensor,
    groups: torch.Tensor,
    n_groups: int,
    q_low: float = 0.1,
    q_high: float = 0.9,
) -> dict[str, torch.Tensor]:
    y = y.view(-1)
    groups = groups.view(-1).long()
    means = torch.zeros(n_groups, dtype=y.dtype)
    stds = torch.ones(n_groups, dtype=y.dtype)
    p10 = torch.zeros(n_groups, dtype=y.dtype)
    p90 = torch.ones(n_groups, dtype=y.dtype)
    for g in range(n_groups):
        mask = groups == g
        if not mask.any():
            continue
        y_g = y[mask]
        means[g] = y_g.mean()
        stds[g] = y_g.std(unbiased=False)
        p10[g] = torch.quantile(y_g, q_low)
        p90[g] = torch.quantile(y_g, q_high)
    return {"mean": means, "std": stds, "p10": p10, "p90": p90}

'''
有四种计算权重的策略，mse、l1、huber是计算基础误差loss的方式。然后通过weight_group_zscore_exp计算动态矩阵权重weights。
计算方法：将当前样本的吸积率与该星系的平均值比较，如果相当，则权重为1+α，如果远超平均值，则权重会指数级增加
'''
def make_criterion( 
    kind: str,
    delta: float,
    alpha: float = 1.0,
    weight_mode: str = "group_zscore_exp",
    group_stats: dict[str, torch.Tensor] | None = None,
    tau: float = 1.0,
    power: float = 1.0,
):
    if kind == "mse":
        base = nn.MSELoss(reduction="none")
    elif kind == "l1":
        base = nn.L1Loss(reduction="none")
    elif kind == "huber":
        base = nn.HuberLoss(delta=delta, reduction="none")
    else:
        raise ValueError(kind)

    def weight_exp(target: torch.Tensor, groups: torch.Tensor | None = None):
        return 1.0 + alpha * torch.exp(target)
    
    def weight_exp_clamp(target: torch.Tensor, groups: torch.Tensor | None = None):
        return 1.0 + alpha * torch.clamp(torch.exp(target), max=1e-2)  
    
    def weight_linear(target: torch.Tensor, groups: torch.Tensor | None = None):
        return 1.0 + alpha * target
    
    def weight_softplus(target: torch.Tensor, groups: torch.Tensor | None = None):
        return 1.0 + alpha * F.softplus(target)

    def weight_group_zscore_exp(target: torch.Tensor, groups: torch.Tensor | None = None):
        if groups is None or group_stats is None:
            raise ValueError("group_zscore_exp requires groups and group_stats")
        g = groups.view(-1).long().to(device=target.device)
        mean = group_stats["mean"].to(device=target.device)[g].view_as(target)
        std = group_stats["std"].to(device=target.device)[g].view_as(target).clamp_min(1e-6)
        z = (target - mean) / std
        return 1.0 + alpha * torch.exp(z / max(1e-6, tau))

    def weight_group_quantile(target: torch.Tensor, groups: torch.Tensor | None = None):
        if groups is None or group_stats is None:
            raise ValueError("group_quantile requires groups and group_stats")
        g = groups.view(-1).long().to(device=target.device)
        p10 = group_stats["p10"].to(device=target.device)[g].view_as(target)
        p90 = group_stats["p90"].to(device=target.device)[g].view_as(target)
        denom = (p90 - p10).clamp_min(1e-6)
        q = ((target - p10) / denom).clamp(0.0, 1.0)
        return 1.0 + alpha * torch.pow(q, max(1e-6, power))

    def weight_uniform(target: torch.Tensor, groups: torch.Tensor | None = None):
        return torch.ones_like(target)

    if weight_mode == "none":
        weights_fn = weight_uniform
    elif weight_mode == "exp_clamp":
        weights_fn = weight_exp_clamp
    elif weight_mode == "group_zscore_exp":
        weights_fn = weight_group_zscore_exp # 输出动态权重矩阵
    elif weight_mode == "group_quantile":
        weights_fn = weight_group_quantile
    else:
        raise ValueError(weight_mode)

    return WeightedLoss(base_criterion=base, weights_fn=weights_fn)  # 根据基础损失函数loss和动态权重矩阵，算出加权损失


'''
学习率调整：预热warmup+余弦退火。刚开始训练时，学习率从很小的值线性增加到预设的初始学习率，然后沿着一个平滑的余弦曲线慢慢下降。
'''
class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float,
        warmup_start_factor: float,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive for WarmupCosineScheduler")

        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = max(0, min(int(warmup_steps), self.total_steps))
        self.decay_steps = max(1, self.total_steps - self.warmup_steps)
        self.start_factor = float(max(0.0, min(warmup_start_factor, 1.0)))
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        min_lr = max(0.0, float(min_lr))
        self.min_lrs = [min(min_lr, base) for base in self.base_lrs]
        self.step_count = 0
        self._last_lr = []
        self._set_lrs(0)

    def _compute_lr(self, step: int) -> list[float]:
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            warmup_progress = step / max(1, self.warmup_steps)
            factor = self.start_factor + (1.0 - self.start_factor) * warmup_progress
            return [base * factor for base in self.base_lrs]

        progress = (step - self.warmup_steps) / max(1, self.decay_steps)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [min_lr + (base - min_lr) * cosine for base, min_lr in zip(self.base_lrs, self.min_lrs)]

    def _set_lrs(self, step: int) -> None:
        lrs = self._compute_lr(step)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        self._last_lr = lrs

    def step(self) -> None:
        if self.step_count >= self.total_steps:
            self._set_lrs(self.total_steps)
            return
        self.step_count += 1
        self._set_lrs(self.step_count)

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)


'''
标准的一轮训练主循环，整合loader，model，输出prediction，计算加权损失，反向传播，更新权重和学习率
'''
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=None, scheduler=None):
    model.train()
    total, n = 0.0, 0 
    for batch in loader: # Batch field 3 is the cache-declared conditioning scalar (KS SFR for current SFR caches).
        x = batch[0].to(device, non_blocking=True)  # (B, C_in, H, W)
        coord = batch[1].to(device, non_blocking=True)  # (B, 2, H, W)
        conditioning_scalar = batch[2].to(device, non_blocking=True).view(-1)
        y = batch[3].to(device, non_blocking=True).view(-1)  # (B,) 真实SFR值
        groups = batch[4].to(device, non_blocking=True).view(-1)  # (B,)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred = predict_batch(model, x, coord, conditioning_scalar)
                loss = criterion(pred, y, groups) # 根据权重函数计算加权损失值
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # 梯度裁剪，强行限制梯度<grad_clip,防止梯度爆炸
            scaler.step(optimizer) # 根据缩放后的梯度更新模型参数
            scaler.update() # 更新学习率和缩放因子，准备下一步迭代
            if scheduler is not None:
                scheduler.step()
        else:
            pred = predict_batch(model, x, coord, conditioning_scalar)
            loss = criterion(pred, y, groups)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        bs = y.size(0)  # 当前批次的样本数量
        total += loss.detach().item() * bs # 累积加权损失总和，乘以批次大小以获得当前批次的总损失
        n += bs
    return total / max(1, n) # 返回当前轮次的平均加权损失


'''
打印当前dataloader里每一种星系（group）的样本数量，用于验证oversample平衡是否生效
'''
@torch.no_grad()
def print_group_counts(loader, name: str):
    counts = None
    for batch in loader:
        groups = batch[4].view(-1)
        if counts is None:
            n_groups_tensor = batch[5]
            if isinstance(n_groups_tensor, torch.Tensor):
                n_groups = int(n_groups_tensor.flatten()[0].item())
            else:
                n_groups = int(n_groups_tensor)
            counts = [0] * n_groups
        for g in range(len(counts)):
            counts[g] += int((groups == g).sum().item())

    if counts is None:
        print(f"[{name}] loader is empty; no group counts available.")
    else:
        print(f"[{name}] group counts: {counts}")

'''
评估函数，计算每个组的平均损失，并返回worst domain loss（所有组中平均损失最大的那个）和所有组的平均损失列表。
'''
@torch.no_grad()
def eval_epoch(model, loader, criterion, device, writer, epoch):
    model.eval()
    try:
        first_batch = next(iter(loader))
    except StopIteration:
        print(f"[{epoch}] Warning: eval_epoch received an empty loader.")
        return float("nan"), []

    n_groups_tensor = first_batch[5]
    if isinstance(n_groups_tensor, torch.Tensor):
        n_groups = int(n_groups_tensor.flatten()[0].item())
    else:
        n_groups = int(n_groups_tensor)
    del first_batch

    if n_groups <= 0:
        # Fallback if no groups are defined
        total_loss = 0.0
        n_samples = 0
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            coord = batch[1].to(device, non_blocking=True)
            conditioning_scalar = batch[2].to(device, non_blocking=True).view(-1)
            y = batch[3].to(device, non_blocking=True).view(-1)
            pred = predict_batch(model, x, coord, conditioning_scalar)
            loss = criterion(pred, y).mean().item()
            total_loss += loss * y.size(0)
            n_samples += y.size(0)
        avg = total_loss / max(1, n_samples)
        return avg, [avg]

    groups_sum = [0.0] * n_groups
    groups_count = [0] * n_groups

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        coord = batch[1].to(device, non_blocking=True)
        conditioning_scalar = batch[2].to(device, non_blocking=True).view(-1)
        y = batch[3].to(device, non_blocking=True).view(-1)
        groups = batch[4].view(-1).to(device=y.device, dtype=torch.long)

        pred = predict_batch(model, x, coord, conditioning_scalar)

        if isinstance(criterion, WeightedLoss):
            base_loss = criterion.base_criterion(pred, y)  # shape: (B, ...) or (B,)
            weights = criterion.weights_fn(y, groups)  # 同形或可广播
            per_sample_loss = base_loss * weights
        elif isinstance(criterion, nn.HuberLoss):
            per_sample_loss = F.huber_loss(pred, y, reduction="none", delta=getattr(criterion, "delta", 1.0))
        elif isinstance(criterion, nn.MSELoss):
            per_sample_loss = F.mse_loss(pred, y, reduction="none")
        elif isinstance(criterion, nn.L1Loss):
            per_sample_loss = F.l1_loss(pred, y, reduction="none")
        else:
            per_sample_loss = criterion(pred, y)

        # (B,) ensure shape: if multi-dimensional, average over non-batch dimensions first
        if per_sample_loss.ndim > 1:
            per_sample_loss = per_sample_loss.view(per_sample_loss.size(0), -1).mean(dim=1)

        # Cumulative by domain
        for g in range(n_groups): # 累加每个group的损失总和，计算每一种星系的平均损失
            m = groups == g
            if m.any():
                groups_sum[g] += per_sample_loss[m].mean().item() * m.sum().item()
                groups_count[g] += m.sum().item()

    # Domain-by-domain mean and worst domain
    groups_mean = [(groups_sum[d] / groups_count[d]) if groups_count[d] > 0 else float("nan") for d in range(n_groups)]
    valid_means = [v for v in groups_mean if not (v != v)]  # 跳过 NaN

    if not valid_means:
        print(
            f"[{epoch}] Warning: eval_epoch found no valid group means. n_groups={n_groups}, groups_count={groups_count}"
        )
        metric = float("nan")
    else:
        metric = sum(valid_means) / len(valid_means) # 恒星形成率是一个更平缓的宏观物理量，对于它的预测，核心指标改为所有星系组的平均损失更为合理

    # Write to TensorBoard
    for d, v in enumerate(groups_mean):
        if v == v:
            writer.add_scalar(f"val/loss_domain_{d}", v, epoch)
    writer.add_scalar("val/loss_metric", metric, epoch)

    # Return main metric loss and all domain means
    return metric, groups_mean


# =============================================
# 4) Main
# =============================================
def main(args=None):
    if args is None:
        args = parse_args()
    args = _apply_hpo_overrides(args) # 取命令行参数，并允许 HPO（自动化调参）覆盖手写参数。
    set_seed(args.seed) # 固定随机数种子，保证实验的可复现性
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"Using device: {device}")
    # Initialize TensorBoard writer
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S") # 根据当前时间生成一个独一无二的 TensorBoard 文件夹，准备记录即将产生的海量数据。
    clean_exp_name = args.exp_name.rstrip("_")
    log_dir = os.path.join(args.log_dir, f"{clean_exp_name}_{run_stamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Datasets / Loaders
    cache_dir = args.cache_dir
    exp = clean_exp_name
    data_exp = args.data_exp if args.data_exp else exp

    train_augmentor = SFRSpatialAugmentor(p_flip=args.sfr_aug_prob, p_rot=args.sfr_aug_prob) # 加入数据增强Augmentor

    train_ds = GalPTDataset(os.path.join(cache_dir, data_exp + "train.pt"), augmentor=train_augmentor) # 训练集
    val_ds = GalPTDataset(os.path.join(cache_dir, data_exp + "val.pt")) # 验证集
    test_ds = GalPTDataset(os.path.join(cache_dir, data_exp + "test.pt")) # 测试集
    cache_split_metadata = _validate_leave_one_out_cache_contract(
        cache_dir,
        data_exp,
        {"train": train_ds, "val": val_ds, "test": test_ds},
    )

    channel_counts = {int(ds.x.shape[1]) for ds in (train_ds, val_ds, test_ds)}
    if len(channel_counts) != 1:
        raise RuntimeError(f"Train/validation/test caches have inconsistent channel counts: {channel_counts}")
    known_channel_sets = {
        ds.input_channels for ds in (train_ds, val_ds, test_ds) if ds.input_channels
    }
    if len(known_channel_sets) > 1:
        raise RuntimeError(f"Train/validation/test caches have inconsistent channel metadata: {known_channel_sets}")
    target_metadata_present = [ds.target_metadata for ds in (train_ds, val_ds, test_ds) if ds.target_metadata]
    known_target_metadata = {
        json.dumps(ds.target_metadata, sort_keys=True)
        for ds in (train_ds, val_ds, test_ds)
        if ds.target_metadata
    }
    if target_metadata_present and (len(target_metadata_present) != 3 or len(known_target_metadata) != 1):
        raise RuntimeError(
            "Train/validation/test caches are missing target metadata or use inconsistent target semantics. "
            f"Found: {known_target_metadata}"
        )
    if train_ds.target_metadata and train_ds.target_metadata.get("target_quantity") != "instantaneous_sfr":
        raise RuntimeError(
            "This SFR training path requires target_quantity='instantaneous_sfr'; "
            f"got {train_ds.target_metadata.get('target_quantity')!r}."
        )
    conditioning_variables = {
        ds.conditioning_variable for ds in (train_ds, val_ds, test_ds)
    }
    if len(conditioning_variables) != 1:
        raise RuntimeError(
            "Train/validation/test caches use inconsistent conditioning variables: "
            f"{sorted(conditioning_variables)}"
        )
    conditioning_variable = next(iter(conditioning_variables))
    if args.model_type in {"resnet", "resfilm"} and conditioning_variable != "mbh":
        raise RuntimeError(
            f"Model type {args.model_type!r} is a legacy BH-mass-conditioned model, "
            f"but cache {data_exp!r} provides {conditioning_variable!r}."
        )
    if args.model_type == "convnet" and args.use_film and conditioning_variable != "sfr_ks":
        raise RuntimeError(
            "SFRConvNet FiLM requires conditioning_variable='sfr_ks'; "
            f"got {conditioning_variable!r}."
        )
    print(
        "Conditioning: "
        f"variable={conditioning_variable}, "
        f"FiLM={'enabled' if args.use_film else 'disabled'}"
    )
    if train_ds.input_channels:
        print(f"Input channels ({len(train_ds.input_channels)}): {', '.join(train_ds.input_channels)}")
    else:
        print(f"Input channels: {train_ds.x.shape[1]} (legacy cache without channel-name metadata)")
    if train_ds.target_metadata:
        print(f"Target metadata: {train_ds.target_metadata}")
    else:
        print("[Warning] Legacy cache has no target metadata; physical target semantics cannot be verified.")

    group_stats = _compute_group_stats( # 调用_compute_group_stats函数，计算训练集中每个组的吸积率分布统计信息（均值、标准差、10%分位数和90%分位数），这些统计信息将用于动态权重计算
        train_ds.y,
        train_ds.groups,
        int(train_ds.n_groups),
        q_low=args.weight_q_low,
        q_high=args.weight_q_high,
    )

    common = dict(batch_size=args.batch_size, pin_memory=(device.type == "cuda"), drop_last=False)
    dl_extra = {}
    if args.num_workers > 0: # 调用cpu准备数据
        # On Windows, persistent_workers=True is highly recommended to avoid repeated spawning
        pw = args.persistent_workers or (os.name == "nt")
        dl_extra.update(dict(persistent_workers=pw, prefetch_factor=max(1, args.prefetch_factor)))

    train_loader = DataLoader(train_ds, shuffle=True, num_workers=max(0, args.num_workers // 2), **common, **dl_extra)
    val_loader = DataLoader(val_ds, shuffle=False, num_workers=max(0, args.num_workers // 2), **common, **dl_extra)
    test_loader = DataLoader(test_ds, shuffle=False, num_workers=max(0, args.num_workers // 2), **common, **dl_extra)

    if args.print_group_counts:
        print_group_counts(train_loader, "train")
        print_group_counts(val_loader, "val")
        print_group_counts(test_loader, "test")

    # Model 根据指令调用model，发射到GPU上
    c_in = train_ds.x.shape[1]
    if args.model_type == "hybrid":
        raise ValueError("Hybrid model type is not implemented in this script.")
    elif args.model_type == "transformer":
        model = AccretionTransformer(
            c_in=c_in,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            pos_num_bands=args.pos_num_bands,
            pos_max_freq=args.pos_max_freq,
            p_drop=args.p_drop,
        ).to(device)
    elif args.model_type == "resnet":
        model = MACNetRes_mbh(
            in_channels=c_in,
        ).to(device)
    elif args.model_type == "resfilm":
        model = MACNetFiLM(
            in_channels=c_in,
        ).to(device)
    elif args.model_type == "convnet":
        stage_depths = _parse_stage_depths(args.stage_depths)
        model = SFRConvNet(
            c_in=c_in,
            base_ch=args.base_ch,
            stage_depths=stage_depths,
            pos_num_bands=args.pos_num_bands,
            pos_max_freq=args.pos_max_freq,
            p_drop=args.p_drop,
            se_reduction=args.se_reduction,
            use_film=args.use_film, # 传入命令行控制的 FiLM 参数
        ).to(device)
    model.conditioning_variable = conditioning_variable

    try:
        model.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            # Use a tiny temp loader with 0 workers to fetch a sample for the graph
            temp_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
            sample_x, sample_coord, sample_conditioning_scalar, _, _, _ = next(iter(temp_loader))
            if isinstance(model, MBH_CONDITIONED_MODELS):
                writer.add_graph(model, (sample_x.to(device), sample_conditioning_scalar.view(-1).to(device)))
            elif isinstance(model, SFRConvNet):
                writer.add_graph(
                    model,
                    (
                        sample_x.to(device),
                        sample_coord[:, 0:1].to(device),
                        sample_coord[:, 1:2].to(device),
                        sample_conditioning_scalar.view(-1).to(device),
                    ),
                )
            else:
                writer.add_graph(
                    model,
                    (
                        sample_x.to(device),
                        sample_coord[:, 0:1].to(device),
                        sample_coord[:, 1:2].to(device),
                        sample_conditioning_scalar.view(-1).to(device),
                    ),
                )
    except Exception as exc:
        print(f"Warning: could not write model graph to TensorBoard: {exc}")

    # Opt / Sched / Loss
    criterion = make_criterion(
        args.loss,
        args.huber_delta,
        alpha=args.weight_alpha,
        weight_mode=args.weight_mode,
        group_stats=group_stats,
        tau=args.weight_tau,
        power=args.weight_power,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # AdamW优化器，结合了Adam的自适应学习率和权重衰减weight_decay
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * max(0, args.epochs)
    scheduler = None
    if total_steps > 0: # warmup+余弦退火
        warmup_steps = max(0, args.lr_warmup_steps)
        warmup_ratio = max(0.0, min(1.0, args.lr_warmup_ratio))
        if warmup_steps == 0 and warmup_ratio > 0.0:
            warmup_steps = int(total_steps * warmup_ratio)
        warmup_steps = min(warmup_steps, total_steps)
        try:
            scheduler = WarmupCosineScheduler(
                optimizer=optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr=args.lr_min,
                warmup_start_factor=args.lr_warmup_start_factor,
            )
        except ValueError as exc:
            print(f"[Warning] Could not initialize scheduler: {exc}. Falling back to constant LR.")
            scheduler = None

    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Save dirs
    if not args.no_save:
        save_dir = os.path.join("Results", f"{exp}_{run_stamp}")
        os.makedirs(save_dir, exist_ok=True)
        hyperparams_path = os.path.join(save_dir, "hyperparameters.json")
        hyperparams = vars(args).copy()
        hyperparams.update(
            {
                "resolved_device": str(device),
                "log_dir": log_dir,
                "save_dir": save_dir,
                "run_stamp": run_stamp,
                "n_input_channels": int(train_ds.x.shape[1]),
                "input_channels": list(train_ds.input_channels),
                "excluded_channels": list(train_ds.excluded_channels),
                "target_metadata": train_ds.target_metadata,
                "conditioning_variable": conditioning_variable,
                "data_split_metadata": cache_split_metadata,
                **train_ds.target_metadata,
            }
        )
        with open(hyperparams_path, "w", encoding="utf-8") as f:
            json.dump(hyperparams, f, indent=2, sort_keys=True)
        best_path = os.path.join(save_dir, f"{exp}_best_model.pth")
    else:
        save_dir = None
        best_path = None

    # Train
    best_val = float("inf")
    best_model_state = None  # To store best model in memory if no_save is True

    patience = max(0, args.patience)
    no_improve = 0
    print("Starting training") 

    for epoch in range(1, args.epochs + 1): # 多轮训练
        tr_loss = train_one_epoch( # 一轮训练主循环
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler if use_amp else None,
            grad_clip=(args.grad_clip if args.grad_clip > 0 else None),
            scheduler=scheduler,
        )
        val_metric, group_means = eval_epoch(model, val_loader, criterion, device, writer, epoch) # 在验证集上测试，返回综合指标与各星系均值

        valid_vals = [v for v in group_means if v == v]
        mean_val = sum(valid_vals) / len(valid_vals) if valid_vals else float("nan")

        print(
            f"Epoch {epoch:03d} | train {tr_loss:.6f} | val {mean_val:.6f} | metric {val_metric:.6f} | lr {optimizer.param_groups[0]['lr']:.3e}"
        )

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("val/loss_mean_over_domains", mean_val, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if val_metric < best_val - 1e-8: # 记录最好成绩，更新权重
            best_val = val_metric
            no_improve = 0
            if args.no_save:
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "val_loss": best_val,
                        "mean_val": mean_val,
                        "epoch": epoch,
                        "n_input_channels": int(train_ds.x.shape[1]),
                        "input_channels": list(train_ds.input_channels),
                        "excluded_channels": list(train_ds.excluded_channels),
                        "target_metadata": train_ds.target_metadata,
                        "conditioning_variable": conditioning_variable,
                        "data_split_metadata": cache_split_metadata,
                    },
                    best_path,
                )
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:  # 如果连续训练patience轮都没有提升，说明很可能过拟合了，触发早停机制
                print(f"Early stopping at epoch {epoch}")
                break

    has_best = False
    if args.no_save: # 训练结束后，加载最好的模型，去测试集验证泛化能力
        if best_val != float("inf") and best_model_state is not None:
            has_best = True
    else:
        if best_val != float("inf") and os.path.exists(best_path):
            has_best = True

    if not has_best:
        print("No improvement was found during training. Best model not saved (or not found).")
        return float("inf")

    print(f"Best val metric loss: {best_val:.6f}. Loading best and testing.")

    if args.no_save:
        model.load_state_dict(best_model_state)
    else:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    test_metric, group_means = eval_epoch(model, test_loader, criterion, device, writer, epoch=0) # 在测试集中评估训练出来的最好的模型。
    print(f"Test metric loss: {test_metric:.6f}, per-domain: {group_means}")
    if not args.no_save:
        print(f"Best model saved to: {best_path}")

    return best_val


if __name__ == "__main__":
    main()
