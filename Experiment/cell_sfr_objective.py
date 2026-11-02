"""逐格 signed closure 的反变换、训练集尺度与物理 SFR 损失；不修改标量目标。"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def _map(values, name):
    """只接受 NHW 或 N1HW，禁止隐式广播到其他样本/网格。"""
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if values.ndim != 4 or values.shape[1] != 1 or any(d < 1 for d in values.shape):
        raise ValueError(f"{name} 必须是非空 (N,H,W) 或 (N,1,H,W)。")
    if not values.is_floating_point():
        raise TypeError(f"{name} 必须是浮点 tensor。")
    return values


@torch.no_grad()
def fit_cell_loss_scales(target, baseline, *, split, provenance=None):
    """仅用训练集估计 RMS 损失尺度；closure 取非零绝对值中位数。

    保留训练 cache 的过采样重复项。全零量的尺度使用 1 code SFR，并明确记录。
    此函数不判断调用方提供的样本身份；生产入口应使用 CellSFRDataset.fit_scales。
    """
    if split != "train":
        raise ValueError("尺度只能从 train split 估计。")
    y = _map(target.detach().cpu().double(), "target")
    b = _map(baseline.detach().cpu().double(), "baseline")
    if y.shape != b.shape:
        raise ValueError("训练 Y/B shape 不一致。")
    if not torch.isfinite(y).all() or not torch.isfinite(b).all() or (y < 0).any() or (b < 0).any():
        raise ValueError("训练 Y/B 必须有限且非负。")
    absolute = (y - b).abs().flatten()
    positive = absolute[absolute > 0]
    zero_fallbacks = []
    if positive.numel():
        closure = float(torch.quantile(positive, 0.5))
    else:
        closure = 1.0
        zero_fallbacks.append("closure_scale")

    def rms(values, name):
        # 先缩放后平方，避免大幅度训练值在 float64 中平方溢出。
        peak = float(values.abs().max())
        if peak == 0:
            zero_fallbacks.append(name)
            return 1.0
        return peak * float(torch.mean((values / peak).square()).sqrt())

    scales = {"closure_scale": closure, "cell_scale": rms(y, "cell_scale"),
              "total_scale": rms(y.sum(dim=(1, 2, 3)), "total_scale")}
    if not all(math.isfinite(v) and v > 0 for v in scales.values()):
        raise ValueError("无法得到有限的正训练尺度。")
    return {"schema_name": "macnet_cell_loss_scales", "schema_version": 1,
            **scales, "source_split": "train", "units": "MACER_mass_code/Gyr",
            "n_samples": len(y), "spatial_shape": list(y.shape[-2:]),
            "closure_estimator": "median_abs_nonzero", "loss_estimator": "rms_target",
            "zero_scale_fallback_code_sfr": 1.0, "zero_fallbacks": zero_fallbacks,
            "sampling": "original_train_cache_including_repeated_uids",
            "provenance": dict(provenance or {})}


def compose_cell_sfr(transformed_closure, baseline, *, closure_scale):
    """可微生成逐格 closure/SFR 及总量，不裁剪负值、不再乘体积。"""
    if not math.isfinite(closure_scale) or closure_scale <= 0:
        raise ValueError("closure_scale 必须为有限正数。")
    z, b = _map(transformed_closure, "prediction"), _map(baseline, "baseline")
    if z.shape != b.shape or z.device != b.device:
        raise ValueError("prediction/baseline 的 shape 和 device 必须一致。")
    # 至少 FP32 执行 sinh；float64 标签时保留精度以减小 B+C 的抵消误差。
    dtype = torch.float64 if torch.float64 in (z.dtype, b.dtype) else torch.float32
    c = z.to(dtype).sinh() * closure_scale
    final = b.to(dtype) + c
    return {"closure_transformed": z, "closure_map_physical": c,
            "sfr_map_physical": final, "sfr_total_physical": final.sum(dim=(1, 2, 3))}


class CellSFRLoss(nn.Module):
    """逐格 MSE + 总量 MSE + 逐格非负平方 hinge，全部在归一化物理空间。

    输出参数化使用 signed-asinh，不对 z_hat 与 z_true 再添加重复主损失。
    缺省所有格子（含零标签）等权，先空间平均再 snapshot 平均。
    """

    def __init__(self, *, closure_scale, cell_scale, total_scale,
                 total_weight=1.0, negative_weight=1.0, check_finite=True):
        super().__init__()
        for name, value in (("closure_scale", closure_scale), ("cell_scale", cell_scale),
                            ("total_scale", total_scale)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数。")
        for name, value in (("total_weight", total_weight), ("negative_weight", negative_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须有限且非负。")
        self.register_buffer("scales", torch.tensor([closure_scale, cell_scale, total_scale], dtype=torch.float64))
        self.register_buffer("weights", torch.tensor([total_weight, negative_weight], dtype=torch.float64))
        self.check_finite = bool(check_finite)

    @classmethod
    def from_scale_contract(cls, contract, **kwargs):
        """从已保存的训练尺度构建，验证/测试不得重新拟合。"""
        if (contract.get("schema_name") != "macnet_cell_loss_scales" or contract.get("schema_version") != 1
                or contract.get("source_split") != "train" or contract.get("units") != "MACER_mass_code/Gyr"):
            raise ValueError("逐格训练尺度契约不匹配。")
        return cls(**{key: contract[key] for key in ("closure_scale", "cell_scale", "total_scale")}, **kwargs)

    def forward(self, transformed_closure, baseline, target):
        z, b, y = (_map(t, name) for t, name in
                   ((transformed_closure, "prediction"), (baseline, "baseline"), (target, "target")))
        if z.shape != b.shape or b.shape != y.shape or z.device != b.device or b.device != y.device:
            raise ValueError("prediction/baseline/target 的 shape 和 device 必须一致，不能广播。")
        dtype = torch.float64 if torch.float64 in (z.dtype, b.dtype, y.dtype) else torch.float32
        scales, weights = self.scales.to(device=z.device, dtype=dtype), self.weights.to(device=z.device, dtype=dtype)
        closure = z.to(dtype).sinh() * scales[0]
        predicted = b.to(dtype) + closure
        error = predicted - y.to(dtype)
        per_cell = (error / scales[1]).square().mean(dim=(1, 2, 3))
        per_total = (error.sum(dim=(1, 2, 3)) / scales[2]).square()
        per_negative = (F.relu(-predicted) / scales[1]).square().mean(dim=(1, 2, 3))
        per_sample = per_cell + weights[0] * per_total + weights[1] * per_negative
        loss = per_sample.mean()
        if self.check_finite:
            valid = torch.stack([torch.isfinite(loss), torch.isfinite(b).all(), torch.isfinite(y).all(),
                                 (b >= 0).all(), (y >= 0).all()]).all()
            if not bool(valid):
                raise FloatingPointError("逐格损失非有限或 Y/B 非法；不以 clamp/nan_to_num 隐藏问题。")
        return loss, {"cell_loss": per_cell.mean(), "total_sfr_loss": per_total.mean(),
                      "negative_penalty": per_negative.mean(), "per_sample_loss": per_sample,
                      "closure_map_physical": closure, "sfr_map_physical": predicted,
                      "sfr_total_physical": predicted.sum(dim=(1, 2, 3))}

    def loss_contract(self):
        """保存权重、尺度、归约和单位；它们不能从旧标量 objective 静默继承。"""
        c, cell, total = self.scales.detach().cpu().tolist()
        w_total, w_neg = self.weights.detach().cpu().tolist()
        return {"schema_name": "macnet_cell_sfr_loss", "schema_version": 1,
                "objective": "signed_closure", "loss_space": "physical_sfr",
                "closure_transform": "signed_asinh", "closure_scale": c,
                "cell_scale": cell, "total_scale": total, "total_weight": w_total,
                "negative_weight": w_neg, "primary": "mse", "reduction": "cell_mean_then_snapshot_mean",
                "include_zero_cells": True, "multiply_volume": False, "output_clamp": False,
                "units": "MACER_mass_code/Gyr"}
