"""二维逐格 hurdle 输出、物理损失及验证集阈值。"""

import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

from Experiment.cell_sfr_objective import _map


class HurdleCellSFRLoss(nn.Module):
    def __init__(self, scales, *, mode, amount_scale, positive_weight=1., total_weight=1.):
        super().__init__()
        if mode not in {"hurdle_closure", "hurdle_direct"} or scales["source_split"] != "train":
            raise ValueError("需要已知 hurdle 模式和训练集尺度。")
        self.mode = mode
        self.amount_scale = float(amount_scale)
        self.closure_scale = float(scales["closure_scale"])
        self.cell_scale = float(scales["cell_scale"])
        self.total_scale = float(scales["total_scale"])
        self.positive_weight = float(positive_weight)
        self.total_weight = float(total_weight)
        for value in (self.amount_scale, self.closure_scale, self.cell_scale, self.total_scale, self.positive_weight):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("尺度与正格权重必须有限且为正。")
        if not math.isfinite(self.total_weight) or self.total_weight < 0:
            raise ValueError("总量权重必须有限且非负。")

    def compose(self, outputs, baseline):
        """同一物理组合用于训练及无标签推理；baseline 只在组合处使用。"""
        raw = _map(outputs["amount_raw"], "amount_raw").double()
        logits = _map(outputs["occurrence_logits"], "occurrence_logits").double()
        b = _map(baseline, "baseline").double()
        if raw.shape != b.shape or logits.shape != b.shape or len({raw.device, logits.device, b.device}) != 1:
            raise ValueError("两头输出与 baseline 必须逐格对齐。")
        if self.mode == "hurdle_closure":
            q = b + self.closure_scale * raw.sinh()
            raw_positive = q / self.amount_scale
        else:
            raw_positive = raw
        # Softplus 极负尾部可能在有限精度下舍入到零；只对正值表示取机器下限。
        amount = (self.amount_scale * F.softplus(raw_positive)).clamp_min(torch.finfo(raw.dtype).tiny)
        p = logits.sigmoid()
        if not (torch.isfinite(raw_positive).all() and torch.isfinite(amount).all()
                and torch.isfinite(logits).all() and torch.isfinite(b).all() and (b >= 0).all()):
            raise FloatingPointError("hurdle 物理组合含非有限值或非法 baseline。")
        return {"probability_map": p, "amount_map": amount, "sfr_map_physical": p * amount}

    def forward(self, outputs, baseline, target):
        maps = self.compose(outputs, baseline)
        y = _map(target, "target").double()
        if y.shape != maps["amount_map"].shape or y.device != maps["amount_map"].device:
            raise ValueError("hurdle 标签与输出必须逐格对齐。")
        if not torch.isfinite(y).all() or (y < 0).any():
            raise ValueError("逐格 Y 必须有限且非负。")
        active = y > 0
        count = active.sum()
        occurrence = F.binary_cross_entropy_with_logits(_map(outputs["occurrence_logits"], "logits").double(),
                                                        active.double())
        positive_sum = ((maps["amount_map"][active] - y[active]) / self.cell_scale).square().sum()
        positive = positive_sum / count.clamp_min(1)
        total = ((maps["sfr_map_physical"] - y).sum((1, 2, 3)) / self.total_scale).square().mean()
        loss = occurrence + self.positive_weight * positive + self.total_weight * total
        if not torch.isfinite(loss):
            raise FloatingPointError("hurdle 损失非有限。")
        return loss, {**maps, "occurrence_loss": occurrence, "positive_loss_sum": positive_sum,
                      "positive_count": count, "total_sfr_loss": total}

    def loss_contract(self):
        return {"schema_name": "macnet_hurdle_cell_loss", "schema_version": 1, "objective": self.mode,
                "label": "Y>0", "units": "MACER_mass_code/Gyr", "multiply_volume": False,
                "amount_scale": self.amount_scale, "amount_scale_source": "train_median_positive_Y_or_1",
                "closure_scale": self.closure_scale, "cell_scale": self.cell_scale, "total_scale": self.total_scale,
                "amount_definition": ("s_pos*softplus((B+s_C*sinh(z))/s_pos)" if self.mode == "hurdle_closure"
                                      else "s_pos*softplus(z)"),
                "mean_definition": "p*amount", "hard_definition": "where(p>=threshold,amount,0)",
                "occurrence_weight": 1., "positive_weight": self.positive_weight, "total_weight": self.total_weight,
                "reduction": ["mean_all_cells_BCE", "mean_true_positive_cells_MSE", "mean_snapshot_sum_MSE"]}


def calibrate_threshold(scores, target, target_fpr, *, split):
    """仅在验证集真零格上选阈值；平分时保守，门使用 >=，并保留实际误报率。"""
    if split != "val" or not 0 <= target_fpr <= 1:
        raise ValueError("阈值必须用验证集和合法目标误报率确定。")
    if scores.shape != target.shape or not torch.isfinite(scores).all():
        raise ValueError("校准分数必须有限且与标签对齐。")
    zeros = scores[target == 0].detach().cpu().double().numpy()
    if zeros.size == 0:
        raise ValueError("验证集没有真零格，无法校准误报率。")
    budget = round(target_fpr * zeros.size, 10)
    k = math.floor(budget)
    if k == zeros.size:
        threshold = float(scores.min())
    else:
        boundary = np.partition(zeros, zeros.size - k - 1)[zeros.size - k - 1]
        threshold = float(np.nextafter(boundary, np.inf))
    return {"threshold": threshold, "target_fpr": float(target_fpr), "actual_fpr": float((zeros >= threshold).mean()),
            "zero_cells": int(zeros.size), "source_split": split, "comparison": ">="}


def occurrence_metrics(probability, target):
    p = probability.detach().cpu().numpy().ravel()
    active = target.detach().cpu().numpy().ravel() > 0
    return {"average_precision": float(average_precision_score(active, p)) if active.any() else None,
            "brier": float(np.mean((p - active)**2)), "positive_fraction": float(active.mean())}
