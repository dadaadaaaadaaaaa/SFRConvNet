"""二维逐格 signed-closure 残差 U-Net；与原 models.py 的标量网络独立。

输入是已标准化的 LR 气体场及原坐标，输出是 (N,1,theta,r) 的变换域
closure。物理 baseline、反变换及损失放在 cell_sfr_objective.py 中。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn import functional as F

from Experiment.models import ConvFiLMBlock, FourierPositionalEncoding2D


class SFRUNet(nn.Module):
    """两次下采样的残差编码器—解码器，使用 Concat 连接对应尺度。

    默认通道 32/64/128，E1/E2/bottleneck/D2/D1 各两个残差块。
    固定 R_h 关闭 FiLM。此类不接收 baseline、标签或网格体积作为额外条件。
    """

    def __init__(self, c_in: int, base_ch: int = 32,
                 encoder_depths=(2, 2), bottleneck_depth: int = 2,
                 decoder_depths=(2, 2), pos_num_bands: int = 4,
                 pos_max_freq: float = 4.0, p_drop: float = 0.0,
                 se_reduction: int = 4):
        super().__init__()
        for name, value in (("c_in", c_in), ("base_ch", base_ch),
                            ("bottleneck_depth", bottleneck_depth),
                            ("pos_num_bands", pos_num_bands), ("se_reduction", se_reduction)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数。")
        for name, depths in (("encoder_depths", encoder_depths), ("decoder_depths", decoder_depths)):
            if len(depths) != 2 or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in depths):
                raise ValueError(f"{name} 必须包含两个正整数。")
        if not math.isfinite(pos_max_freq) or pos_max_freq < 1:
            raise ValueError("pos_max_freq 必须有限且不小于 1。")
        if not math.isfinite(p_drop) or not 0 <= p_drop < 1:
            raise ValueError("p_drop 必须在 [0,1) 内。")
        self.c_in = c_in
        self.use_film = False
        self.prediction_mode = "signed_closure"
        self.config = dict(c_in=c_in, base_ch=base_ch, encoder_depths=list(encoder_depths),
                           bottleneck_depth=bottleneck_depth, decoder_depths=list(decoder_depths),
                           pos_num_bands=pos_num_bands, pos_max_freq=float(pos_max_freq),
                           p_drop=float(p_drop), se_reduction=se_reduction)
        self.posenc = FourierPositionalEncoding2D(pos_num_bands, pos_max_freq,
                                                 include_input=True, theta_pi_periodic=False)
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        def stage(ci, co, depth):
            return nn.Sequential(*[
                ConvFiLMBlock(ci if index == 0 else co, co, stride=1,
                              p_drop=p_drop, se_reduction=se_reduction)
                for index in range(depth)
            ])

        self.stem = nn.Sequential(nn.Conv2d(c_in + self.posenc.out_dim, c1, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(c1), nn.ReLU())
        self.encoder1 = stage(c1, c1, encoder_depths[0])
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False)
        self.encoder2 = stage(c2, c2, encoder_depths[1])
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False)
        self.bottleneck = stage(c3, c3, bottleneck_depth)
        self.up2_projection = nn.Conv2d(c3, c2, 1, bias=False)
        self.decoder2 = stage(c2 * 2, c2, decoder_depths[0])
        self.up1_projection = nn.Conv2d(c2, c1, 1, bias=False)
        self.decoder1 = stage(c1 * 2, c1, decoder_depths[1])
        self.head = nn.Conv2d(c1, 1, 1)
        # 小幅初始 closure 接近物理 baseline，非全零权重仍允许首步梯度进入主干。
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, r, theta, conditioning_scalar=None):
        """返回逐格 z_hat；只允许 inactive 的全零条件占位，兼容旧 predict_batch 调用形状。"""
        if x.ndim != 4 or x.shape[1] != self.c_in or x.shape[0] < 1 or min(x.shape[-2:]) < 4:
            raise ValueError("x 必须为非空 (N,c_in,H,W)，且 H/W 至少为 4。")
        expected = (x.shape[0], 1, *x.shape[-2:])
        if r.shape != expected or theta.shape != expected:
            raise ValueError("r/theta 必须为与 x 对齐的 (N,1,H,W)。")
        if conditioning_scalar is not None:
            if conditioning_scalar.numel() != x.shape[0] or torch.count_nonzero(conditioning_scalar).item():
                raise ValueError("SFRUNet 固定 R_h 禁用 FiLM，不接受活动条件。")
        if self.training and x.shape[0] * ((x.shape[-2] + 3) // 4) * ((x.shape[-1] + 3) // 4) <= 1:
            raise ValueError("当前 batch/空间尺寸不足以计算瓶颈 BatchNorm 统计。")
        pos = self.posenc(r, theta, normalize=True)
        e1 = self.encoder1(self.stem(torch.cat([x, pos], dim=1)))
        e2 = self.encoder2(self.down1(e1))
        h = self.bottleneck(self.down2(e2))
        # 显式匹配目标尺寸，兼容奇数网格；插值的是隐藏特征，不是物理场。
        u2 = self.up2_projection(F.interpolate(h, size=e2.shape[-2:], mode="bilinear", align_corners=False))
        d2 = self.decoder2(torch.cat([u2, e2], dim=1))
        u1 = self.up1_projection(F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False))
        d1 = self.decoder1(torch.cat([u1, e1], dim=1))
        return self.head(d1)

    def model_contract(self):
        """供 checkpoint 保存；不是原标量模型配置的替代品。"""
        return {"architecture": "SFRUNet", "version": 1, "config": dict(self.config),
                "output": "signed_asinh_cell_closure", "output_axes": ["sample", "channel", "theta", "r"],
                "conditioning": "none", "spatial_output": True,
                "baseline_is_model_input": False}


def main():
    """只做合成前向/反向检查；不训练正式数据、不写 checkpoint。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--c-in", type=int, default=12)
    parser.add_argument("--shape", type=int, nargs=2, default=(12, 20))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not args.smoke_test:
        parser.print_help()
        return
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = SFRUNet(args.c_in).to(args.device)
    h, w = args.shape
    x = torch.randn(2, args.c_in, h, w, device=args.device)
    r = torch.logspace(-1, 2, w, device=args.device)[None, None, None, :].expand(2, 1, h, w)
    theta = torch.linspace(.1, 3., h, device=args.device)[None, None, :, None].expand_as(r)
    output = model(x, r, theta)
    output.square().mean().backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("存在缺失或非有限梯度。")
    print(f"SFRUNet parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"output={tuple(output.shape)}; forward/backward PASS; device={args.device}")


if __name__ == "__main__":
    main()
