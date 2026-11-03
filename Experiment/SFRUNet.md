# SFRUNet：二维逐格 signed-closure 模型

此模块实现方案 C 的**逐格预测与逐格监督**，独立于 `models.py` 的标量 SFRConvNet。
网络、损失和数据适配器已经接通；现有 `run_train.py/run_optuna.py` 仍是标量入口。
2026-11-03 新增独立 [run_train_sfr_unet.py](run_train_sfr_unet.md) 和
[run_optuna_sfr_unet.py](run_optuna_sfr_unet.md)，支持 Exp1/Exp2。旧分析 notebook 尚未适配空间结果。

## 1. 文件职责与结构

- `SFRUNet.py`：网络及合成前向/反向检查。
- `cell_sfr_objective.py`：物理组合、训练集尺度估计、逐格与总量损失。
- `cell_sfr_dataset.py`：联合读取原 scalar cache 与逐格 sidecar，检查身份及监督契约。
- `tests/test_sfr_unet.py`：尺寸、梯度、物理损失、尺度与数据接口测试。

默认结构（输入 12×20 时）：

```text
已标准化气体场 x + Fourier(r,theta)
    ↓ stem 3×3 Conv → BN → ReLU
E1：32 通道，12×20，2 个残差块 ──────────────────────┐
    ↓ 3×3 stride=2 Conv                              │
E2：64 通道，6×10，2 个残差块 ──────────┐            │
    ↓ 3×3 stride=2 Conv                  │            │
瓶颈：128 通道，3×5，2 个残差块          │            │
    ↓ 双线性上采样至 6×10 → 1×1 Conv至64│            │
    Concat(E2) → 128通道 → 2残差块至64 ←┘            │
    ↓ 双线性上采样至 12×20 → 1×1 Conv至32            │
    Concat(E1) → 64通道 → 2残差块至32 ←───────────────┘
    ↓ 1×1 Conv，线性输出
z_hat：(N,1,N_theta,N_r)
```

Concat 沿通道维拼接，跳跃分支保留梯度；不是相加。残差块内部另有相加跳跃连接。
复用 `ConvFiLMBlock` 的 BN→ReLU→Conv→BN→ReLU→Dropout→Conv→SE 结构，固定 R_h 禁用 FiLM。
SE 的隐藏特征全局汇聚保留，用于通道权重；网络没有将空间输出压成标量的池化回归头。
上采样仅操作隐藏特征，输出仍对应原 coarse cells。使用显式目标尺寸，支持奇数 H/W。
这并不表示已验证跨分辨率泛化；改变物理分辨率仍须重建 coarse 数据及 cache。

12 个物理输入通道、4 个 Fourier bands 时默认 **1,156,097** 个可训练参数；
`bottleneck_depth=1` 时为 **852,321**。默认每阶段两个残差块。

## 2. 模型参数

| 参数 | 默认 | 含义与可选范围 |
|---|---|---|
| `c_in` | 必填 | cache 实际物理通道数，正整数，不能靠实验名称推断 |
| `base_ch` | 32 | 三尺度通道为 c、2c、4c，正整数 |
| `encoder_depths` | `(2,2)` | E1/E2 残差块数，各为正整数 |
| `bottleneck_depth` | 2 | 瓶颈残差块数，1 可作容量消融 |
| `decoder_depths` | `(2,2)` | D2/D1 残差块数，各为正整数 |
| `pos_num_bands` | 4 | Fourier 频带数，正整数 |
| `pos_max_freq` | 4.0 | 最高频率，有限且 ≥1；对照实验应与标量网络统一 |
| `p_drop` | 0.0 | 残差块 Dropout2D，范围 `[0,1)` |
| `se_reduction` | 4 | SE 压缩比，正整数，隐藏层至少 8 通道 |

`forward(x,r,theta)` 中 x 为 NCHW，r/theta 为 N1HW；H/W 至少 4。
训练时瓶颈的 batch×空间格数必须大于 1，以支持 BatchNorm。
`conditioning_scalar` 仅接受空值或全零占位；不允许活动质量条件。
输入须有限，r 为正；适配器已检查。输出无 ReLU/Softplus，可正可负。
头部权重小幅初始化，使初始预测接近 baseline，但不是严格等于 baseline。

## 3. 物理输出与损失

每格 Y 是 Data-Set-11 保守归约后的**已积分 SFR 贡献**，B 是该粗格的 MACER-LR baseline。
单位 `MACER_mass_code/Gyr`；乘 0.025 转为 Msun/yr。以下求和均不再乘体积。

\[
\widehat C_{ni}=s_C\sinh(z_{ni}),\quad
\widehat Y_{ni}=B_{ni}+\widehat C_{ni},\quad
\widehat Y_{n,\mathrm{total}}=\sum_i\widehat Y_{ni}.
\]

令误差 \(e_{ni}=\widehat Y_{ni}-Y_{ni}\)，N 为 snapshot 数，K 为每帧格数：

\[
L_\mathrm{cell}=\frac1{NK}\sum_{n,i}(e_{ni}/s_\mathrm{cell})^2,
\qquad
L_\mathrm{total}=\frac1N\sum_n\left(\frac{\sum_i e_{ni}}{s_\mathrm{total}}\right)^2,
\]
\[
L_-=\frac1{NK}\sum_{n,i}\left(\frac{\max(-\widehat Y_{ni},0)}{s_\mathrm{cell}}\right)^2,
\qquad L=L_\mathrm{cell}+\lambda_\mathrm{total}L_\mathrm{total}+\lambda_-L_-.
\]

`CellSFRLoss` 默认 `total_weight=1.0, negative_weight=1.0, check_finite=True`。
权重可设为任意有限非负数；例如 total_weight=0 可检验总量约束的作用。
零标签参与监督。第一版不加 burst 权重、不做空间平滑正则、不再重复添加 closure MSE。
逐格项区分总量相同但空间分布不同的预测；总量项防止同号局部误差积累。
非负惩罚作用于最终 SFR，而不是 signed closure；它是软约束，不保证严格非负。
不 clamp 输出；非有限损失直接报错，不静默跳过。

损失在归一化的**物理 SFR 空间**计算，不是 z 空间 MSE。
signed-asinh 在这里是输出参数化，不能据此认为损失压低了高 SFR 的影响。
相同绝对物理误差使用相同平方惩罚；高 SFR 样本可能更主导梯度，需在正式验证中分段评估。

三个尺度只从训练集拟合，验证/测试固定复用：

- `closure_scale=s_C`：训练集非零 `abs(Y-B)` 的中位数，控制输出参数化；
- `cell_scale`：训练集所有 Y 格子的 RMS；
- `total_scale`：训练集每帧总 Y 的 RMS。

训练 cache 中过采样重复项保留。某一量全零时对应尺度回退为 1 code SFR，并写入契约。
不能复用旧标量 closure scale。`fit_scales()` 只接受声明为 train 的数据对象；
调用方仍需保证文件确实由正确的 train split 生成，不能通过改名绕过数据划分。

地图标签默认 float64；网络通常 float32，sinh/物理组合至少 FP32，遇到 float64 标签保留 FP64。
这是减小 B+C 抵消误差的处理；GPU 性能和混合精度尚未实测。

## 4. 原 cache 与逐格监督的适配

先按 [build_cell_supervision_cache.md](build_cell_supervision_cache.md) 为每个 split 生成 sidecar。
原 scalar cache 保留原输入、标准化及 split；sidecar 仅保存物理地图与契约。

```python
from torch.utils.data import DataLoader
from Experiment.SFRUNet import SFRUNet
from Experiment.cell_sfr_dataset import CellSFRDataset, validate_cell_splits
from Experiment.cell_sfr_objective import CellSFRLoss

# 替换路径；只加载自己生成的可信 cache。
train = CellSFRDataset(".cache/EXP_train.pt", ".cache/CELL_train_cells.pt", split="train")
val = CellSFRDataset(".cache/EXP_val.pt", ".cache/CELL_val_cells.pt", split="val")
validate_cell_splits(train=train, val=val)  # 正式使用时也加入 test/out
scales = train.fit_scales()
model = SFRUNet(c_in=train.base.x.shape[1])
objective = CellSFRLoss.from_scale_contract(scales)
batch = next(iter(DataLoader(train, batch_size=8, shuffle=True)))
coord = batch["coord"]
z = model(batch["x"], coord[:, :1], coord[:, 1:2])
loss, prediction = objective(z, batch["baseline_map"], batch["target_map"])
loss.backward()
```

返回字典的 `sfr_map_physical` 为 N1HW，`sfr_total_physical` 为 N。
独立推断可以使用 `compose_cell_sfr(z, baseline, closure_scale=scales['closure_scale'])`。
上述代码仅演示一次前向/反向；正式训练使用独立 SFRUNet 入口。

适配器验证源 cache SHA256、UID/parent 顺序、产品契约摘要、各地图摘要、Y=B+C、
地图与原标量求和一致、无监督量输入、固定 R_h、相同坐标 shape。
支持带完整物理 Y/B/C 的原 Direct 或 Closure cache；不会改变原标量目标。
`validate_cell_splits` 检查 split 间 sample/parent UID 隔离及通道/形状一致。
过采样仅允许在同一 split 内重复；torus 禁止作为 train/val。
当前仅支持默认 data_compatible baseline 的二维 spatial_block 产品。

## 5. 保存与后续训练

新 checkpoint 至少保存 `model.state_dict()`、`model.model_contract()`、
`objective.state_dict()`、`objective.loss_contract()`、训练 scales 契约、
源 scalar/sidecar SHA256、split manifest、归一化契约、种子与 optimizer/scheduler 状态。
恢复模型时 `SFRUNet(**model_contract['config'])` 后加载 state_dict；
验证/测试复用训练 scales，禁止重新估计。

独立空间训练入口已实现调度/早停及恢复，不将 N1HW 输出广播给旧标量 loss。
HPO 仅使用验证集，空间模型与标量模型应固定输入/split/预算，并明确损失差异。
评估同时报告逐格、每帧总量、低/高 SFR 与 burst 指标；正式效果须经过训练才能判断。

## 6. 验证命令

```powershell
python Experiment/SFRUNet.py --smoke-test --c-in 12 --shape 12 20 --device cpu
python -m unittest tests.test_sfr_unet -q
```

CLI 仅支持 `--smoke-test`、`--c-in`（默认12）、`--shape H W`（默认12 20）、
`--device cpu|cuda`（默认cpu），不会写训练结果。
2026-11-02 已通过 CPU 合成测试和五组真实盘星系共 15 帧的三步梯度 smoke；
未进行正式训练或全量 sidecar 生成。2026-11-03 补充了 CPU 恢复及 CUDA/BF16 smoke，
并用五组真实盘星系的15帧跑通 Exp1/Exp2 角色的小规模训练，详见研究计划；未验证 A100 吞吐量。
