# `models.py` 使用说明

## 1. 功能概述

`models.py` 定义 MACNet 使用的 PyTorch 网络结构。模型输入通常包括：

- 多通道二维气体物理场 `x`，形状为 `(B,C,H,W)`；
- 半径与极角坐标 `r, theta`，形状为 `(B,1,H,W)`；
- 可选全局条件标量。

模型输出为每个 snapshot 一个标量。输出语义由训练契约决定：

- `direct_sfr`：变换后的全局 SFR；
- `signed_closure`：变换后的闭合项，随后在训练/推理代码中计算 (B+\widehat C)。

本文件是模型库，没有独立命令行入口。

## 2. 可用模型

### `MACNetRes_mbh`

旧版残差 CNN，在全局池化后拼接黑洞质量。

```python
MACNetRes_mbh(in_channels=15)
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `in_channels` | `15` | 二维物理场通道数；实际训练由 cache 决定 |

只适合条件变量明确为 `mbh` 的旧吸积率任务，不适合当前固定 (R_h) SFR cache。

### `MACNetFiLM`

旧版残差 CNN，使用黑洞质量生成逐残差块 FiLM 参数。

```python
MACNetFiLM(in_channels=15)
```

同样要求条件变量为 `mbh`。

### `FourierPositionalEncoding2D`

将 `(r,theta)` 映射为 Fourier 位置特征。

主要参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `num_bands` | 构造时必填 | 每个坐标使用的频带数 |
| `max_frequency` | 构造时必填 | 最高频率 |
| `include_input` | `True` | 是否保留归一化后的原始坐标 |
| `theta_pi_periodic` | `False` | 是否强制极角具有 \(\pi\) 周期 |

当前 MACER 球极坐标不应未经物理论证就开启任意角向周期假设。

### `AccretionTransformer`

将每个网格像素转换为 token，加入 Fourier 位置编码和 CLS token，再通过 Transformer 与逐层 FiLM 回归标量。

```python
AccretionTransformer(
    c_in,
    d_model=256,
    n_layers=8,
    n_heads=8,
    d_ff=1024,
    pos_num_bands=32,
    pos_max_freq=64.0,
    p_drop=0.1,
)
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `c_in` | 必填 | 输入物理场通道数 |
| `d_model` | `256` | token 隐空间维数；必须能被 `n_heads` 整除 |
| `n_layers` | `8` | Transformer block 数 |
| `n_heads` | `8` | 多头注意力头数 |
| `d_ff` | `1024` | 前馈网络隐藏维数 |
| `pos_num_bands` | `32` | Fourier 频带数 |
| `pos_max_freq` | `64.0` | 位置编码最高频率 |
| `p_drop` | `0.1` | dropout 概率，范围通常为 `[0,1)` |

注意：`train.py` 的命令行默认值是 `n_layers=4`、`n_heads=4`、`d_ff=512`、`pos_max_freq=101`，会覆盖这里的构造默认值。

### `SFRConvNet`

当前 SFR 任务的主要 CNN：Fourier 坐标编码与物理通道拼接，经过多阶段残差卷积、SE block、可选 FiLM 和全局池化后输出标量。

```python
SFRConvNet(
    c_in,
    base_ch=64,
    stage_depths=(2,2,2,2),
    pos_num_bands=32,
    pos_max_freq=64.0,
    p_drop=0.1,
    se_reduction=4,
    use_film=False,
    prediction_mode="direct_sfr",
)
```

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| `c_in` | 必填 | 输入物理场通道数 |
| `base_ch` | `64` | 第一阶段通道数，后续阶段按 2 倍增长 |
| `stage_depths` | `(2,2,2,2)` | 每阶段残差块数量；元素必须为正整数 |
| `pos_num_bands` | `32` | Fourier 频带数 |
| `pos_max_freq` | `64.0` | Fourier 最高频率 |
| `p_drop` | `0.1` | dropout 概率 |
| `se_reduction` | `4` | SE 隐层压缩比，必须为正整数 |
| `use_film` | `False` | 是否用活动条件标量调制网络；固定单分辨率 (R_h) 必须为 `False` |
| `prediction_mode` | `direct_sfr` | `direct_sfr` 或 `signed_closure`，只规定输出语义，不在模型内部执行反变换 |

`forward` 接口：

```python
y = model(x, r, theta, conditioning_scalar=None)
```

若 `use_film=True`，必须提供与 cache metadata 一致的活动条件。目前正式支持的 SFR 条件是 (R_m) 的 `log10(M_filter)`。

## 3. 辅助模块

- `Residual`、`resnet_block`：旧版普通残差块；
- `FiLMResidual`：旧版带 FiLM 的残差块；
- `FiLMConditioner`：为 Transformer 各层生成 `gamma/beta`；
- `MLP`、`TransformerBlock`：Transformer 基础模块；
- `SEBlock`：通道注意力；
- `ConvFiLMBlock`：SFRConvNet 残差卷积块；
- `ConvFiLMConditioner`：为 CNN 各 stage 生成 FiLM 参数。

## 4. 调用示例

固定 (R_h) direct-SFR：

```python
from Experiment.models import SFRConvNet

model = SFRConvNet(
    c_in=11,
    base_ch=32,
    stage_depths=(2, 2, 1, 1),
    use_film=False,
    prediction_mode="direct_sfr",
)
prediction = model(x, r, theta)
```

signed closure：

```python
model = SFRConvNet(
    c_in=11,
    use_film=False,
    prediction_mode="signed_closure",
)
closure_transformed = model(x, r, theta)
```

物理反变换和 (B+\widehat C) 应调用 `train.py` 中的组合函数，不能假设网络输出本身就是物理 SFR。

## 5. 执行方式

本文件不应作为训练脚本直接执行：

```powershell
# 只做导入/语法检查
python -m py_compile Experiment/models.py

# 正式训练
python Experiment/train.py --model_type convnet
```
