# `train.py` 使用说明

## 1. 功能概述

`train.py` 是单次 MACNet 训练的核心脚本，负责：

1. 读取版本化 CachePT 和通道 metadata；
2. 检查模型条件变量与 cache 契约是否一致；
3. 构造 Transformer、旧版 ResNet/ResFiLM 或 SFRConvNet；
4. 训练 direct-SFR 或 signed-closure 目标；
5. 使用 AdamW、warmup+cosine 学习率和 early stopping；
6. 保存最佳 checkpoint、超参数、TensorBoard 日志和测试指标。

signed closure 模式下网络输出 (\widehat C) 的变换值，训练代码执行可微反变换并显式构造：

\[
\widehat Y=B+\widehat C.
\]

主误差作用于最终物理 SFR；另可对负的最终 SFR 加平方 hinge 惩罚。

## 2. 数据与目标参数

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| `--cache_dir` | `.cache` | CachePT 所在目录 |
| `--exp_name` | `Exp5_` | 输出实验名前缀 |
| `--data_exp` | `None` | 输入 cache 前缀；`None` 时使用 `exp_name` |
| `--objective_mode` | `direct_sfr` | `direct_sfr` 或 `signed_closure` |
| `--negative_sfr_weight` | `1.0` | closure 模式下负最终 SFR 惩罚系数；`0` 关闭，必须非负 |

`signed_closure` 当前只允许 `model_type=convnet`，且要求 cache schema 至少包含物理 (Y,B,C)。KS baseline 不能冒充 MACER baseline。

## 3. 训练与优化参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--batch_size` | `64` | batch 大小 |
| `--epochs` | `500` | 最大 epoch 数 |
| `--lr` | `1e-3` | AdamW 初始学习率 |
| `--weight_decay` | `0` | AdamW 权重衰减 |
| `--lr_min` | `0.0` | cosine 衰减后的最低学习率 |
| `--lr_warmup_steps` | `0` | 显式 warmup step 数；为 0 时转用 ratio |
| `--lr_warmup_ratio` | `0.2` | `warmup_steps=0` 时 warmup 占总 step 的比例；0 关闭 |
| `--lr_warmup_start_factor` | `0.0` | warmup 起点相对 `lr` 的比例 |
| `--patience` | `20` | early stopping 容忍 epoch 数；非正值可理解为不早停 |
| `--grad_clip` | `0.3` | 梯度范数上限；不大于 0 时关闭 |
| `--seed` | `42` | Python/NumPy/PyTorch 随机种子 |

## 4. DataLoader 与设备参数

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| `--num_workers` | Windows 为 `0`；其他系统至多 `16` | DataLoader 子进程数 |
| `--persistent_workers` | 关闭 | 保持 worker；仅 `num_workers>0` 有效 |
| `--prefetch_factor` | `3` | 每个 worker 预取 batch 数；worker 为 0 时忽略 |
| `--device` | `auto` | `auto`、`cpu`、`cuda` |
| `--no_amp` | 关闭 | 开启后禁用自动混合精度 |
| `--cudnn_benchmark` | 关闭 | 开启 cuDNN benchmark；固定输入尺寸时可能加速 |
| `--print_group_counts` | 关闭 | 打印各数据集合分组计数 |

## 5. 损失和样本权重

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| `--loss` | `mse` | `mse`、`l1`、`huber` |
| `--huber_delta` | `1` | Huber 转折尺度 |
| `--weight_mode` | `none` | `none`、`exp_clamp`、`group_zscore_exp`、`group_quantile` |
| `--weight_alpha` | `0.5` | 权重变化强度 |
| `--weight_tau` | `3.0` | z-score 指数权重温度 |
| `--weight_power` | `1.0` | 分位权重幂指数 |
| `--weight_q_low` | `0.1` | 低分位点 |
| `--weight_q_high` | `0.9` | 高分位点 |

不同 objective 的训练 loss 数值不应直接横向比较。模型优劣必须在相同测试样本的最终物理 SFR 指标上比较。

## 6. 模型参数

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| `--model_type` | `transformer` | `transformer`、`resnet`、`resfilm`、`convnet` |
| `--d_model` | `256` | Transformer 隐空间维数 |
| `--n_layers` | `4` | Transformer 层数 |
| `--n_heads` | `4` | 注意力头数，必须整除 `d_model` |
| `--d_ff` | `512` | Transformer FFN 维数 |
| `--pos_num_bands` | `32` | Fourier 位置频带数 |
| `--pos_max_freq` | `101` | Fourier 最高频率 |
| `--p_drop` | `0.3` | dropout 概率 |
| `--base_ch` | `64` | SFRConvNet 第一阶段通道数 |
| `--stage_depths` | `2,2,2,2` | SFRConvNet 每阶段 block 数 |
| `--se_reduction` | `4` | SE 压缩比 |
| `--use_film` | 关闭 | 使用活动条件标量调制 SFRConvNet；固定 (R_h) 必须关闭，(R_m) 可开启 |
| `--hpo_params` | `None` | JSON 超参数文件，匹配字段会覆盖命令行值 |

## 7. 数据增强和输出参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--aug_prob` | `0.0` | 旧径向裁剪概率；正式 SFR 实验保持 0 |
| `--aug_min_crop_ratio` | `0.6` | 径向裁剪最小保留比例 |
| `--sfr_aug_prob` | `0.0` | 旧 SFR 空间增强概率；当前实现不满足球极网格物理对称性，应保持 0 |
| `--log_dir` | `runs` | TensorBoard 日志根目录 |
| `--no_save` | 关闭 | 开启后不保存正式模型，常用于 HPO |

## 8. 执行命令

固定 (R_h) direct-SFR：

```powershell
python Experiment/train.py `
  --cache_dir .cache `
  --data_exp Exp1_Rh8x8_Direct_ `
  --exp_name Rh8x8_Direct `
  --model_type convnet `
  --objective_mode direct_sfr `
  --batch_size 256 `
  --epochs 100 `
  --sfr_aug_prob 0
```

固定 (R_h) signed closure：

```powershell
python Experiment/train.py `
  --cache_dir .cache `
  --data_exp Exp1_Rh8x8_Closure_ `
  --exp_name Rh8x8_Closure `
  --model_type convnet `
  --objective_mode signed_closure `
  --negative_sfr_weight 1.0 `
  --batch_size 256 `
  --epochs 100 `
  --sfr_aug_prob 0
```

不要为固定 (R_h) 加 `--use_film`。查看全部参数：

```powershell
python Experiment/train.py --help
```

## 9. 输出

未使用 `--no_save` 时，结果目录中通常包括：

- `*_best_model.pth`：最佳 checkpoint；
- `hyperparameters.json`：模型、目标、通道、split 和 conditioning 契约；
- TensorBoard 日志；
- 控制台输出的测试指标。

closure checkpoint 会额外记录 `objective_mode`、`closure_transform`、`closure_scale` 和 `negative_sfr_weight`。

## 10. 公平比较注意事项

1. direct 与 closure 应使用同一 HDF 来源、输入通道、split、模型容量和训练预算；
2. 至少运行多个随机种子；
3. 分别报告 baseline (B)、direct 预测和 (B+\widehat C)；
4. 不要用旧 mirror cache 的结果作为正式 matched direct 对照；
5. 目前随机 snapshot split 仍有时间泄漏风险，不能据此声称时间外推能力。
