# `run_train.py` 使用说明

## 1. 功能概述

`run_train.py` 是 `train.py` 的配置驱动启动器。它从 `.config/ExperimentSetting.json` 读取实验配置、展平各配置段、转换为 `train.py` 命令行参数，并启动独立训练子进程。

此外它支持：

- `exp1` 至 `exp8` 简写；
- 代码内可复现的实验别名和通道消融别名；
- 根据 exp2 的 `folds.json` 自动展开全部 leave-one-galaxy-out 训练；
- 仅训练指定 held-out galaxy；
- 批量训练失败后停止或继续；
- 为固定 (R_h) 强制关闭 FiLM。

## 2. 配置来源

默认实验键：

```python
TARGET_EXPERIMENT = "Exp1_InstantSFR_NoInitialStellarDensity_SFRConvNet"
```

脚本优先在 `.config/ExperimentSetting.json` 中查找完整键；若不存在，则检查 `EXPERIMENT_ALIASES`，复制基础实验并应用覆盖项。

JSON 常见配置段包括：

- `BaseInformation`：cache、实验名、随机种子和设备；
- `ModelArchitecture`：模型结构；
- `TrainingLoop`：batch、epoch、worker；
- `Optimization`：学习率、权重衰减、warmup；
- `LossFunction`：损失函数；
- `DataAugmentation`：增强概率；
- `System`：AMP 和 cuDNN 设置。

最终支持哪些字段由 `train.py` 的参数决定。未知字段会作为命令行参数传给 `train.py`，并导致其明确报错。

## 3. 命令行参数

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| 位置参数 `experiment` | 默认实验键 | 单个完整实验键或简写；为旧兼容保留 |
| `--experiment` | 空 | 可重复指定实验键或 `exp3` 等简写 |
| `--experiments` | 空 | 一次指定多个实验键/简写 |
| `--held-out-galaxy` | 全部折 | 对多折 manifest 只训练指定留出星系；可重复 |
| `--continue-on-error` | 关闭 | 开启后一个 job 失败仍继续后续 job |
| `--use-film` | `None`，遵循配置 | `true/false`、`1/0`、`yes/no`、`on/off`；覆盖全部所选 job |

位置参数不能与 `--experiment/--experiments` 同时使用。

## 4. exp2 自动展开

当 cache 目录存在：

```text
<data_exp>folds.json
```

且 `split_strategy=leave_one_galaxy_out` 时，脚本会把一个 exp2 配置展开为多个训练 job，每个 job 使用对应折的 `cache_prefix`。若显式请求 `exp2` 但找不到 manifest，脚本会停止并提示先执行 `prepare_data.py --experiments exp2`。

新 `exp3` 简写固定把 `sampling_mode` 覆盖为 `run_time_balanced`：训练 epoch 保持
`len(train)` 次抽样，每个 Zou 运行的总概率相同，运行内部按物理时间间隔加权。
固定 12×20 R_h 的 conditioning 不活动，因此该简写也自动设置 `use_film=false`。
validation 与 torus test 顺序读取且不重采样。旧 exp3 的配置名和 cache 前缀已经迁移为
`exp8`；显式使用旧的 `Exp3_*` 本地配置可能指向过期 cache，应改用 `Exp8_*`。

## 5. 执行命令

运行默认实验：

```powershell
python Experiment/run_train.py
```

运行一个完整配置键：

```powershell
python Experiment/run_train.py Exp1_InstantSFR_NoInitialStellarDensity_SFRConvNet
```

运行 exp2 全部留一折：

```powershell
python Experiment/run_train.py --experiments exp2
```

运行 Zou→torus 域外测试 Exp3：

```powershell
python Experiment/run_train.py --experiments exp3
```

只运行某个留出星系：

```powershell
python Experiment/run_train.py `
  --experiments exp2 `
  --held-out-galaxy fiducial_4
```

固定分辨率 (R_h) 强制关闭 FiLM：

```powershell
python Experiment/run_train.py `
  --experiment Exp1_InstantSFR_NoInitialStellarDensity_SFRConvNet `
  --use-film=false
```

批量运行且失败后继续：

```powershell
python Experiment/run_train.py `
  --experiments exp1 exp2 exp3 `
  --continue-on-error
```

## 6. 注意事项

1. `run_train.py` 不生成 cache；必须先执行 `prepare_data.py`；
2. 配置中的 `use_film=true` 不适用于固定 (R_h)，应使用 `--use-film=false` 或建立专用配置；
3. signed closure 还需要配置传递 `objective_mode=signed_closure`；
4. 脚本通过当前 Python 解释器启动子进程，运行前应先激活正确的 PyTorch 环境；
5. 每次训练的最终参数会写入结果目录，分析时应以该文件为准，而不是只看全局 JSON 配方。

## 7. Exp3.1 / Exp3.2 消融入口

| 简写 | 输入模式 | objective / 配方 |
|---|---|---|
| `exp3.1` 或 `exp3.1-closure` | 绝对半径 | signed_closure，原 Trial23 |
| `exp3.2` 或 `exp3.2-closure` | 绝对半径 + 两方向单元宽度 | signed_closure，原 Trial23 |
| `exp3.1-direct` | 绝对半径 | direct_sfr，原 Trial61 |
| `exp3.2-direct` | 绝对半径 + 两方向单元宽度 | direct_sfr，原 Trial61 |

这些入口在 tracked 源码中保存独立配方，不继承 Exp1，不读取未跟踪的本机 JSON 或旧
HPO 文件；原 `exp3` 简写行为不变。固定 R_h 不启用 FiLM，仍使用 `run_time_balanced`。
结果名分别为 `Exp3.1_Closure_Trial23_seed42` 等，cache 前缀及完整参数写入
`hyperparameters.json`。消融入口会检查几何模式，不能误用原无几何通道 cache。

运行前必须准备相应独立 cache；Windows PowerShell / Linux Bash 的完整命令见
[exp3_geometry_ablations.md](exp3_geometry_ablations.md)。
