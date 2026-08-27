# `run_optuna.py` 使用说明

## 1. 功能概述

`run_optuna.py` 使用 Optuna 搜索 SFRConvNet 超参数。每个 trial 会：

1. 清理 Python 和 CUDA 缓存；
2. 调用 `train.parse_args([])` 获得训练默认参数；
3. 从预定义搜索空间采样；
4. 将训练轮数和部分系统参数设为 HPO 固定值；
5. 调用 `train.main(args)`；
6. 以最佳验证损失作为最小化目标；
7. 将 study 持久化到 SQLite，可中断后继续。

## 2. 当前搜索空间

| 参数 | 采样方式 | 可选值/范围 |
|---|---|---|
| `model_type` | 固定 | `convnet` |
| `base_ch` | categorical | `32`、`64`、`128` |
| `se_reduction` | categorical | `4`、`8`、`16` |
| `pos_num_bands` | categorical | `8`、`16`、`32`、`64` |
| `p_drop` | 等间隔浮点 | `0.0` 至 `0.5`，步长 `0.1` |
| `lr` | 对数均匀 | `1e-5` 至 `1e-2` |
| `weight_decay` | 对数均匀 | `1e-6` 至 `1e-3` |
| `batch_size` | categorical | `64`、`128`、`256`、`512`、`1024` |

代码中被注释掉的 Transformer、`use_film` 和增强搜索项不会参与优化。

## 3. 每个 trial 的固定参数

| 参数 | 固定值 | 含义 |
|---|---:|---|
| `epochs` | `90` | HPO 最大训练轮数 |
| `patience` | `15` | early stopping |
| `num_workers` | `0` | Windows 安全设置 |
| `log_dir` | `runs/HPO` | TensorBoard 目录 |
| `no_save` | `True` | 不保存正式最佳模型 |

未在本脚本中覆盖的参数继承 `train.py` 默认值。因此当前会继承 `objective_mode=direct_sfr`、`use_film=False`、`loss=mse` 等默认值。要优化 signed closure，必须先显式修改 HPO 配方，不能只换 cache 名。

## 4. 命令行参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--data-exp` | `Exp1_InstantSFR_NoInitialStellarDensity_` | Cache 文件前缀 |
| `--n-trials` | `600` | 最大 trial 数 |
| `--timeout` | `28800` 秒 | study 总时限，默认 8 小时 |

二者任一先达到都会停止当前 `study.optimize`。

## 5. 执行命令

运行默认 HPO：

```powershell
python Experiment/run_optuna.py
```

指定 cache、trial 数和 2 小时时限：

```powershell
python Experiment/run_optuna.py `
  --data-exp Exp1_Rh8x8_Direct_ `
  --n-trials 100 `
  --timeout 7200
```

查看参数：

```powershell
python Experiment/run_optuna.py --help
```

## 6. 输出与恢复

SQLite 数据库路径为：

```text
.cache/<data-exp>sfrconvnet_hpo.db
```

study 名称为 `.cache/<data-exp>sfrconvnet_hpo`，`load_if_exists=True`，因此同一 `data-exp` 再次运行会继续已有 study，而不是重新开始。

## 7. 注意事项

1. 不要用 direct-SFR 的最佳超参数直接宣布 closure 模型较差；两种目标应获得相同 HPO 预算；
2. `objective()` 当前捕获所有训练异常并返回无穷大，需结合日志判断是坏超参数还是数据/代码错误；
3. batch size 上限 `1024` 可能超过某些模型或 GPU 的显存；
4. HPO 使用的验证 split 必须无时间泄漏，否则 Optuna 会优化到过于乐观的指标；
5. 正式最优参数应重新进行多随机种子训练，而不是直接把某个 trial 当最终结果。
