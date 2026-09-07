# `run_optuna.py` 使用说明

## 1. 功能概述

`run_optuna.py` 使用 Optuna 搜索 SFRConvNet 超参数。每个 study 会：

1. 检查 train/val/test cache 和 `metadata.json` 是否完整；
2. 验证可用训练目标和条件变量契约；
3. 默认从 `prepared_objective` 自动解析训练目标，也允许显式指定；
4. 从搜索空间为每个 trial 采样配置；
5. 只用训练集训练、用验证集选择超参数，不在 HPO 阶段评估测试集；
6. 将 study 持久化到包含训练目标名称的 SQLite 数据库。

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

主损失函数、训练目标、负 SFR 惩罚系数和是否启用 FiLM 是固定实验条件，
不是同一 study 内的待搜索变量。

## 3. 每个 trial 的固定参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `epochs` | `90` | HPO 最大训练轮数 |
| `patience` | `15` | early stopping |
| `num_workers` | `0` | 可通过命令行修改 |
| `log_dir` | `runs/HPO/<objective>` | TensorBoard 目录 |
| `no_save` | `True` | 不保存正式最佳模型 |
| `skip_test` | `True` | 不在每个 trial 反复评估测试集 |

固定 (R_h) cache 会自动关闭 FiLM；带活动 `log_mass_filter` 条件的 (R_m)
cache 会自动启用 FiLM。二者均由 metadata 决定。

## 4. 命令行参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--cache-dir` | `.cache` | CachePT 和 metadata 所在目录 |
| `--data-exp` | `Exp1_InstantSFR_NoInitialStellarDensity_` | Cache 文件前缀 |
| `--objective-mode` | `auto` | `auto`、`direct_sfr` 或 `signed_closure` |
| `--negative-sfr-weight` | `1.0` | closure 模式下最终负 SFR 的惩罚系数 |
| `--loss` | `mse` | 所有 trial 固定使用 `mse`、`l1` 或 `huber` |
| `--num-workers` | `0` | DataLoader worker 数量 |
| `--n-trials` | `600` | 最大 trial 数 |
| `--timeout` | `28800` 秒 | study 总时限，默认 8 小时 |

trial 数或总时限任一先达到，都会停止当前 `study.optimize`。

## 5. 执行命令

为 signed-closure cache 调参：

```powershell
python Experiment/run_optuna.py `
  --data-exp Exp1_Rh8x8_Closure_auto_ `
  --objective-mode signed_closure `
  --negative-sfr-weight 1.0 `
  --loss l1 `
  --n-trials 100 `
  --timeout 28800
```

也可以省略 `--objective-mode`。此时程序从 cache metadata 自动读取；启动时应明确看到：

```text
Resolved learning objective: signed_closure
```

查看参数：

```powershell
python Experiment/run_optuna.py --help
```

## 6. 输出与恢复

SQLite 数据库路径包含训练目标：

```text
.cache/<data-exp><objective-mode>_sfrconvnet_hpo.db
```

study 名称同样包含 `<objective-mode>`，因此 direct-SFR 和 signed-closure 不会
恢复到同一个历史 study。study user attributes 还会记录 cache 前缀、目标、条件变量
和 closure 变换，并保存 cache metadata 的 SHA-256 指纹、固定损失函数、负 SFR
惩罚系数及搜索空间版本；恢复时任一契约变化，程序都会拒绝继续。

## 7. 注意事项

1. 两种目标应使用相同的 split、HPO 搜索空间和计算预算；
2. CUDA OOM 只剪枝当前 trial；其他数据契约、维度或代码异常会立即中止；
3. batch size `1024` 可能使大模型 trial OOM，此时属于可预期剪枝；
4. HPO 验证 split 必须避免时间泄漏；
5. HPO 不评估测试集。确定超参数后应多随机种子正式训练，最后才使用测试集；
6. 旧版 `<data-exp>sfrconvnet_hpo.db` 可能记录了错误的 direct-SFR trial，应单独归档。
