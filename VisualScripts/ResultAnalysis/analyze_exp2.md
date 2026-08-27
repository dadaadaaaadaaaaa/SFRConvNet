# `analyze_exp2.py` 使用说明

## 1. 功能概述

`analyze_exp2.py` 专门分析 leave-one-galaxy-out 的 Exp2 结果，并把四个评估域分开：

1. cache 中的 in-distribution `test.pt`；
2. cache 中的 held-out galaxy `out.pt`；
3. 训练星系的完整 processed HDF5 轨迹；
4. held-out galaxy 的完整 processed HDF5 轨迹。

脚本会验证训练组和留出组互斥，检查 cached held-out 与 raw held-out 的时间、真值和预测一致性，并保存逐样本预测、指标、图和分析 metadata。

## 2. 必需输入

每个结果目录必须包含：

- `hyperparameters.json`；
- 且恰好一个 `*best_model.pth`。

由 `hyperparameters.json` 中的 `cache_dir` 和 `data_exp` 定位：

- `<prefix>out.pt`；
- `<prefix>norm.npz`；
- `<prefix>metadata.json`；
- 若启用 in-distribution 分支，还需要 `<prefix>test.pt`。

metadata 必须声明：

```text
split_strategy = leave_one_galaxy_out
```

并包含非空 `train_groups` 和唯一 `held_out_groups`。

## 3. 命令行参数

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| `--result-dir` | 空 | 要分析的结果目录；可重复 |
| `--all-folds` | 关闭 | 自动发现 `results-root` 下每个 exp2 cache 的最新结果 |
| `--results-root` | `<项目根>/Results` | 结果根目录 |
| `--device` | `auto` | `auto`、`cpu`、`cuda` 或任意合法 PyTorch device 字符串 |
| `--batch-size` | `128` | ConvNet 和非 ViT 推理 batch 大小，必须为正 |
| `--plot-in-distribution` | `true` | `true/false`；是否执行 cached test 推理和绘图 |
| `--plot-in-domain` | `true` | `true/false`；是否执行训练星系完整 raw 轨迹推理和绘图 |

`--result-dir` 与 `--all-folds` 不能同时使用。两个 held-out 分支始终执行，不能通过上述绘图开关关闭。

## 4. 主要 Python 接口

```python
analyze_result(
    result_dir,
    device,
    plot_in_distribution=True,
    plot_in_domain=True,
    batch_size=128,
)
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `result_dir` | 必填 | 单个训练结果目录 |
| `device` | 必填 | `torch.device` |
| `plot_in_distribution` | `True` | cached test 分支 |
| `plot_in_domain` | `True` | raw in-domain 分支 |
| `batch_size` | `128` | 推理 batch 大小 |

函数返回 `metrics_exp2.csv` 路径。

## 5. 执行命令

分析一个结果目录：

```powershell
python VisualScripts/ResultAnalysis/analyze_exp2.py `
  --result-dir Results/Exp2_LOOfiducial4_20260823 `
  --device auto `
  --batch-size 128
```

一次分析多个指定折：

```powershell
python VisualScripts/ResultAnalysis/analyze_exp2.py `
  --result-dir Results/Exp2_LOOfiducial4_20260823 `
  --result-dir Results/Exp2_LOOfiducial7_20260823
```

发现并分析全部最新折：

```powershell
python VisualScripts/ResultAnalysis/analyze_exp2.py --all-folds
```

只保留 held-out 分支：

```powershell
python VisualScripts/ResultAnalysis/analyze_exp2.py `
  --result-dir Results/Exp2_example `
  --plot-in-distribution=false `
  --plot-in-domain=false
```

## 6. 输出

每个结果目录中可能生成：

- `in_dist_test_predictions.csv`；
- `held_out_test_predictions.csv`；
- `raw_in_domain_predictions.csv`；
- `raw_held_out_predictions.csv`；
- `metrics_exp2.csv`；
- `exp2_analysis_metadata.json`；
- 根据分支组合命名的 `figures_*` 目录。

分析多个折时还会在 `results-root` 生成：

- `Exp2_LOO_metrics_all_folds.csv`；
- `Exp2_LOO_cross_fold_summary.csv`。

## 7. 一致性检查

脚本拒绝以下情况：

- train 与 held-out 组重叠；
- `out.pt` 的实际组与 metadata 不一致；
- cached held-out 与 raw held-out 样本数不同；
- 时间最大差超过 `1e-5 Gyr`；
- 真值最大差超过 `1e-6`；
- 预测最大差超过 `1e-5`；
- 结果目录有零个或多个 best checkpoint。

这些阈值用于发现 cache/raw 数据路径、mirror、标准化或 checkpoint 契约不一致，不是模型精度阈值。

## 8. 注意事项

1. 本脚本只接受正式 leave-one-galaxy-out metadata，不能拿随机 exp1 结果冒充 exp2；
2. raw 推理会读取完整轨迹，运行时间和内存开销明显高于 cache 推理；
3. direct 与 signed closure 可共享本分析工具，但必须依赖 checkpoint 中记录的目标变换和 closure 契约；
4. 多折汇总只比较同一种 cache 构造和训练协议，不能把 legacy mirror 与新原生网格结果混合。
