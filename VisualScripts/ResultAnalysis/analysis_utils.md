# `analysis_utils.py` 使用说明

## 1. 功能概述

`analysis_utils.py` 是训练结果分析的共享工具库，连接 CachePT、processed HDF5、模型 checkpoint、物理单位变换、推理、指标和绘图。它主要用于结果 notebook 与 `analyze_exp2.py`，不提供独立命令行入口。

导入本模块时会调用 `setup_project_env()`：寻找包含 `Experiment`、`raw_data_processing` 或 `.config` 的项目根目录，将其加入 `sys.path`，并尝试把当前工作目录切换到项目根目录。

## 2. 兼容的模型和数据

### 模型

- 当前 `MACNetRes_mbh`；
- 当前 `MACNetFiLM`；
- 当前 `AccretionTransformer`；
- 当前 `SFRConvNet`；
- `LegacyAccretionTransformer`：用于读取旧 checkpoint 的兼容实现。

### 数据

- `CachedPTDataset`：读取训练时保存的 `.pt` cache；
- `HDF5Dataset`：读取完整 processed HDF5 轨迹，并复用训练集通道和标准化统计量；
- 支持 cache schema 3、4、5 及部分 legacy tuple cache；
- schema 4/5 可读取物理 (Y,B,C) 和 signed-closure 契约。

## 3. 主要类及可变参数

### `LegacyAccretionTransformer`

```python
LegacyAccretionTransformer(
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

仅用于旧 checkpoint 兼容，不建议训练新模型。

### `CachedPTDataset(pt_path)`

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `pt_path` | 必填 | `train.pt`、`val.pt`、`test.pt` 或 `out.pt` 路径 |

读取后提供 `x`、`coord`、`y`、条件标量、分组、时间、UID、目标 metadata；closure cache 还提供物理 baseline 与 closure。

### `HDF5Dataset(...)`

```python
HDF5Dataset(
    hdf5_conf,
    resolution="coarse",
    mean=None,
    std=None,
    log_transform=True,
    mirror=False,
    balance=None,
    input_channels=None,
    exclude_channels=None,
    reference_dataset=None,
    target_metadata=None,
    exclude_raw_snapshot_indices=None,
)
```

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| `hdf5_conf` | 必填 | HDF5 目录，或 `(目录,组名)` |
| `resolution` | `coarse` | 兼容保留参数；实际目录由 `hdf5_conf` 指定 |
| `mean`、`std` | `None` | 训练集标准化统计；正式推理必须传入 |
| `log_transform` | `True` | 是否对需要的通道执行加载器定义的 log 预处理 |
| `mirror` | `False` | `False/True/"match_reference"`；正式新数据通常为 `False` |
| `balance` | `None` | 可传 `oversample`；分析完整物理轨迹通常不应平衡 |
| `input_channels` | `None` | 显式输入通道顺序 |
| `exclude_channels` | `None` | 排除通道；若已有 include 列表则忽略 exclude |
| `reference_dataset` | `None` | 用于继承通道、目标 metadata 和空间形状 |
| `target_metadata` | `None` | 显式目标语义；通常从 reference cache 继承 |
| `exclude_raw_snapshot_indices` | `None` | 需排除的原始 snapshot index |

`mirror="match_reference"` 只为兼容旧 cache：原生形状与参考相同时不镜像；参考宽度恰为原生两倍时才镜像；其他形状组合会报错。

## 4. 模型与数据加载函数

### `load_cache_metadata(path)`

接受 cache split 路径或 cache prefix，读取相邻 `metadata.json`；不存在时返回空字典。

### `load_norm_stats(norm_path)`

读取 `.npz` 中的 `mean/std`、通道名和目标 metadata。

### `load_analysis_model(model_path, device, datasets_dict=None)`

从 checkpoint metadata 判断模型类型、通道数、结构和目标契约。`datasets_dict` 用于检查模型与测试/留出 cache 的通道及条件语义一致。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `model_path` | 必填 | `*.pth` 路径 |
| `device` | 必填 | `torch.device` |
| `datasets_dict` | `None` | 名称到 dataset 的映射，用于一致性校验 |

### `predict_dataset(ds, model, device, model_type, batch_size=32, return_feats=False)`

对一个 dataset 推理并返回 DataFrame；closure 模式会反变换 (\widehat C) 并显式计算 (B+\widehat C)。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `ds` | 必填 | `CachedPTDataset` 或 `HDF5Dataset` |
| `model` | 必填 | 已加载模型 |
| `device` | 必填 | 推理设备 |
| `model_type` | 必填 | checkpoint 解析出的模型类型 |
| `batch_size` | `32` | 非 ViT 推理 batch 大小 |
| `return_feats` | `False` | 是否同时返回中间特征 |

## 5. 单位和目标变换

- `_physical_sfr(values, metadata)`：按 metadata 反变换到物理 SFR；
- `_transformed_sfr_allow_negative_predictions(...)`：只变换非负物理预测，负值保留为 NaN，不静默裁剪；
- `_sfr_time_integration_factor(df, explicit_factor=None)`：把 SFR×Gyr 转为太阳质量：
  - `MACER_mass_code/Gyr` → `2.5e7`；
  - `Msun/yr` → `1e9`；
  - `Msun/Gyr` → `1`；
  - 未知单位必须显式传 `explicit_factor`。

## 6. 绘图函数

| 函数 | 关键可变参数 | 默认值/含义 |
|---|---|---|
| `save_plot(plt_obj,name,fig_dir,fmt="png")` | `fmt` | 默认 PNG |
| `plot_scatter(df,title,fig_dir,limit_range=True)` | `limit_range` | 是否限制绘图范围 |
| `plot_error_dist(df,title,fig_dir,model_label="Model")` | `model_label` | 图例名称 |
| `plot_kde_contour(df,title,fig_dir)` | 无可选参数 | 目标-预测 KDE |
| `plot_physical_sfr_scatter(...)` | 无可选参数 | 物理 SFR 散点 |
| `plot_closure_scatter(...)` | 无可选参数 | closure 预测图 |
| `plot_mass_evolution(...,model_label="Model")` | `model_label` | 累计质量演化图例 |
| `plot_cumulative_sfr(...,model_label="Model",sfr_unit_scale=None)` | `sfr_unit_scale` | 未给定时从 metadata 推导单位换算 |
| `plot_sfr_history(...,model_label="Model")` | `model_label` | SFR 时间序列图例 |

## 7. 指标函数

### `get_metrics_summary(y_true, y_pred, name="")`

返回变换空间的 RMSE、MAE、\(R^2\) 和离群比例等摘要。

### `get_physical_sfr_metrics(df, prediction_column="sfr_pred", name="Model")`

在物理 SFR 空间计算指标。比较 direct 与 closure 时应优先使用该函数或等价的共同物理指标，而不是直接比较两套训练 loss。

## 8. 调用示例

```python
from pathlib import Path
import torch
from VisualScripts.ResultAnalysis import analysis_utils as utils

test_ds = utils.CachedPTDataset(Path(".cache/Exp1_example_test.pt"))
model, model_type = utils.load_analysis_model(
    Path("Results/example/example_best_model.pth"),
    torch.device("cuda"),
    {"test": test_ds},
)
frame = utils.predict_dataset(
    test_ds,
    model,
    torch.device("cuda"),
    model_type,
    batch_size=128,
)
print(utils.get_physical_sfr_metrics(frame))
```

## 9. 执行方式与注意事项

本文件不作为独立程序运行：

```powershell
python -m py_compile VisualScripts/ResultAnalysis/analysis_utils.py
```

正式使用应导入它，或执行：

```powershell
python VisualScripts/ResultAnalysis/analyze_exp2.py --help
```

注意：

1. 分析 raw HDF5 时必须复用训练 cache 的通道顺序与 `mean/std`；
2. 不得为了适配旧模型而静默改变 mirror 模式；
3. SFR 单位未知时累计恒星质量会明确报错；
4. legacy checkpoint 的结果应与新 schema 结果分开标记。
