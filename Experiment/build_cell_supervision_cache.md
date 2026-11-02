# `build_cell_supervision_cache.py`：逐格监督附加 cache

读取已有标量 CachePT，按 `sample_uid` 匹配含逐格标签的 HDF5，输出独立 sidecar `.pt`。
**不改原 cache，不重新 split/标准化，不训练模型。** 过采样重复 UID 原样保留；
网格不作为独立样本重新划分。

需先用[逐格产品入口](../raw_data_processing/build_cell_sfr_supervision.md)补齐数据。
cache 与产品的 UID、parent UID、坐标、shape、时间和标量 Y/B/C 必须一致；
不接受 legacy mirror、无 UID cache 或 v6/v7 时间/物理版本混搭。

## 命令与参数

```powershell
python Experiment/build_cell_supervision_cache.py --cache-path .cache/<原实验前缀>_train.pt --resolution-dirname coarse_rh_12x20_sfr_v7_cell_v1 --output-path .cache/<新逐格前缀>_train_cells.pt
```

依次对原 `train/val/test` 和 Exp2 的 `out` 文件运行。尖括号内容是占位符，需替换为实际文件名。
不需要重新生成原 cache。后续跨 split 的 UID 隔离仍按原实验契约检查。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--cache-path` | 必填 | 原标量 cache；只读取自己生成的可信文件 |
| `--output-path` | 必填 | 新附加 cache，拒绝覆盖 |
| `--resolution-dirname` | 无 | 自动查找 `data/*/<目录名>/*.h5` |
| `--product-dir` | 无，可重复 | 显式产品目录，与上一参数二选一 |
| `--config-dir` | `.config` | 自动发现产品时读取 BaseConfig |
| `--data-dir` | BaseConfig.data_dir | 产品根目录覆盖值 |

每个独立 snapshot 只加载一次，每 100 帧显示进度。缺失标签、重复产品 UID、地图求和与
cache 标量不符会停止，不填零回退。

## 输出与下一阶段接口

独立 schema 为 `macnet_cell_supervision_cache/version=1`：

- `source_cache.filename/sha256` 绑定原 cache；
- `sample_identity` 原样保留 UID/parent UID 顺序；
- `tensors.target_sfr_map_physical`：`float64 (N,N_theta,N_r)` 的真值；
- `tensors.macer_sfr_baseline_map`：相同 shape 的 baseline；
- `tensors.closure_true_map_physical`：相同 shape 的 signed closure；
- `cell_product_uid/cell_contracts`：逐样本身份及去重后的局部契约。

单位为 code SFR，尚未 log/asinh/标准化，零标签保留。**原总量 closure scale 不自动复用为
逐格 scale**；未来局部 loss/scale 需要单独设计，只能由训练集估计。

sidecar 不含 x，不能单独交给旧 `run_train.py`。下一阶段新空间模型/dataset 应联合读取原 cache
与 sidecar，验证源 SHA256/UID，逐格预测后求和供原总量评估使用。
本次未修改 `prepare_data.py`、`models.py`、`train.py`、`run_train.py`、`run_optuna.py`。
