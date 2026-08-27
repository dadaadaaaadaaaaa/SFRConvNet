# `prepare_data.py` 使用说明

## 1. 功能概述

`prepare_data.py` 把 `raw_data_processing` 生成的 processed HDF5 转换为 PyTorch CachePT。它负责：

1. 加载多个盘星系运行；
2. 检查目标语义、粗粒化算子和输入通道；
3. 排除初始化 snapshot、NaN/Inf 和可选的零 SFR；
4. 完成数据划分和训练组过采样；
5. 只用训练集计算通道均值与标准差；
6. 保存 direct-SFR 或 signed-closure 监督量；
7. 保存通道、单位、目标变换、条件变量、UID 和 split metadata。

正式流程默认不执行 mirror。`SFRSpatialAugmentor` 也不在本脚本中执行。

## 2. Cache 模式

### `direct_sfr`

模型目标是变换后的全局 SFR：

\[
z_Y=T(Y_\alpha).
\]

### `signed_closure`

需要 processed HDF5 已保存：

\[
Y_\alpha,\quad B_\alpha,\quad
C_\alpha^{\rm true}=Y_\alpha-B_\alpha.
\]

闭合项使用 `signed_asinh` 或 `identity`，允许正值和负值。Cache 中同时保留物理空间的 (Y,B,C)，并验证 (Y=B+C)。

## 3. (R_h/R_m) 条件契约

- 固定单分辨率 (R_h)：`conditioning.name="none"`、`active=false`，保存零占位但不允许 FiLM 使用；
- (R_m)：`conditioning.name="log_mass_filter"`、`active=true`，数值为 `log10(M_filter)`；
- 当前禁止在同一个 CachePT 中混合具有不同条件语义的 (R_h) 与 (R_m)。

## 4. 实验定义

| 实验 | 当前含义 |
|---|---|
| `exp1` | 盘星系组内随机 snapshot 60/20/20 参考划分 |
| `exp2` | 对所有可用盘星系逐一执行 leave-one-galaxy-out；每折保存 train/val/test/out |
| `exp3` | 删除 `disk_galaxy_low` 后做随机 snapshot 划分 |
| `exp4` | 旧版单星系留出：将第三个可用的非 low 运行作为 out；新研究优先使用 exp2 |

重要限制：exp1/exp3 和 exp2 折内的 train/val/test 仍是随机 snapshot 划分，相邻时间帧可能泄漏。正式时间泛化结论应另行实现 time-block/run-block split。

## 5. 命令行参数

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| `--exclude-channels` | `initial_stellar_density` | 逗号分隔的排除通道；空字符串表示不排除 |
| `--cache-tag` | `InstantSFR_NoInitialStellarDensity` | Cache 文件名前缀标签 |
| `--resolution-dirname` | `coarse` | 每个 galaxy 目录下的 processed 子目录，如 `coarse_rh_8x8_sfr_v6` |
| `--target-transform` | `log10_1p` | `log10_1p` 或 `log10_floor` |
| `--target-scale` | `1.0` | SFR 目标变换尺度，单位必须与 HDF metadata 一致 |
| `--target-floor` | `1e-9` | `log10_floor` 使用的正数下限 |
| `--objective-mode` | `direct_sfr` | `direct_sfr` 或 `signed_closure` |
| `--closure-transform` | `signed_asinh` | `signed_asinh` 或 `identity` |
| `--closure-scale` | `1.0` | 闭合项变换尺度；必须只根据训练集确定 |
| `--drop-zero-targets` | 关闭 | 开启后删除零 SFR；默认保留物理零 SFR |
| `--exclude-raw-snapshot-index` | 空 | 精确排除原始 snapshot index；可重复，例如初始化帧 `0` |
| `--seed` | `42` | 划分及抽样随机种子 |
| `--only-exp1` | 关闭 | 旧兼容别名，只生成 exp1 |
| `--experiments` | `exp1 exp2 exp3 exp4` | 一次选择一个或多个实验 |
| `--experiment` | 空 | 可重复的单数形式 |
| `--held-out-galaxy` | 全部可用组 | 只生成指定 exp2 留出折；可用昵称或完整组名，可重复 |
| `--oversample` | `true` | `true/false`；是否对训练组过采样平衡 |

## 6. 输出文件

以 cache prefix 为 `Exp1_<tag>_` 为例：

```text
.cache/Exp1_<tag>_train.pt
.cache/Exp1_<tag>_val.pt
.cache/Exp1_<tag>_test.pt
.cache/Exp1_<tag>_norm.npz
.cache/Exp1_<tag>_metadata.json
```

exp2 每个留出折还包含 `out.pt`，并生成总 manifest：

```text
.cache/Exp2_<tag>_folds.json
```

## 7. 执行命令

固定 `8x8` (R_h) direct-SFR：

```powershell
python Experiment/prepare_data.py `
  --experiments exp1 `
  --resolution-dirname coarse_rh_8x8_sfr_v6 `
  --cache-tag Rh8x8_Direct `
  --objective-mode direct_sfr `
  --exclude-raw-snapshot-index 0 `
  --oversample=false
```

相同 (R_h) signed closure：

```powershell
python Experiment/prepare_data.py `
  --experiments exp1 `
  --resolution-dirname coarse_rh_8x8_sfr_v6 `
  --cache-tag Rh8x8_Closure `
  --objective-mode signed_closure `
  --closure-transform signed_asinh `
  --closure-scale 1.0 `
  --exclude-raw-snapshot-index 0 `
  --oversample=false
```

生成 exp2 的某一个留出折：

```powershell
python Experiment/prepare_data.py `
  --experiments exp2 `
  --held-out-galaxy fiducial_4
```

查看参数：

```powershell
python Experiment/prepare_data.py --help
```

## 8. 注意事项

1. direct 与 closure 公平比较时必须使用同一 HDF 数据、通道、split manifest 和随机种子；
2. `closure_scale` 不得使用验证集或测试集统计量；
3. (R_h) cache 训练时必须关闭 FiLM；(R_m) 的活动 `log_mass_filter` 条件通常需要 FiLM；
4. 旧 `8x16` mirror cache 只能作为 legacy 结果，不能与新的原生 `8x8` cache 直接比较；
5. 训练前检查 metadata 中的 `target_units`、`operator_id`、`sample_uid` 和 `parent_hr_uid`。
