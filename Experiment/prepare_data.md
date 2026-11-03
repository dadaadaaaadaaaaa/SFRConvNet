# `prepare_data.py` 使用说明

## 逐格监督的独立附加文件（2026-11-02）

本脚本仍生成原标量 cache，不自动启用逐格模型。已补齐局部标签的二维 R_h HDF5 也可正常
走原标量流程。为了复用相同 split/UID/标准化，可通过
[build_cell_supervision_cache.md](build_cell_supervision_cache.md) 生成逐格监督 sidecar；
原 cache 和 `closure_scale` 不改。逐格模型使用独立训练尺度；Exp1/Exp2 的新训练入口见
[run_train_sfr_unet.md](run_train_sfr_unet.md)，不改变本脚本的标量 cache 接口。

## 1. 功能概述

`prepare_data.py` 把 `raw_data_processing` 生成的 processed HDF5 转换为 PyTorch CachePT。它负责：

1. 加载多个盘星系运行；
2. 检查目标语义、粗粒化算子和输入通道；
3. 排除初始化 snapshot、NaN/Inf 和可选的零 SFR；
4. 完成数据划分，并按实验契约决定训练期是否过采样；
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
| `exp3` | 五组 Zou 盘星系按运行分层后随机做 75% train / 25% validation；`torus_100yr` 是唯一 test-only 域 |
| `exp4` | 旧版单星系留出：将第三个可用的非 low 运行作为 out；新研究优先使用 exp2 |
| `exp8` | 原 exp3：删除 `disk_galaxy_low` 后做随机 snapshot 60/20/20 划分 |

Exp3 的五个 Zou 运行固定为 `fiducial`、`fiducial_4`、`fiducial_7`、
`model_0_75`、`model_1_1`。配置中标记 `role=test_only` 的 torus 不会进入其他盘星系
实验的默认训练路径。Exp3 不在 CachePT 中复制少样本运行；训练时应使用
`run_time_balanced` sampler，使每个运行总抽样权重相同，并在运行内部按 snapshot 所代表的
物理时间区间加权。validation 和 torus test 都不重采样。

重要限制：exp1/exp8、exp2 折内划分以及当前 exp3 的 Zou train/validation 划分仍采用随机
snapshot。相邻时间帧可能跨 split，因此当前 Exp3 的 validation 不是严格的时间外推验证；
这是本轮明确保留的实验选择。

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
| `--closure-scale` | `1.0` | 正浮点数或 `auto`。`auto` 只从临时训练 cache 的 `closure_true_physical` 估计尺度 |
| `--drop-zero-targets` | 关闭 | 开启后删除零 SFR；默认保留物理零 SFR |
| `--exclude-raw-snapshot-index` | 空 | 精确排除原始 snapshot index；可重复，例如初始化帧 `0` |
| `--seed` | `42` | 划分及抽样随机种子 |
| `--only-exp1` | 关闭 | 旧兼容别名，只生成 exp1 |
| `--experiments` | `exp1 exp2 exp3 exp4 exp8` | 一次选择一个或多个实验 |
| `--experiment` | 空 | 可重复的单数形式 |
| `--held-out-galaxy` | 全部可用组 | 只生成指定 exp2 留出折；可用昵称或完整组名，可重复 |
| `--oversample` | `true` | `true/false`；是否对训练组过采样平衡 |
| `--progress` | `true` | `true/false`；显示跨星系的 processed HDF5 snapshot 加载进度 |

## 6. 输出文件

以 cache prefix 为 `Exp1_<tag>_` 为例：

```text
.cache/Exp1_<tag>_train.pt
.cache/Exp1_<tag>_val.pt
.cache/Exp1_<tag>_test.pt
.cache/Exp1_<tag>_norm.npz
.cache/Exp1_<tag>_metadata.json
```

这些路径相对于启动脚本时的当前工作目录，而不是相对于 `prepare_data.py` 所在目录。
因此应从项目根目录运行；例如 `--cache-tag Rh8x8_Closure` 会生成
`.cache/Exp1_Rh8x8_Closure_train.pt`、`val.pt` 和 `test.pt` 等文件。

exp2 每个留出折还包含 `out.pt`，并生成总 manifest：

```text
.cache/Exp2_<tag>_folds.json
```

Exp3 输出仍是 `train.pt`、`val.pt`、`test.pt`：前两者只含五组 Zou 数据，`test.pt`
只含 torus。归一化均值、标准差、输入通道的 `log1p` 掩码及可选 closure scale 都只由
train split 确定；torus 不参与这些统计量。

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
  --closure-scale auto `
  --exclude-raw-snapshot-index 0 `
  --oversample=false
```

当 `--closure-scale auto` 时，每个实验（Exp2 则为每个 fold）独立执行：

1. 完成 train/val/test 划分和可选的训练组平衡；
2. 用临时的 `*_train.auto_scale_tmp.pt` 保存训练集；
3. 只读取其中的 `closure_true_physical`，计算
   `median(abs(closure_true_physical))`；
4. 如果该中位数因至少一半样本严格为零而退化为 0，则回退到非零绝对值的中位数；
5. 删除临时文件，用确定的尺度重新生成正式 train/val/test（以及 out）cache。

如果训练闭合项全部为零，程序会停止并要求显式检查数据或指定尺度。最终数值会写入
各 `.pt` 的 `learning_contract.closure_scale`、`norm.npz` 的 `closure_scale`，并在
`metadata.json` 的 `closure_scale_selection` 中记录来源、方法和训练集统计量。

生成 exp2 的某一个留出折：

```powershell
python Experiment/prepare_data.py `
  --experiments exp2 `
  --held-out-galaxy fiducial_4
```

生成本轮 12×20 Exp3 cache：

```powershell
python Experiment/prepare_data.py `
  --experiments exp3 `
  --resolution-dirname coarse_rh_12x20_sfr_v6 `
  --cache-tag InstantSFR_NoInitialStellarDensity `
  --objective-mode direct_sfr `
  --oversample=false
```

该命令要求五组 Zou 和 torus 的相同 resolution 目录均已生成；缺少任何一组都会停止，
不会自动缩减训练域或用别的运行代替。torus 的 11 个坏帧应在 `raw → fine` 阶段通过
`build_fine_from_raw.py --exclude-snapshot ...` 排除，而不是把同一个
`--exclude-raw-snapshot-index` 清单施加到所有六个运行。

查看参数：

```powershell
python Experiment/prepare_data.py --help
```

## 8. 注意事项

1. direct 与 closure 公平比较时必须使用同一 HDF 数据、通道、split manifest 和随机种子；
2. `closure_scale` 不得使用验证集或测试集统计量；推荐用 `auto` 避免手工先生成一次 cache；
3. (R_h) cache 训练时必须关闭 FiLM；(R_m) 的活动 `log_mass_filter` 条件通常需要 FiLM；
4. 旧 `8x16` mirror cache 只能作为 legacy 结果，不能与新的原生 `8x8` cache 直接比较；
5. 训练前检查 metadata 中的 `target_units`、`operator_id`、`sample_uid` 和 `parent_hr_uid`。
6. Exp3 会同时检查 group allowlist、输入通道、target metadata、`sample_uid` 与
   `parent_hr_uid` 的 split 互斥；任何不一致都会停止写 cache。

## 9. Exp3.1 / Exp3.2 的显式物理尺度通道

新增 `--geometry-input-mode`，默认 `none` 保持原输入：

| 模式 | 追加到 x 的通道 | 对应实验 |
|---|---|---|
| `absolute_radius` | `log10_radius_kpc` | Exp3.1 |
| `absolute_radius_widths` | 上述通道及 `log10_radial_width_kpc`、`log10_polar_arc_width_kpc` | Exp3.2 |

公式分别是 `log10(r/(1 kpc))`、`log10(Δr/(1 kpc))`、
`log10(rΔtheta/(1 kpc))`。读取 processed HDF5 的坐标中心和保存边界；不按
索引估算宽度，不覆盖 Data-Set-15 的权威体积。几何通道已取 log10，只做训练集
均值/标准差标准化，不再做 log1p，不在每帧重新 min-max。torus 复用训练统计。

新增输入使用独立 cache schema 6；旧 R_h schema 5 和 R_m schema 3/4 不变。
PT、metadata JSON、norm NPZ 保存几何公式、单位、通道及训练统计来源；新几何 cache
拒绝覆盖现有产物。必须指定独立 `--cache-tag`，不支持把几何消融用于 R_m FiLM。

仍使用 `--experiments exp3` 的 75/25 Zou 划分和 torus test-only 契约；训练阶段用
`run_train.py --experiments exp3.1` 或 `exp3.2`。完整命令见
[exp3_geometry_ablations.md](exp3_geometry_ablations.md)。
