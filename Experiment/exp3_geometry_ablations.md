# Exp3.1 / Exp3.2：显式物理尺度输入消融

## 1. 实验定义与不变项

| 实验 | geometry-input-mode | 追加到原 x 的通道 |
|---|---|---|
| 原 Exp3 | `none` | 无 |
| Exp3.1 | `absolute_radius` | `log10_radius_kpc` |
| Exp3.2 | `absolute_radius_widths` | 上述半径及 `log10_radial_width_kpc`、`log10_polar_arc_width_kpc` |

新增通道分别为
`log10(r/(1 kpc))`、`log10((r_plus-r_minus)/(1 kpc))`、
`log10(r*(theta_plus-theta_minus)/(1 kpc))`，顺序为 `(theta,r)`。
读取 processed HDF5 保存的中心及 `axis_0_edges/axis_1_edges`，不由粗中心重新猜边界。
保存边界继承既有上游几何流程，其中 fine 边界由中心推断，不能把它们称为独立测量的 raw faces。

不修改原相对 Fourier、物理场、权威体积、Y/B/C、target/closure 反变换、CNN 池化/输出头。
标准化只用 Zou train；几何通道已经 log10，禁止再 log1p，禁止逐帧 min-max。
torus 仍是唯一 test-only；五组 Zou 随机 75/25，训练运行等权、运行内按物理时间加权。
不新增 snapshot 排除规则，不删除 torus index 0，不修改用户指定的 11 个坏帧清单。

默认模式 `none` 保留旧 schema 3/4/5；几何 cache 使用新 schema 6，与 processed HDF5
format-version 6 是两个不同概念。PT、NPZ、JSON 和 checkpoint 记录几何契约。
独立 cache 拒绝覆盖已有产物；旧 checkpoint 第一层通道数不同，必须重新训练。

## 2. 物理宽度是否等于空间分辨率输入？

它是局部、方向性的几何分辨率信息，不只是一个全域分辨率标量。
`Δr` 与 `rΔtheta` 表达各粗格平均区域的长度；相同体积也可能具有不同形状。
它们不等于求解器包含数值扩散、软化和反馈尺度在内的完整有效分辨率，不等于
R_m 输运核支撑尺度，也不使普通像素卷积变成固定物理长度卷积。

GIZMO/MFM 没有同样的球极 cell faces，应从目标代码可获得的核支撑/邻域几何定义
对应尺度，再重新验证契约。宇宙学模拟通常在核区比 MACER 更粗，但物理尺度取决于
局部密度、质量分辨率和细化，不能断言其所有单元必然更宽。本实验尚未验证真实 GIZMO。

## 3. 数据前提：先统一日志时间修复产品

2026-10-07 本机检查时只有旧 `coarse_rh_12x20_sfr_v6`，未生成
`coarse_rh_12x20_sfr_v7_timefix`。下面正式命令以修复后的目录为准；目录未齐时会停止。
既有三个 Fiducial flat 的 v6 仍保留旧日志上下文，不能把在 v6 上的后续训练冒充时间修复后结果。

先按研究计划 31.9 重建三个 flat 的 `fine_sfr_v7_timefix`，再将六个运行的 coarse
统一写入 `coarse_rh_12x20_sfr_v7_timefix`。本机 Model0.75/1.1 的可复用 fine 是
`fine_sfr_v6`；torus 是 `fine`。不覆盖旧产品。raw 重建需要挂载 E 盘及 pyhdf/Parquet 环境。
若选择在别处存放新数据，先配置 `.config/BaseConfig.json`，不要混淆本机/服务器绝对路径。

PowerShell 前置重建示例（先对独立 smoke 目录抽样核验，再执行全量）：

```powershell
python raw_data_processing/build_fine_from_raw.py `
  --galaxy disk_galaxy_fiducial `
  --galaxy disk_galaxy_fiducial_4 `
  --galaxy disk_galaxy_fiducial_7 `
  --fine-dirname fine_sfr_v7_timefix

python raw_data_processing/build_coarse_from_fine.py `
  --galaxy disk_galaxy_fiducial `
  --galaxy disk_galaxy_fiducial_4 `
  --galaxy disk_galaxy_fiducial_7 `
  --fine-dirname fine_sfr_v7_timefix `
  --coarsener spatial_block --coarse-shape 12 20 `
  --coarse-dirname coarse_rh_12x20_sfr_v7_timefix

python raw_data_processing/build_coarse_from_fine.py `
  --galaxy disk_galaxy_model_0_75 --galaxy disk_galaxy_model_1_1 `
  --fine-dirname fine_sfr_v6 `
  --coarsener spatial_block --coarse-shape 12 20 `
  --coarse-dirname coarse_rh_12x20_sfr_v7_timefix

python raw_data_processing/build_coarse_from_fine.py `
  --galaxy disk_galaxy_torus_100yr --fine-dirname fine `
  --coarsener spatial_block --coarse-shape 12 20 `
  --coarse-dirname coarse_rh_12x20_sfr_v7_timefix
```

Linux 使用相同参数，将每个行末反引号替换为反斜杠 `\`。原 Exp3 对照也必须使用同一修复后
产品、snapshot 清单、split/UID、归一化和本路线超参数重新训练；不能把旧 v6 的 13 倍结果
与新产品结果直接比较后把全部变化归因于几何输入。仅新增几何通道本身不要求重做 coarse；
上述重建是此前日志时间问题的独立前提。

## 4. 本机 Windows PowerShell：Closure

从项目根目录运行，先激活原训练环境。反引号必须是每行最后一个字符，后面不能有空格。

```powershell
python Experiment/prepare_data.py `
  --experiments exp3 `
  --resolution-dirname coarse_rh_12x20_sfr_v7_timefix `
  --cache-tag Rh12x20_FiveRuns_Closure_AbsRadius `
  --geometry-input-mode absolute_radius `
  --objective-mode signed_closure `
  --closure-transform signed_asinh --closure-scale auto `
  --seed 42 --oversample=false

python Experiment/run_train.py `
  --experiments exp3.1

python Experiment/prepare_data.py `
  --experiments exp3 `
  --resolution-dirname coarse_rh_12x20_sfr_v7_timefix `
  --cache-tag Rh12x20_FiveRuns_Closure_AbsRadiusWidths `
  --geometry-input-mode absolute_radius_widths `
  --objective-mode signed_closure `
  --closure-transform signed_asinh --closure-scale auto `
  --seed 42 --oversample=false

python Experiment/run_train.py `
  --experiments exp3.2
```

## 5. 服务器 Linux Bash：Closure

先进入服务器项目根目录，激活环境，并将 `.config/BaseConfig.json` 改为服务器的实际路径。
续行符是反斜杠 `\`，不是正斜杠 `/`；行末同样不能有空格。

```bash
python Experiment/prepare_data.py \
  --experiments exp3 \
  --resolution-dirname coarse_rh_12x20_sfr_v7_timefix \
  --cache-tag Rh12x20_FiveRuns_Closure_AbsRadius \
  --geometry-input-mode absolute_radius \
  --objective-mode signed_closure \
  --closure-transform signed_asinh --closure-scale auto \
  --seed 42 --oversample=false

python Experiment/run_train.py \
  --experiments exp3.1

python Experiment/prepare_data.py \
  --experiments exp3 \
  --resolution-dirname coarse_rh_12x20_sfr_v7_timefix \
  --cache-tag Rh12x20_FiveRuns_Closure_AbsRadiusWidths \
  --geometry-input-mode absolute_radius_widths \
  --objective-mode signed_closure \
  --closure-transform signed_asinh --closure-scale auto \
  --seed 42 --oversample=false

python Experiment/run_train.py \
  --experiments exp3.2
```

## 6. Direct 路线

在上面的 cache 命令中将 `--objective-mode` 改为 `direct_sfr`，cache-tag 的
`Closure` 改为 `Direct`，其余几何/seed/split 参数不变；训练入口增加 `-direct`。
下面单行命令在 PowerShell/Bash 均可使用：

```text
python Experiment/prepare_data.py --experiments exp3 --resolution-dirname coarse_rh_12x20_sfr_v7_timefix --cache-tag Rh12x20_FiveRuns_Direct_AbsRadius --geometry-input-mode absolute_radius --objective-mode direct_sfr --seed 42 --oversample=false
python Experiment/run_train.py --experiments exp3.1-direct
python Experiment/prepare_data.py --experiments exp3 --resolution-dirname coarse_rh_12x20_sfr_v7_timefix --cache-tag Rh12x20_FiveRuns_Direct_AbsRadiusWidths --geometry-input-mode absolute_radius_widths --objective-mode direct_sfr --seed 42 --oversample=false
python Experiment/run_train.py --experiments exp3.2-direct
```

## 7. 配方、cache 与结果契约

| 路线 | 配方 | base_ch / batch | Fourier bands / max freq | loss |
|---|---|---|---|---|
| Closure | 原 Exp3 Trial23 | 32 / 64 | 8 / 101 | 最终 SFR/closure_scale 的 L1 |
| Direct | 原 Exp3 Trial61 | 128 / 128 | 16 / 32 | transformed target 的 MSE |

同一路线两项消融的其他参数相同：seed42、90 epoch、patience15、depths2,2,2,2、
无 FiLM/增强/HPO 文件、run_time_balanced、tensor_cpu。两路线配方不同，不能把两路线的
差别全归因于 Direct/Closure；几何消融应在各自路线内部比较。
额外输入会增加 stem 第一层参数，故不是严格等参数量比较。

cache 前缀分别为：

```text
.cache/Exp3_Rh12x20_FiveRuns_Closure_AbsRadius_
.cache/Exp3_Rh12x20_FiveRuns_Closure_AbsRadiusWidths_
.cache/Exp3_Rh12x20_FiveRuns_Direct_AbsRadius_
.cache/Exp3_Rh12x20_FiveRuns_Direct_AbsRadiusWidths_
```

结果写入 `Results/Exp3.1_Closure_Trial23_seed42_<timestamp>/` 等独立目录。
运行时检查 `Geometry input contract`、输入通道、objective、conditioning 和 split。
当前原 Exp3 的 12 通道配方将分别成为 13/15 通道；若源字段契约变化，应以打印值为准。
原 `Results_stat_Exp3.ipynb` / `analyze_exp3.py` 的共享加载器已接受新 schema，结果目录
可显式选择新结果。不得根据实验名猜 objective 或几何模式。

新增验证覆盖几何公式/边界/单位、标签与守恒量不变、训练统计不泄漏、75/25 和 UID、
cache/sidecar/消融模式防错、Direct/Closure 前后向及共享分析接口。
本轮不执行正式全量数据构建或训练，不登记 Train_history，不承诺改善 13 倍高估。

后续选择架构/参数只使用 Zou 内部验证。torus 已用于失败诊断，后续针对该失败设计的
消融属于探索性域外分析，不能再把它称为完全未接触的独立最终测试。
