# MACNet 项目指令

本文件供 Codex 在新对话中直接加载。它只规定项目级事实、工作边界与验证要求；研究细节和脚本参数应按下列索引查阅，不要在本文件中补猜。

## 1. 项目定位

- MACNet 使用 MACER 高分辨率模拟学习宇宙学模拟未解析的小尺度物理闭合，目标代码为 GIZMO/MFM。
- 当前二维主任务：从粗粒化气体场预测单个 snapshot 的总瞬时 SFR。
- 两条受支持路线必须并存：`direct_sfr` 直接预测真值 `Y`；`signed_closure` 学习 `C = Y - B`，最终预测为 `B + C_hat`。
- 这是分辨率条件闭合问题，不应默认改写成 LR→HR 超分辨率问题。
- filtered MACER 上有效不代表已迁移到真实 GIZMO；当前没有正式 GIZMO snapshot，`R_m` 也不等同于 GIZMO particle mass 或 MFM 重构。
- 三维目前仅完成 raw→fine→coarse 的 Direct/Residual 数据流程；3D dataset、CachePT、`SFRConvNet3D`、训练和结果分析尚未接通。不得宣称三维训练可用。

## 2. 开工前的文档路由

先阅读与任务直接相关的源码、测试和文档。涉及代码或研究语义的修改，至少先读 [README.md](README.md)，再按任务选择：

- 研究目标、闭合定义与长期路线：[MACNET_RESEARCH_CONTEXT.md](MACNET_RESEARCH_CONTEXT.md)
- 二维粗粒化、SFR 算子、UID、metadata 和验证记录：[COARSE_GRAINING_RM_RH_RESEARCH_PLAN.md](COARSE_GRAINING_RM_RH_RESEARCH_PLAN.md)
- 三维任务（必须完整阅读）：[3DSFRConvnet.md](3DSFRConvnet.md)
- 二维数据接口：[raw_data_processing/galaxy_data.md](raw_data_processing/galaxy_data.md)
- raw→fine：[raw_data_processing/build_fine_from_raw.md](raw_data_processing/build_fine_from_raw.md)
- fine→coarse：[raw_data_processing/build_coarse_from_fine.md](raw_data_processing/build_coarse_from_fine.md)
- Cache、模型、训练与 HPO：[prepare_data.md](Experiment/prepare_data.md)、[models.md](Experiment/models.md)、[train.md](Experiment/train.md)、[run_train.md](Experiment/run_train.md)、[run_optuna.md](Experiment/run_optuna.md)
- 已确认可靠的训练结果：[Train_history.md](Train_history.md)。不能仅因 `Results/` 中有文件就认定实验有效。

事实冲突时按以下优先级处理：

1. 当前可运行源码、数据内 metadata 契约和通过的测试；
2. `COARSE_GRAINING_RM_RH_RESEARCH_PLAN.md` 最新核验记录；
3. `3DSFRConvnet.md` 最新三维记录；
4. README、脚本文档；
5. 早期设想或公开数据 README。

仍无法消解时，列出冲突和证据，请用户确认；不要选择最方便的解释。

## 3. 目录职责

- `.config/`：本机路径、原始运行、字段映射和训练配置。`BaseConfig.json` 含机器相关绝对路径，不可假定盘符可移植。
- `raw_data_processing/`：日志解析、raw→fine、fine→coarse、物理算子、粗粒化和 metadata。
- `data/`：Parquet、processed HDF5 和诊断产品；不同契约必须使用不同子目录。
- `Experiment/`：CachePT、dataset、模型、训练、Optuna 和性能工具。
- `.cache/`：训练/验证/测试张量、统计量、metadata、split manifest 和 HPO 数据库。
- `Results/`：正式训练输出；`runs/`：TensorBoard；`VisualScripts/`：分析与可视化。
- `tests/`：数据语义、I/O、守恒、训练契约和回归测试。

不要覆盖用户已有数据、cache、checkpoint 或结果。smoke test 使用少量 snapshot 和独立目录。不要清理未由当前任务创建的文件。

## 4. 不可猜测的物理语义

二维数组标准顺序为 `(N_theta, N_r)`；轴对称仍保留方位速度 `v3`。

- 二维基本代码单位：`L0 = 1 kpc`、`t0 = 1 Gyr`、`M0 = 2.5e7 Msun`。1 code SFR = `0.025 Msun/yr`。
- Data-Set-5 / `density` 是质量密度，不是数密度。
- Data-Set-10 / `gas_energy` 是内能密度；总内能为 `sum(e * volume)`。
- Data-Set-11 / `sfr` 是每格已经乘体积的瞬时 SFR 贡献；snapshot 总 SFR 直接求和，不再乘体积。
- SFR 真值 `Y` 必须由 Data-Set-11 保守归约；HR 上重算物理算子只能用于验证，不能替代标签。
- Data-Set-15 / `grid_volume` 是权威体积，不能用规则索引面积代替。
- 日志 `NEWstar` 是累计新恒星质量，不是瞬时 SFR 标签。
- 二维温度满足 `T = 93 * e / rho`；不可套用于三维。
- 冷却量粗粒化应守恒总冷却功率 `sum(cooling_rate_density * volume)`，不能直接平均 cooling time。

三维数组统一为 `(r, theta, phi)`；原始 Athena block 顺序为 `(phi, theta, r)`。

- 三维默认候选输入为 `rho, press, vel1, vel2, vel3`；`rhodotsf` 是监督量，`cool_rate` 是 Residual baseline 专用量。
- `rhodotsf/sfr_density_true/star/newstar` 不得进入输入；`metallicity` 是否作为输入必须做独立消融。
- 三维内能密度为 `press/(gamma-1)`，温度使用 `mu(Z)` 关系；不要复用二维公式。
- 三维动量粗粒化先转换到固定笛卡尔基底，不能直接平均球坐标速度分量。
- 三维的完整语义以 `3DSFRConvnet.md` 为准。

物理量不确定时检查数据属性、源码、metadata 和测试；不得根据变量名、旧注释或二维经验猜测。

## 5. 数据集与泄漏边界

- `torus_100yr` 仅用于域外泛化测试，不得进入训练、验证、归一化统计、target/closure scale 估计或 HPO。
- torus 科学数据在嵌套的 `torus_100yr/torus_100yr/`；忽略 `__MACOSX/` 和 `.DS_Store`。
- torus 必须在 raw→fine 前排除坏帧：`747, 754, 755, 763, 764, 765, 766, 767, 768, 769, 770`。
- 保留 source run、原始 snapshot index、物理时间、坐标范围和排除清单；不要把 torus 称为与 Zou 等价的第六个完整盘星系。
- `exp1` 是组内随机 snapshot 60/20/20，仅适合参考和代码验证，不能作为严格时间外推证据。
- `exp2` 是 leave-one-galaxy-out；`exp3` 仅用五组 Zou 训练/验证，torus 为唯一 test-only；具体契约以当前代码和 README 为准。
- 相邻 snapshot 高度相关。正式泛化需使用 time-block、run-block、leave-one-run/galaxy-out 等设计，并优先做多随机种子和分组 bootstrap。
- 训练、验证、测试和 out 的 `sample_uid` / `parent_hr_uid` 不得交叉。
- 标准化、`log1p` 掩码、target transform 和 `closure_scale=auto` 只能由训练集估计。
- 标签、未来量、baseline、closure 及其派生量不得混入模型输入。

## 6. 数据与训练契约

- 二维新产品使用 format-version 6；三维使用独立的 `macnet_3d_format=1`。
- processed HDF5 必须带完整 `metadata/contract_json`、来源身份、单位/字段语义、算子参数、UID、守恒诊断和 canonical `(Y, B, C_true)`。
- 始终核验 `Y = B + C_true`；closure 可正可负。
- 不同 operator、shape、mass filter、kernel、物理算子版本或字段契约不得混写目录。
- 禁止把 coarse→coarse 作为默认生产流程。
- 修改分辨率时必须重建 coarse HDF5 和 CachePT，不能只改 CNN 输入 shape。
- `R_h` 固定单一分辨率使用 `conditioning=none` 并关闭 FiLM；`R_m` 只有在活动的 `log10(mass_filter)` 条件存在时才可启用 FiLM。
- 启动训练/HPO 时核对打印出的 learning objective、target metadata、conditioning contract 和输入通道；实验名不能替代 metadata。
- HPO 只使用验证集。选好超参数后重新进行正式训练，最好多随机种子；trial 结果不能冒充最终 test 结果。
- Direct 与 closure 公平比较需保持 coarse 数据、通道、split/UID、种子、模型容量、HPO 预算、loss 和指标一致。
- 正式分析同时报告模型与 MACER-LR baseline，并区分 physical-space 与 transformed target-space 指标。

## 7. 常用入口

```powershell
# 二维日志
python raw_data_processing/parse_log_file.py

# raw → fine（先用 --snapshot-index 或 --max-snapshots smoke test）
python raw_data_processing/build_fine_from_raw.py --galaxy disk_galaxy_fiducial --fine-dirname fine_sfr_v6

# fine → R_h coarse
python raw_data_processing/build_coarse_from_fine.py --galaxy disk_galaxy_fiducial --fine-dirname fine_sfr_v6 --coarsener spatial_block --coarse-shape 8 8 --coarse-dirname coarse_rh_8x8_sfr_v6

# CachePT / HPO / 正式训练；参数先查对应文档和 metadata
python Experiment/prepare_data.py --help
python Experiment/run_optuna.py --help
python Experiment/run_train.py <完整实验配置键>

# 测试
python -m pytest tests/<相关测试文件>.py -q
```

批处理前先运行最小数据 smoke test、HDF5 round-trip 和相关单元测试。不要因数据量大而跳过契约与守恒检查。

## 8. 修改与验收规范

- 先定位现有实现和测试，再做最小范围修改；保留用户的无关改动。
- 新代码沿用现有风格。除专有名词和公式外，代码注释与项目文档使用中文。
- 修改粗粒化、SFR 算子、字段、目标、split 或训练契约时，同步更新 metadata schema、测试、对应脚本文档和研究计划。
- 保留 direct-SFR 接口，同时维护 signed-closure 路线；不得为单一路线破坏另一条路线。
- 新算法至少验证：质量、三方向动量、内能、总冷却功率守恒；正体积与 shape；有限值；单位一致；closure 恒等式；UID 隔离；标签泄漏；边界/零 SFR/爆发期行为。
- 正式二维流程默认禁用 legacy mirror 和 `SFRSpatialAugmentor`；不要把 8×16 mirror cache 与原生 8×8 产品直接比较。
- 不把二维结论外推到三维，不把 filtered-HR 结论外推到 GIZMO，不把相关性描述为因果性。
- 不在 `Train_history.md` 登记仅 smoke-tested、未正式训练或仅存在 checkpoint 的实验。
- 重要结论注明证据类型：论文、数据 README、源码、metadata、测试或经验推断。
- 完成时说明修改文件、验证命令与结果、未验证项和剩余风险；不得声称未实际运行的测试已通过。

## 9. GitHub 同步约定

- 目标仓库为 `https://github.com/dadaadaaaadaaaaa/SFRConvNet`，远程名为 `origin`。
- 用户要求：每次完成较大的独立代码修改，运行相关验证后，提交并推送当前工作分支；无需就这一常规同步重复询问。提交信息应说明主要修改，交付时报告提交号和推送结果。
- 提交前检查差异和文件范围，只暂存本次任务相关文件；不要自动夹带其他未确认改动。
- 原始模拟数据（尤其 `*.athdf`）、训练 cache、模型权重、机器私有配置和凭据不推送。`.gitignore` 不会自动取消已跟踪文件，也不会移除历史中的文件。
- 不自动强制推送，不改写远端已有历史。需要清理未推送历史时，先明确范围并保留可恢复的本地备份引用，不能删除本地原始数据。
- 验证失败、认证失败、远端出现新提交或存在冲突时，报告具体情况；不要把失败或尚未验证的修改宣称为已完成同步。
