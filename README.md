# MACNet：面向宇宙学模拟的 MACER 小尺度物理闭合模型

> **2026-11-02：方案 C 已完成二维逐格监督数据层。** 新 raw→fine→R_h 可保存逐格 Y/B/C；
> [旧产品补齐入口](raw_data_processing/build_cell_sfr_supervision.md) 与
> [原 cache 的逐格监督附加文件](Experiment/build_cell_supervision_cache.md) 已接通。
> 独立的 [SFRUNet、逐格损失和数据适配器](Experiment/SFRUNet.md) 已通过合成及真实数据 smoke。
> **2026-11-03：Exp1/Exp2 逐格训练与 HPO 入口已接通**，见
> [正式训练](Experiment/run_train_sfr_unet.md)、[Optuna](Experiment/run_optuna_sfr_unet.md)。
> 原标量流程不变；3D/R_m 局部标签及其他 Exp 的逐格训练尚未接通。

> 本文档是本项目面向 AI、代码协作者和课题组成员的总入口。接手任务时应先阅读本文，再按“必读顺序”加载更详细的研究与代码文档。本文同时概述已核验的 MACER2D 盘星系流程与 MACER3D 矮星系数据/预处理流程；三维训练尚未接通，不能凭二维经验自行补写或宣称可用。

## 1. 项目目标

MACER 能够在单星系及星系核附近解析比典型宇宙学模拟更小的空间尺度，并计算恒星形成、黑洞吸积、AGN 风/喷流及其与星际介质的相互作用。典型宇宙学模拟不能直接解析这些尺度，因而依赖解析、经验或半经验亚网格模型。

本项目的长期目标是：

> 使用神经网络、神经算子或其他物理 AI，从宇宙学模拟可解析的低分辨率气体状态中推断未解析的小尺度物理，把 MACER 高分辨率模拟提供的知识以高效、稳定、守恒且可泛化的方式嵌入宇宙学模拟。

当前选定的目标宇宙学代码是 **GIZMO**，计划首先面向其 MFM 求解器建立适配层。MACER2D 是固定球极坐标 Eulerian 网格，GIZMO/MFM 使用随流体运动的近似固定质量单元；因此项目同时比较固定空间尺度粗粒化 \(\mathcal R_h\) 与质量自适应粗粒化 \(\mathcal R_m\)。目前还没有用于正式域迁移验证的真实 GIZMO snapshot，现阶段的 \(\mathcal R_m\) 只是理论性、守恒的近似表示，不能直接宣称等价于 GIZMO particle mass 或 MFM 邻域重构。

最终可能替代或修正的物理量包括：

- 恒星形成率（SFR）及新形成恒星质量；
- 黑洞吸积率（BHAR）；
- AGN 风和喷流的质量、动量与能量注入；
- 未解析冷致密气体比例、湍流能、冷却率等中间闭合量。

当前第一阶段聚焦于从粗粒化气体物理场预测 snapshot 的总瞬时 SFR，并比较两条路线：

1. `direct_sfr`：直接学习

   \[
   \widehat Y_\alpha=\mathcal F_\theta(\boldsymbol X_\alpha),
   \qquad
   \boldsymbol X_\alpha=\mathcal R_\alpha(\boldsymbol U_{\rm HR}).
   \]

2. `signed_closure`：先在粗粒化状态上计算物理基线

   \[
   B_\alpha
   =\mathcal S_\alpha(\boldsymbol X_\alpha),
   \]

   再学习可正可负的闭合修正

   \[
   C_\alpha^{\rm true}=Y_\alpha-B_\alpha,
   \qquad
   \widehat Y_\alpha=B_\alpha+\widehat C_\theta.
   \]

这里，\(\mathcal R_\alpha\) 是粗粒化算子，\(\mathcal S_\alpha\) 是作用在 LR 状态上的 MACER-like SFR 算子，\(Y_\alpha\) 是由 HR snapshot 中真实 SFR 字段保守归约得到的监督真值。

### 1.1 核心研究判断

本项目首先是一个**分辨率条件闭合问题**，不应被简化为唯一的 LR→HR 超分辨率问题：

\[
\boldsymbol U_{\rm LR}^{\rm sim}
\neq
\mathcal R_\alpha(\boldsymbol U_{\rm HR}^{\rm sim})
\]

通常成立。人工过滤的 HR 状态与真实低分辨率求解器产生的状态，会因数值扩散、时间步、几何、重力软化、反馈注入尺度和未闭合非线性源项而不同；同一 LR 状态也未必唯一决定 HR 微观结构。因此：

- 神经算子可以是后续场到场建模工具，但不是当前标量 SFR 任务的先验最优答案；
- 当前优先建立保守粗粒化、物理 baseline 和 signed closure；
- 最终部署模型所需输入必须能从目标宇宙学模拟中获得；
- 仅在 filtered-MACER 数据上表现良好，不等于已经证明可迁移到真实 GIZMO 状态。

完整研究背景、数学修正和长期路线见 [MACNET_RESEARCH_CONTEXT.md](MACNET_RESEARCH_CONTEXT.md)。

## 2. 必读文档与事实优先级

AI 开始修改代码前，应按下列顺序阅读：

1. 本文档：项目入口、共同术语和当前工作流；
2. [MACNET_RESEARCH_CONTEXT.md](MACNET_RESEARCH_CONTEXT.md)：研究背景、最终目标和闭合问题定义；
3. [COARSE_GRAINING_RM_RH_RESEARCH_PLAN.md](COARSE_GRAINING_RM_RH_RESEARCH_PLAN.md)：二维粗粒化、SFR 算子、metadata/UID、训练契约和历次验证；
4. 二维任务再读 [galaxy_data.md](raw_data_processing/galaxy_data.md)、[build_fine_from_raw.md](raw_data_processing/build_fine_from_raw.md)、[build_coarse_from_fine.md](raw_data_processing/build_coarse_from_fine.md) 以及相关脚本文档；
5. 三维任务必须先读 [3DSFRConvnet.md](3DSFRConvnet.md)，不能直接把二维假设移植到三维；
6. 判断哪些模型结果已完成可靠训练时只查 [Train_history.md](Train_history.md)，不要仅凭 `Results/` 中存在文件就认定实验有效。

当不同文档出现冲突时，使用以下优先级：

1. 当前可运行源码、数据文件内 metadata 契约和通过的测试；
2. `COARSE_GRAINING_RM_RH_RESEARCH_PLAN.md` 中日期最新的核验记录；
3. `3DSFRConvnet.md` 中日期最新的三维实现记录；
4. 本 README 和各脚本 `.md`；
5. 早期研究设想或公开数据 README。

若仍有冲突，不要自行选取方便的解释；应报告冲突、列出证据并请求确认。

## 3. 数据来源与适用范围

### 3.1 MACER2D 盘星系

盘星系模型、数值设置和物理背景来源于项目根目录中的：

- [Zou 等 - 2026 - A Systematic Study of AGN Feedback in a Disk Galaxy. I. Global Overview.pdf](<Zou 等 - 2026 - A Systematic Study of AGN Feedback in a Disk Galaxy. I. Global Overview.pdf>)；
- [readme_Zou.pdf](readme_Zou.pdf)：公开 snapshot、日志、字段编号、时间和代码单位说明。

Zou et al. 使用 ZEUS-MP/2，在球极坐标 \((r,\theta,\phi)\) 中进行二维轴对称 Eulerian 模拟。盘星系网格为 72 个均匀 \(\theta\) 单元与 280 个对数 \(r\) 单元，径向范围约为 100 pc–500 kpc，最小单元尺度约为 4 pc。数组在本项目中的标准顺序为 `(N_theta, N_r)`；二维轴对称不意味着没有方位速度，`v3` 仍表示 \(\phi\) 方向速度。

原始数据通常由以下内容组成：

- `hdfra.<index>`：HDF4 snapshot；
- 每个运行对应的文本日志；
- 项目根目录的 `data_loading.py`：数据提供方给出的读取示例。

公开 README 中部分字段描述已被后续源码核查修正，尤其不能直接把 Data-Set-5 当作数密度，详见第 4 节。

#### 3.1.1 `torus_100yr` 核区盘/环面域外测试集（2026-09-18 核验）

新增原始数据位于 `BaseConfig.raw_data_dir/DiskGalaxy/torus_100yr`，本机对应
`E:/科研项目与原始数据/ai+MACERsimulation/MACNet-v1/RawData/DiskGalaxy/torus_100yr`。
用户提供的信息是：模拟范围为 1–1000 pc，初始盘内流约为 1000 km/s，相邻 snapshot
间隔 100 yr，总演化时间约为 0.1 Myr；模拟作者说明这批数据包含与超爱丁顿状态有关的额外输出。

这批数据的用途被固定为**仅用于域外泛化测试**，不得进入训练集、验证集、训练归一化统计、
目标变换尺度估计或超参数选择。训练仍只使用 Zou 的五组盘星系数据。设计动机是把内边界从
Zou 数据的约 100 pc 推进至约 1 pc，从而检验模型面对小半径处密度、SFR 等量快速增长时的
外推能力。是否“没有新增物理过程”目前只有作者说明，snapshot 和日志本身不足以独立证明全部
源项、边界条件和反馈开关完全相同。

##### 数据身份与目录选择

- 科学数据在 `torus_100yr/torus_100yr/`：包含 `hdfra.000`–`hdfra.999` 共 1000 个真实
  HDF4 snapshot；同级的 `torus_100yr.log` 是约 421 MiB 的运行日志。
- `torus_100yr/__MACOSX/` 不是另一套模拟，也不是“超爱丁顿额外输出”。其中 1002 个文件
  全部是 176–276 byte 的 macOS AppleDouble/资源叉元数据，文件名形如 `._hdfra.000`，没有
  任何可读的科学 HDF4 snapshot。预处理和泛化测试必须使用 `torus_100yr/torus_100yr/`，并
  忽略 `__MACOSX/`。
- `.DS_Store` 是 macOS Finder 的 Desktop Services Store，用于保存目录显示方式、图标位置、
  标签等界面元数据，不是模拟数据；读取、清单和训练时应忽略。它和 `__MACOSX/` 都可以在
  数据副本中删除，但不需要改动原始归档。
- 作者所说的“超爱丁顿额外输出”不能据两个文件夹的名称解释为两套物理数据；若需要识别
  具体新增字段，仍需模拟输入文件、输出变量清单或作者确认。HDF4 中 `Data-Set-24`–`30` 也
  存在于抽查的 Zou snapshot，不能仅凭这些编号认定其为本批独有的超爱丁顿量。

##### 是否属于盘星系模拟

该数据使用与 Zou 数据相同的二维轴对称球极坐标 HDF4 结构：每个物理场 shape 为
`(1, 72, 280)`，同时保存 \(v_r,v_\theta,v_\phi\)，角网格与 Zou 数据逐点相同。初始 snapshot
中约 85.5% 的气体质量位于赤道面上下 \(5^\circ\) 内，且存在显著方位旋转，因此可以确认其
模拟对象是盘状/环面气体，而不是球对称流。但其径向域只有 1 pc–1 kpc，目录名也明确为
`torus`；更准确的分类是**盘星系核区的盘/环面模拟**，不能把它等同于 Zou 覆盖 ISM、CGM
和外边界宇宙学流入的完整 100 pc–500 kpc 盘星系模拟。

##### 单位、字段和转换核验

新数据与 Zou 数据的 `Data-Set-2`–`30` 编号、`long_name` 前缀和数组 shape 一致；核心字段继续
采用第 4 节的二维 MACER 代码单位，即 \(L_0=1\,\mathrm{kpc}\)、\(t_0=1\,\mathrm{Gyr}\)、
\(M_0=2.5\times10^7\,M_\odot\)。本批日志初值 `mbh=2` 对应
\(5\times10^7\,M_\odot\)，也支持质量单位未改变。应继续使用以下换算：

| 检查项 | `torus_100yr` 实测结果 | 与 Zou 数据的关系/结论 |
|---|---|---|
| 坐标 | `fakeDim2` 约为 0.00100034–0.980788 kpc；由端点外推边界约为 0.001–1.000 kpc；`fakeDim1` 为 72 个极角中心 | 长度仍以 kpc 保存；极角网格与 Zou 逐点相同 |
| 时间 | HDF `long_name` 时间从 0 到 \(9.99\times10^{-5}\) Gyr，步长严格为 \(10^{-7}\) Gyr = 100 yr；日志末端达到 \(10^{-4}\) Gyr | 时间单位仍为 Gyr；1000 帧覆盖 0–99,900 yr，0.1 Myr 是运行终点而不是最后一个 snapshot 的时间戳 |
| 速度 | `Data-Set-2/3/4` 为 \(v_r,v_\theta,v_\phi\)，原始单位 \(L_0/t_0\)；1 code velocity \(\simeq0.978\,\mathrm{km\,s^{-1}}\) | 初始 `Data-Set-2=-1000` 对应约 \(-978\,\mathrm{km\,s^{-1}}\)，与作者所说约 1000 km/s 一致；不要把数组数值机械当成精确 km/s |
| 密度与温度 | `Data-Set-5` 是质量密度；全部 989 个有效帧均满足 \(T=93e/\rho\)，相对误差低于 \(10^{-5}\) | 与 Zou/当前预处理契约一致；不能按 `shujujuzhen.py` 早期行尾注释把原始密度直接当成 cm⁻³ |
| 体积与气体质量 | 全部有效帧满足 `Data-Set-16 = Data-Set-5 * Data-Set-15`，相对误差低于 \(10^{-5}\) | `Data-Set-15` 仍是 kpc³，`Data-Set-16` 仍是代码质量 \(M_0\) |
| SFR | `Data-Set-11` 是逐单元已乘体积的瞬时 SFR 贡献；全网格求和后乘 0.025 得 \(M_\odot\,\mathrm{yr^{-1}}\) | 与 Zou 的 target reduction 和换算一致；HDF 的 `SFR (M/yr)` 标签不能替代代码单位换算 |
| 能量与冷却 | `Data-Set-10` 是内能密度，`Data-Set-17` 是 Gyr；有效帧沿用 \(e/\tau_{\rm cool}\) 的冷却率密度定义 | 与 Zou 相同；初始帧的 `coolt=0` 是输出初值，不应据此做除法 |

因此，`shujujuzhen.py` 的 DataSet→物理量映射可以复用，但其中把速度、密度和 SFR 简写成
`km/s`、`cm^-3`、`M_sun/yr` 的行尾注释不是原始数组的严格单位契约；应以本 README 第 4 节和
上述换算为准。

##### 与 Zou 五组数据的完整比较

这里的“Zou 五组”指当前训练实际使用的 `Fiducial`、`Fiducial4`、`Fiducial7`、`Model0.75`
和 `Model1.1`。论文直接给出 Fiducial/Model0.75/Model1.1 的模型参数；Fiducial4/7 的差异则按
本地 snapshot 与日志核验，不能把未提供的输入参数补猜出来。

| 比较项 | `torus_100yr` | Zou：Fiducial / Fiducial4 / Fiducial7 | Zou：Model0.75 / Model1.1 | 影响与注意事项 |
|---|---|---|---|---|
| 模拟对象 | 中央 1 kpc 内的核区盘/环面；二维轴对称球极坐标 | 完整盘星系 + ISM/CGM + 宇宙学流入；二维轴对称球极坐标 | 同左 | 新数据是相关但明显域外的核区问题，不应简称为与 Zou 等价的“第六个完整盘星系” |
| 数值代码与几何 | HDF 结构和字段表明沿用 MACER/ZEUS 风格二维球极网格；具体编译参数未随数据提供 | 论文确认 ZEUS-MP/2、Eulerian、二维轴对称 | 同左 | 结构一致不等于所有物理开关和边界条件一致 |
| 初始黑洞/恒星尺度 | 日志 `mbh=2`，即 \(5\times10^7M_\odot\)；初始气体强烈集中在赤道面 | Fiducial 日志 `mbh=2`；Fiducial4/7 分别为 `mbh=8/14`，即 Fiducial 的 4/7 倍；其初始 `dstar` 和 `Mstar_i` 也逐格为 Fiducial 的 4/7 倍，而初始气体密度和网格体积相同 | 两者 formal-start snapshot `hdfra.045` 字节级相同，初始 SMBH 为 \(5\times10^7M_\odot\) | Fiducial4/7 不只是随机重采样；它们改变了黑洞和恒星质量尺度。其余未随数据发布的参数保持“未核验” |
| Zou 论文中的共同星系初值 | 不适用；该批没有独立输入文件可核验 | Fiducial 论文初值：\(M_{\rm halo}=1.6\times10^{12}M_\odot\)，bulge/disk/gas 为 \(1.5/4.7/0.9\times10^{10}M_\odot\)，气盘 \(R_g=3.5\) kpc、\(z_g=0.325\) kpc，盘边界 \(R=24\) kpc、\(\lvert z\rvert=2\) kpc，CGM 初始质量 \(2\times10^{10}M_\odot\) | 与 Fiducial 共用同一 formal-start 状态 | 不应把这些全星系初值自动赋给 torus 数据 |
| 初始/边界流入 | 用户/作者给定约 1000 km/s；初始 HDF 中 \(v_r=-1000\) code units 的区域可见 | Zou 论文：总宇宙学流入约 \(100M_\odot\,\mathrm{yr^{-1}}\)，冷/热 60:40；冷流 \(v_r=-200\,\mathrm{km\,s^{-1}}\)，高角动量情形 \(v_\phi=200\,\mathrm{km\,s^{-1}}\) | 同左 | 新测试同时改变流入速度和边界域，失败不能只归因于“更小内半径” |
| 角动量输运参数 | 未提供输入文件，无法仅由 snapshot 唯一反演 | 论文确认 Fiducial 的 ν=0.55；Fiducial4/7 的 ν 尚未由独立输入文件核验 | 论文确认 ν=0.75 / 1.10 | torus 与训练域的参数差异需要作者输入文件才能闭合确认 |
| 径向范围 | 约 1 pc–1 kpc | 约 100 pc–500 kpc | 同左 | 内边界缩小 100 倍、外边界缩小约 500 倍；重叠区仅约 100 pc–1 kpc |
| 网格大小 | \(N_\theta\times N_r=72\times280\)，每场 20,160 cells | \(72\times280\) | \(72\times280\) | 数组 shape 相同，便于复用模型接口，但物理分辨率和坐标分布完全不同 |
| 极角网格 | 中心约 \(3.222^\circ\)–\(176.778^\circ\)，\(\Delta\theta\approx2.444^\circ\) | 与 torus 逐点相同 | 与 torus 逐点相同 | 均切除了极轴附近小锥区 |
| 径向空间分辨率 | 中心间距约 \(6.93\times10^{-4}\)–37.7 pc；内边界角向弧长约 0.0427 pc；径向拉伸不是 Zou 的同一对数坐标数组 | 中心间距约 3.87 pc–14.35 kpc；内边界角向弧长约 4.35 pc；论文称最小 cell 约 4 pc | 同左 | torus 的最内径向采样约比 Zou 细 \(5.6\times10^3\) 倍，内边界角向尺度约细 102 倍；应保存真实坐标/体积，不能只按 280 个索引归一化 |
| 模拟时间范围 | 0–0.1 Myr；snapshot 实际为 0–99,900 yr | 绝对时间从特殊 `hdfra.000=0` 跳至首个正编号约 0.56375 Gyr，末帧约 12.4875 Gyr；减去 0.5625 Gyr 后的 production 范围约 0–11.925 Gyr | `hdfra.045` 的绝对时间为 0.5625 Gyr，可定义 production \(t=0\)，末帧绝对时间约 12.4875 Gyr | torus 时间基线约短 \(1.19\times10^5\) 倍，只检验短时核区状态，不能检验 Gyr 级演化/淬火 |
| 时间分辨率 | 100 yr | 通常为 1.25 Myr；扁平化的 999 帧重启段边界存在 2.5 Myr 间隔，其他缺失文件形成更大真实间隔 | 12.5 Myr | torus 相对名义 cadence 分别细 12,500 倍和 125,000 倍；相邻帧高度相关，989 帧不能视为 989 个独立样本 |
| 实际 snapshot 可用性 | 1000 个文件，排除 11 个损坏帧后 989 个有效帧 | Fiducial/Fiducial4/Fiducial7 实际分别 8475/8816/7071 帧，索引到 9531，区间内分别缺 1057/716/2461 帧；公开说明也明确原始分段输出存在缺失 | 各 1000 帧，索引 000–999，无缺号 | 所有比较和 bootstrap 都要按真实文件清单与时间块处理，不能假设连续独立采样 |
| 初始松弛 | 未提供独立 relaxation 说明；`hdfra.000` 是用户所述初始内流状态 | `hdfra.000` 为绝对时间 0 的特殊初值；中间 0.5625 Gyr 未在 flat 正编号序列输出，`hdfra.001` 已是 post-relaxation | 0–44 为 0–0.55 Gyr，`hdfra.045=0.5625` Gyr 可作为 production 起点 | 分析层可另定义 production time，但日志匹配必须使用绝对时间；torus 纳入帧仍由显式排除清单决定 |
| 目标积分域 | `sum(Data-Set-11)` 只积分 1 pc–1 kpc | 积分 100 pc–500 kpc 全域 | 同左 | 两边“总 SFR”不是同一空间孔径；主结果必须明确这是不同 aperture 的域外预测，最好补充共同 100 pc–1 kpc 径向窗口的对照 |
| 超爱丁顿与反馈 | 作者说明存在超爱丁顿相关额外输出；具体字段/开关未由输入文件独立确认 | Zou 论文的 AGN 模型本身也包含超爱丁顿冷吸积分支 | 同左 | 不能据此声称加入了全新的物理过程，也不能声称设置完全相同；两种说法都需要输入文件或源码证据 |

##### 已知损坏帧与测试协议

对全部 1000 个 HDF4 文件逐帧检查了核心字段 `long_name`、NaN/Inf、正值性、
\(T=93e/\rho\) 和 `mgas=rho*vol`。异常帧恰好为以下 11 个，没有发现额外异常：

- step 747（74,700 yr）：SFR 出现负值，`vol/mgas` 等后续字段串写，温度和质量恒等式失效；
- step 754–755（75,400–75,500 yr）；
- step 763–770（76,300–77,000 yr）：step 754–755 和 763–770 这 10 帧的
  `Data-Set-4`–`17` 多个字段被错误写成 `2-VELOCITY`，并同时出现负密度、负能量、负 SFR
  和非正体积。

预处理必须在读取/归约前按原始 snapshot index 排除
`[747, 754, 755, 763, 764, 765, 766, 767, 768, 769, 770]`，不能只靠 SFR 阈值事后筛选。
这 11 帧约占 1.1%，直接删除对 0.1 Myr 序列覆盖影响较小，但对应时间缺口必须保留在 metadata。

正式泛化测试至少遵守：

1. 用 Zou 训练集确定全部输入标准化、target transform、closure scale、阈值与超参数，torus 标签不得参与；
2. 分别报告完整 torus 域和共同径向窗口 100 pc–1 kpc（若模型/预处理支持）的结果，避免把 aperture
   差异误判为模型误差；
3. torus 使用用户给定的 snapshot 排除清单；代码不自动建立第二套敏感性测试集。当前坏帧清单为
   `747, 754, 755, 763–770`，应在 `raw → fine` 前排除；
4. 保留 `source_run=torus_100yr`、原始 index、物理时间、坐标范围和坏帧清单，确保 test-only
   约束和结果可追溯；
5. 当前结论只证明字段/单位/代数关系兼容，不证明边界条件、黏滞、反馈参数和全部源项与 Zou 完全一致。

### 3.2 MACER3D 矮星系

矮星系数据对应论文 [Positive AGN Feedback Enhances Star Formation in Starburst Dwarf Galaxies](https://doi.org/10.3847/1538-4357/ae6f12)，使用 MACER3D 在三维球坐标中模拟盘状矮星系。数据说明列出 `fullFB`、`SNonly`、`fullFB_inflow` 和 `SNonly_inflow` 四种反馈/流入配置；截至 2026-09-11，项目实际挂载并完成核验的只有 `fullFB_inflow`。

当前运行在配置中的名称为 `dwarf_galaxy_fullFB_inflow`，包含 `00050–00299` 共 250 个 ATHDF snapshot。单个文件约 216 MiB，全局网格为 `(N_r,N_theta,N_phi)=(256,64,128)`，实际 `MeshBlockSize=(16,16,16)`，无 AMR。Athena block 数组顺序是 `(phi,theta,r)`，项目统一转换为 `(r,theta,phi)`；网格、坐标范围和 MeshBlock 信息必须以 ATHDF 属性与 `x1f/x2f/x3f` 为准。

三维流程当前的字段边界是：

- 默认模型候选输入：`rho、press、vel1、vel2、vel3`；
- Residual baseline 专用：`cool_rate`，它是单位体积冷却率；
- 监督目标：`rhodotsf`，总瞬时 SFR 为 `sum(rhodotsf * cell_volume)`；
- `rhodotsf/sfr_density_true/star/newstar` 不得进入模型输入；
- `metallicity` 用于温度重算和后续冷却物理，是否作为模型输入应通过独立消融决定。

MACER3D 的质量、长度和时间代码单位约为 `2.5008e7 Msun`、`1 kpc` 和 `1.000017 Gyr`，一个 code SFR 约为 `0.025 Msun/yr`。ATHDF 保存 pressure；内能密度为 `press/(gamma-1)`。温度使用作者提供的 `mu(Z)` 关系，当前系数约为 `71.4*press/rho`，不能复用二维 `T=93e/rho`。

普通 snapshot 不含引力字段；根目录 `Fid_yuan18.out1.00000.athdf` 中的 `grav` 已确认为可跨时间复用的静态总径向引力加速度。独立三维 SFR 算子和 Direct/Residual 守恒降采样已经实现，但三维 dataset、CachePT、`SFRConvNet3D`、训练和结果分析尚未完成。

> 三维字段全集、代码单位、温度与数密度推导、SFR 算子验证、引力限制、降采样公式、真实数据审计和最新实现状态统一见 [3DSFRConvnet.md](3DSFRConvnet.md)。任何三维修改都必须先完整阅读该文档，不能根据本 README 的摘要补猜物理语义。

## 4. 二维代码单位与物理量语义

### 4.1 基本代码单位

盘星系流程默认采用：

| 基本量 | 代码单位 | 补充换算 |
|---|---|---|
| 长度 | \(L_0=1\,{\rm kpc}\) | — |
| 时间 | \(t_0=1\,{\rm Gyr}\) | — |
| 质量 | \(M_0=2.5\times10^7\,M_\odot\) | — |
| 速度 | \(L_0/t_0\) | 约 \(0.978\,{\rm km\,s^{-1}}\) |
| 总 SFR | \(M_0/t_0\) | \(1\) code SFR = \(0.025\,M_\odot\,{\rm yr^{-1}}\) |
| 密度 | \(M_0/L_0^3\) | \(2.5\times10^7\,M_\odot\,{\rm kpc^{-3}}=0.025\,M_\odot\,{\rm pc^{-3}}\) |

二维数据中的速度、加速度、内能密度和冷却率密度也沿用由这三个基本单位组合出的代码单位。温度使用 K，角坐标使用 rad。

### 4.2 已核验的关键字段

| 原始/processed 字段 | 物理含义 | 典型单位或定义 |
|---|---|---|
| `fakeDim2` / `r` | 球坐标半径 | kpc |
| `fakeDim1` / `theta` | 球坐标极角 | rad |
| Data-Set-2 / `v1` | 径向速度 \(v_r\) | kpc/Gyr |
| Data-Set-3 / `v2` | 极角速度 \(v_\theta\) | kpc/Gyr |
| Data-Set-4 / `v3` | 方位速度 \(v_\phi\) | kpc/Gyr |
| Data-Set-5 / `density` | **气体质量密度** \(\rho\)，不是 processed 流程中的数密度 | \(M_0/{\rm kpc^3}\) |
| Data-Set-6 / `temperature` | 气体温度 | K；二维数据满足 \(T=93e/\rho\) |
| Data-Set-8 / `radial_acceleration` | 引力加速度的 \(r\) 分量 | kpc/Gyr² |
| Data-Set-9 / `theta_acceleration` | 引力加速度的 \(\theta\) 分量 | kpc/Gyr² |
| Data-Set-10 / `gas_energy` | **气体内能密度** \(e\)，不是单元总能量，也不是比内能 | \(M_0/({\rm kpc\,Gyr^2})\) |
| Data-Set-11 / `sfr` | 每个网格已乘体积的瞬时恒星形成率贡献 | 对所有网格求和得到 snapshot 总 SFR，单位 \(M_0/{\rm Gyr}\) |
| Data-Set-13 / `initial_stellar_density` | 初始恒星质量密度 | \(M_0/{\rm kpc^3}\)；默认从模型输入排除 |
| Data-Set-15 / `grid_volume` | 权威网格单元体积 | kpc³ |
| `gas_mass` | 网格气体质量 | `density * grid_volume`，单位 \(M_0\) |
| Data-Set-17 / `cooling_time` | 冷却时标 | Gyr |
| `cooling_rate_density` | 单位体积冷却功率密度 | `gas_energy / cooling_time` |
| `sfr_target` / \(Y_\alpha\) | HR Data-Set-11 保守归约后的真实总 SFR | \(M_0/{\rm Gyr}\) |
| `sfr_macer_baseline` / \(B_\alpha\) | MACER-like SFR 算子作用于当前 LR 状态的总 SFR | \(M_0/{\rm Gyr}\) |
| `sfr_closure_true` / \(C_\alpha^{\rm true}\) | signed closure 真值 | \(Y_\alpha-B_\alpha\)，可正可负 |

必须遵守的语义：

- 尽管公开 `readme_Zou.pdf` 把 Data-Set-5 描述为 gas number density，项目后续源码核查确认 `density` 是质量密度；恒星形成密度阈值中才使用 \(n=1.02\rho\)。
- `gas_energy` 是内能**密度**。总内能应计算为 \(\sum_i e_iV_i\)，不能再把它当作单元能量直接相加。
- Data-Set-11 不是全局标量，也不是仍需乘体积的 SFR 密度；其每格值已经包含体积因子，总 SFR 是网格求和。
- 真值 \(Y_\alpha\) 直接由 Data-Set-11 归约；在 HR 场重算 SFR 算子的用途是验证/诊断，不是替代真值。
- `grid_volume` 必须优先使用 Data-Set-15，不能用规则数组索引面积代替球坐标真实体积。
- 日志 `NEWstar` 是累计新形成恒星质量，不是训练用的瞬时 SFR。

### 4.3 二维 MACER SFR baseline

代码实现位于 [macer_sfr_operator.py](raw_data_processing/macer_sfr_operator.py)。当前 canonical 数据兼容版本使用：

\[
\dot\rho_\star
=c_\star\rho\,f_{\rm step}(\Delta t,\tau_{\rm SF}),
\qquad
\tau_{\rm SF}=\max\!\left[\tau_{\rm cool},\min(\tau_{\rm ff},\tau_{\rm rot})\right],
\]

\[
\tau_{\rm ff}=\sqrt{\frac{3\pi}{32G\rho}},
\qquad
T=\frac{93e}{\rho},
\]

并施加低温和密度阈值。当前默认参数是 `G=112.3`、`eta_SF=0.01`，旋转时标暂用黑洞主导的 enclosed-mass 近似。论文物理对照版本使用 `eta_SF=0.1` 和加速度形式的旋转时标。两者都应保留；canonical baseline 的近似来源和版本必须写入 metadata，待取得完整运行源码/参数后再更新。

粗粒化冷却量时，先令 \(C_i=e_i/\tau_{{\rm cool},i}\)，再按体积守恒总冷却功率：

\[
C_C=\frac{\sum_{i\in C}C_iV_i}{V_C},
\qquad
\tau_{{\rm cool},C}=\frac{e_C}{C_C}.
\]

这里守恒的是 \(\sum C_iV_i\)，而不是把冷却时标直接平均。

## 5. 目录与职责

| 路径 | 职责 |
|---|---|
| `.config/` | 本机路径、原始运行清单、字段映射、目标语义和训练配方 |
| `raw_data_processing/` | 日志解析、raw→fine、fine→coarse、SFR 算子、粗粒化和 metadata 校验 |
| `data/` | processed Parquet、fine/coarse HDF5 和部分诊断数据；不要把不同契约产品混入同一子目录 |
| `Experiment/` | CachePT 构建、PyTorch dataset、模型、训练、Optuna 和性能工具 |
| `.cache/` | `train.pt`、`val.pt`、`test.pt`、归一化、metadata、split manifest 和 Optuna 数据库 |
| `Results/` | 正式训练输出、模型权重与配置 |
| `runs/` | TensorBoard 日志 |
| `VisualScripts/` | 结果分析与可视化；分析前先核对所读模型和 cache 契约 |
| `tests/` | 数据语义、I/O、粗粒化、训练契约和回归测试 |

`.config/BaseConfig.json` 包含机器相关的绝对路径。把项目同步到另一台服务器后，应先修改 `workspace_root`、`raw_data_dir`、`data_dir` 和 `results_dir`，不要把本机盘符路径视为可移植配置。

## 6. 二维端到端工作流

### Step 0：阅读与配置

1. 完整阅读本 README、`MACNET_RESEARCH_CONTEXT.md` 和 `COARSE_GRAINING_RM_RH_RESEARCH_PLAN.md`；
2. 核对 `.config/BaseConfig.json` 的本机路径；
3. 核对 `.config/RawDataConfig.json` 中运行目录、日志名、snapshot 范围和索引模式；
4. 核对 `.config/HdfraConfig.json` 的字段映射和 Data-Set-11 目标契约；
5. 激活包含 `numpy`、`pandas`、`pyarrow`/Parquet 引擎、`h5py`、`pyhdf`、`torch` 的环境。

Fiducial/Fiducial4/Fiducial7 的 flat 文件来自覆盖式分段输出，不能直接用
`global_index*snapshot_dt` 作为日志绝对时间。配置中的 `time_mapping` 已显式处理
index 0、0.5625 Gyr 的正编号起点偏移，以及每 999 个扁平编号对应 1000 个原始时间步的
重启段规则。raw→fine 会用 HDF 的 `AT TIME=` 标签进行打印精度内核验。

### Step 1：解析日志

```powershell
python raw_data_processing/parse_log_file.py
```

输出为 `BaseConfig.data_dir/<galaxy_name>.parquet`。脚本当前会重新解析并覆盖对应 Parquet。盘星系 SFR 真值不来自日志。

详细接口见 [parse_log_file.md](raw_data_processing/parse_log_file.md)。

### Step 2：原始 HDF4 → fine HDF5

```powershell
python raw_data_processing/build_fine_from_raw.py `
  --galaxy disk_galaxy_fiducial `
  --fine-dirname fine_sfr_v6
```

建议先用 `--snapshot-index` 或 `--max-snapshots` 在独立目录 smoke test。二维 fine 文件应包含 `/snapshot`、`/coordinates`、`/geometry`、`/log_data`、`/metadata/contract_json` 和 `/sfr_supervision`。不要继续使用缺少 metadata 与 supervision 的旧 format-v4 fine 文件。

三个 Fiducial flat 运行若已有完整 `fine_sfr_v6`，可使用
[二维 fine 时间上下文迁移](raw_data_processing/migrate_fine_time_context.md) 写入独立的
`fine_sfr_v7_timefix`，保留全部物理场和 Y，仅修复时间/日志并重算监督产品。迁移已经通过少量
真实 raw→fine/coarse 对照；尚未全量执行。仅改时间标签或从旧 fine 重新粗化不能完成修复。

### Step 3：fine HDF5 → coarse HDF5

固定空间块 \(R_h\) 8×8 示例：

```powershell
python raw_data_processing/build_coarse_from_fine.py `
  --galaxy disk_galaxy_fiducial `
  --fine-dirname fine_sfr_v6 `
  --coarsener spatial_block `
  --coarse-shape 8 8 `
  --coarse-dirname coarse_rh_8x8_sfr_v6
```

该阶段保存保守粗粒化场、守恒诊断、\(Y_\alpha\)、\(B_\alpha\)、\(C_\alpha^{\rm true}\)、UID 和完整派生契约。不同分辨率、算子或参数必须使用不同输出目录。

支持的二维算子：

- `spatial_block`：固定规则空间块 \(R_h\)；
- `axisymmetric_rm_v0`：核支撑图 + Sinkhorn/IPFP 的质量自适应 \(R_m\)-v0；
- `axisymmetric_rm_v1`：预计算邻居、自适应局部输运图、Wendland-C2 核和加速平衡。

### Step 4：coarse HDF5 → CachePT

Direct 示例：

```powershell
python Experiment/prepare_data.py `
  --experiments exp1 `
  --resolution-dirname coarse_rh_8x8_sfr_v6 `
  --cache-tag Rh8x8_Direct `
  --objective-mode direct_sfr `
  --exclude-raw-snapshot-index 0 `
  --oversample=false
```

Signed closure 示例：

```powershell
python Experiment/prepare_data.py `
  --experiments exp1 `
  --resolution-dirname coarse_rh_8x8_sfr_v6 `
  --cache-tag Rh8x8_Closure_auto `
  --objective-mode signed_closure `
  --closure-transform signed_asinh `
  --closure-scale auto `
  --exclude-raw-snapshot-index 0 `
  --oversample=false
```

输出位于项目根目录 `.cache/`。`closure-scale auto` 只使用训练集的 `closure_true_physical` 估计尺度，然后重新生成正式 cache，不读取验证/测试标签统计量。

当前 `exp1` 是组内随机 snapshot 60/20/20 划分，仅适合参考和代码验证；相邻 snapshot 高度相关，不能把它作为严格时间外推证据。正式泛化实验需要 time-block、run-block 或 leave-one-galaxy-out 设计。

### Step 5：超参数搜索

```powershell
python Experiment/run_optuna.py `
  --data-exp Exp1_Rh8x8_Closure_auto_ `
  --objective-mode signed_closure `
  --loss l1 `
  --n-trials 100 `
  --timeout 28800
```

`run_optuna.py` 不读取 `.config/ExperimentSetting.json` 来决定本次 cache 的目标；目标和条件契约来自 cache metadata，并由命令行显式核对。HPO 只用验证集选择超参数，默认不保存正式最优模型，也不应在每个 trial 上评估测试集。数据库可续跑，但 cache 指纹、目标、loss 和搜索空间契约不一致时必须新建 study/database。

### Step 6：用最优参数正式训练

把选定超参数写入 `.config/ExperimentSetting.json` 的独立实验配置，再运行：

```powershell
python Experiment/run_train.py <完整实验配置键>
```

也可以直接调用 `Experiment/train.py`，但所有参数都必须显式记录。Optuna 完成后仍需用最优参数重新正式训练，最好运行多个随机种子；HPO trial 的验证结果本身不能替代最终训练与测试评估。

### Step 7：分析与记录

1. 只对正式训练保存的 checkpoint 做 test/out 分析；
2. 同时报告模型、MACER-LR baseline 和其他基线的物理空间指标；
3. 区分 transformed target-space 指标和 physical-space 指标；
4. 将确认可靠的实验条件与结论登记到 `Train_history.md`；
5. 不在本文重复训练历史，防止两份记录漂移。

共享分析接口见 [analysis_utils.md](VisualScripts/ResultAnalysis/analysis_utils.md)。它负责按 checkpoint/cache 契约加载模型、执行 direct 或 closure 推理、把输出还原到物理 SFR 空间并计算指标。`exp2` 的 leave-one-galaxy-out 结果应使用 [analyze_exp2.py](VisualScripts/ResultAnalysis/analyze_exp2.py)：

```powershell
python VisualScripts/ResultAnalysis/analyze_exp2.py --all-folds
```

其参数、四类评估域和一致性检查见 [analyze_exp2.md](VisualScripts/ResultAnalysis/analyze_exp2.md)。`VisualScripts/DataAnalysis/` 用于数据本身和时间序列诊断；`VisualScripts/ResultAnalysis/` 中的 legacy notebook 只应在确认其 cache、mirror、目标变换和 checkpoint 契约兼容后使用。

## 7. 可控制的研究变量

### 7.1 数据与粗粒化

| 变量 | 主要入口 | 可选值/含义 |
|---|---|---|
| 运行/星系 | `--galaxy` | 可重复指定配置中的运行，例如 `disk_galaxy_fiducial` |
| fine 输入目录 | `--fine-dirname` | 隔离不同版本的 fine 产品 |
| 粗粒化算子 | `--coarsener` | 二维：`spatial_block`、`axisymmetric_rm_v0`、`axisymmetric_rm_v1` |
| 输出网格尺寸 | `--coarse-shape` | 二维顺序为 `N_theta N_r`，如 `8 8`、`16 16`；必须满足算子约束 |
| 三维粗粒化 | `--coarsener spatial_3d` | 非重叠球坐标索引块；当前默认 `(N_r,N_theta,N_phi)=(32,16,32)` |
| 三维预测模式 | `--prediction-mode` | `direct` 或 `residual`；决定是否在 processed 产品中携带 baseline 必要字段 |
| 质量尺度 | `--mass-filter` | \(R_m\) 必需，单位为 MACER 质量代码单位；\(R_h\) 禁止伪造 |
| 核函数 | `--kernel` | v0：`top_hat` 或 `cubic_spline`；v1 固定 `wendland_c2` |
| 平衡与支撑参数 | coarse builder | 初始邻居数、平衡容差、最大迭代数、v1 邻居/运行时间上限 |
| snapshot 子集 | `--snapshot-index` / `--max-snapshots` | 用于 smoke test、诊断或受控抽样 |

修改分辨率不能只改 CNN 输入 shape；应重新生成 coarse HDF5 和 CachePT，并重新核验守恒、位置编码、网络下采样层数、batch size 和显存。

### 7.2 监督目标与变换

| 变量 | 可选值/规则 |
|---|---|
| `objective_mode` | `direct_sfr` 或 `signed_closure` |
| direct 目标变换 | `log10_1p` 或 `log10_floor`；尺度和 floor 必须写入 metadata |
| closure 变换 | `signed_asinh` 或 `identity` |
| closure scale | 正数或 `auto`；`auto` 只能由训练集确定 |
| 负 SFR 惩罚 | `negative_sfr_weight >= 0`；约束 \(B+\widehat C<0\) 的非物理预测 |
| 零 SFR | 默认保留；`--drop-zero-targets` 只用于有充分物理理由的消融 |
| 输入通道 | `--exclude-channels` 等接口；目标、baseline 和 closure 严禁混入输入 |

Direct 与 closure 的公平比较必须共享相同 coarse 数据、通道、split/UID、随机种子、模型容量、HPO 预算和评价指标。若一个使用 MSE、另一个使用 L1，则结果同时混入“目标路线”和“loss”差异，不能直接归因于 closure。

### 7.3 数据划分

`prepare_data.py` 当前提供：

- `exp1`：盘星系组内随机 snapshot 划分；
- `exp2`：leave-one-galaxy-out，多折输出；
- `exp3`：五组 Zou 运行按组分层后随机做 75% train / 25% validation，torus 是唯一 test-only 数据；
- `exp4`：旧版单星系留出兼容方案；
- `exp8`：原 exp3，删除 low 运行后的随机 60/20/20 划分。

Exp3 的 CachePT 不做运行过采样。训练 loader 使用“每个运行总权重相同、运行内部按物理
时间间隔加权”的抽样；validation 与 torus test 不重采样。所有输入归一化统计、通道
`log1p` 掩码和 target/closure 变换参数只从 Zou train split 得到。配置中的
`role=test_only` 同时阻止 torus 被其他默认盘星系训练入口加载。

Exp3.1 / Exp3.2 是 Exp3 的几何输入消融：保留原相对 Fourier 编码，分别追加固定
`log10(r/kpc)`，以及再追加 `log10(Δr/kpc)`、`log10(rΔtheta/kpc)`。
两者使用独立 cache schema 6，仍只用 Zou train 估计标准化，Direct/Closure 均支持。
入口为 `run_train.py --experiments exp3.1` / `exp3.2`（默认 Closure）；Direct 使用
`exp3.1-direct` / `exp3.2-direct`。先准备对应几何 cache，且正式实验应使用日志时间
修复后的统一 coarse 产品。完整本地/服务器命令见
[Exp3 几何尺度消融说明](Experiment/exp3_geometry_ablations.md)。

当前按用户决定保留随机 snapshot split，因此 Zou train/validation 仍存在相邻时间帧跨 split
的风险；Exp3 的核心泛化结论应来自完全独立的 torus test，而不应把 validation 当成严格
时间外推证据。后续若研究时间泛化，再另行使用：

- 连续 time-block split，并在块间设置时间 gap；
- run-level split；
- leave-one-galaxy/configuration-out；
- 多随机种子重复和分组 bootstrap 置信区间。

当前位置编码也限制了 Exp3 的物理解读：`FourierPositionalEncoding2D` 对每个样本分别将
`log(r)` min-max 到 `[0,1]`。因此 Zou 的约 100 pc–500 kpc 与 torus 的 1 pc–1 kpc 都被
映射到相同归一化范围，网络直接看到的是各自网格内的相对径向位置，而不是统一的绝对 pc
尺度。现阶段按本轮决定保留该编码；若 torus 表现异常，不能把误差只归因于流体物理域差异，
还必须把绝对半径信息丢失列为可能原因。

### 7.4 模型、优化与系统

训练入口支持的主要模型类型为 `transformer`、`resnet`、`resfilm` 和 `convnet`。当前二维 SFR 主模型为 `SFRConvNet`。主要可变参数包括：

- 网络：`base_ch`、`stage_depths`、`se_reduction`、Fourier 位置编码频带与最大频率、dropout；
- 条件：固定单一 \(R_h\) 使用 `conditioning=none` 且必须关闭 FiLM；\(R_m\) 使用活动的 `log10(mass_filter)` 条件时可启用 FiLM；
- 优化：batch size、epochs、AdamW 学习率、weight decay、warmup、cosine 最低学习率、gradient clipping；
- 损失：`mse`、`l1`、`huber`，以及 closure 的负最终 SFR 惩罚；
- 系统：AMP、cuDNN benchmark、DataLoader workers、`standard`/`tensor_cpu` batch backend、是否写模型图。

各参数的当前默认值和限制应查对应脚本文档，而不要从旧命令复制：

- [prepare_data.md](Experiment/prepare_data.md)
- [models.md](Experiment/models.md)
- [train.md](Experiment/train.md)
- [run_train.md](Experiment/run_train.md)
- [run_optuna.md](Experiment/run_optuna.md)

## 8. 数据与训练契约

### 8.1 processed HDF5

二维新产品使用 format-version 6；三维产品使用独立的 `macnet_3d_format=1`。两者都必须包含完整 `metadata/contract_json`，并保存：

- 数据来源与 snapshot 身份；
- 字段语义、单位和目标定义；
- 粗粒化算子及全部参数；
- `parent_hr_uid`：来源 HR snapshot 身份；
- `sample_uid`：当前物理产品身份；
- 守恒诊断；
- canonical \((Y,B,C^{\rm true})\) 和恒等式 \(Y=B+C^{\rm true}\)。

不同 operator、shape、mass filter、kernel、物理算子版本或字段契约的产品不能写入同一目录。禁止 coarse→coarse 作为默认生产流程。

### 8.2 CachePT

CachePT 至少应带有：

- 输入张量、坐标和训练集归一化统计量；
- 目标的物理空间值和变换空间值；
- objective/transform/scale/units 契约；
- 条件变量语义与活动状态；
- galaxy/run 分组；
- `sample_uid`、`parent_hr_uid` 和 split metadata。

`run_train.py` 与 `run_optuna.py` 启动时打印出的 `Resolved learning objective`、`Target metadata`、`Conditioning contract` 和输入通道必须与预期一致。实验名中出现 `Closure` 或 `Direct` 不能代替 metadata 校验。

## 9. 可靠性、物理与泄漏检查

每次新增数据产品、分辨率或训练路线，至少检查：

1. 质量、三个方向动量、内能和总冷却功率的守恒误差；
2. `grid_volume > 0`，坐标、字段和体积 shape 一致；
3. 所有输入、target、baseline 和 closure 的单位一致；
4. \(Y=B+C^{\rm true}\) 在浮点容差内成立；
5. 训练、验证、测试和 out 的 UID 无交叉；
6. 标准化和 `closure_scale` 只由训练集估计；
7. Data-Set-11、未来时刻量、标签派生量和 baseline 不出现在模型输入通道；
8. 固定 \(R_h\) 不错误启用 resolution FiLM，\(R_m\) 条件值不缺失；
9. 零 SFR、末期极低 SFR、爆发期、AGN 活跃/宁静期均被单独检查；
10. 先做少量 snapshot smoke test 和 HDF5 round-trip，再批量生成；
11. Optuna 只看验证集，确定超参数后才做正式多种子训练和一次性 test/out 评估；
12. filtered-MACER 测试与真实宇宙学 snapshot 域外测试必须分开报告。

旧的 mirror 和 `SFRSpatialAugmentor` 不满足当前球极网格的严格物理对称性，正式二维 SFR 流程默认禁用。不要把 8×16 mirror legacy cache 与原生 8×8 产品直接比较。

## 10. 已完成训练与结果解释

已经完成且经确认可依赖的训练历史，只维护在 [Train_history.md](Train_history.md)。本 README 不重复列出实验和数值结果。

在解释结果时：

- physical RMSE/MAE/R² 表示代码物理 SFR 空间的误差；
- target-space RMSE/MAE/R² 表示经过 `log10_1p` 或 `signed_asinh` 等变换后的误差；
- relative error 对极低/零 SFR 很敏感，必须注明非零筛选与 floor；
- 总体高 R² 不代表爆发期、末期低 SFR 或未见运行上都可靠；
- 与 MACER-LR baseline 比较是最低要求，但不是跨宇宙学代码泛化证明。

## 11. 三维矮星系流程入口

当前已完成的三维数据流程是：

```text
ATHDF + input.txt [+ 静态 grav]
  → build_fine_from_raw.py
  → macnet_3d_format=1 fine HDF5
  → build_coarse_from_fine.py --coarsener spatial_3d
  → Direct 或 Residual coarse HDF5
```

默认 `(32,16,32)` Direct 示例：

```powershell
python raw_data_processing/build_fine_from_raw.py `
  --galaxy dwarf_galaxy_fullFB_inflow `
  --prediction-mode direct `
  --fine-dirname fine_3d_direct

python raw_data_processing/build_coarse_from_fine.py `
  --galaxy dwarf_galaxy_fullFB_inflow `
  --coarsener spatial_3d `
  --prediction-mode direct `
  --fine-dirname fine_3d_direct `
  --coarse-shape 32 16 32 `
  --coarse-dirname coarse_3d_32x16x32_direct
```

Residual 使用相同入口，但 raw→fine 和 fine→coarse 都必须显式设置 `--prediction-mode residual`，并使用独立目录。Residual fine 可以派生 Direct coarse；Direct fine 缺少 `cool_rate/grav`，不能派生 Residual coarse。三维时间直接读取 ATHDF metadata，因此不需要运行 `parse_log_file.py`。

三维降采样以真实 cell faces 计算/加和体积，守恒质量、内能、金属质量、三个笛卡尔动量和总 SFR；Residual 另审计总冷却功率和质量加权径向量。速度先转换到固定笛卡尔基底守恒动量，再保存 coarse 笛卡尔速度及投影后的 `vel1/2/3`，不能直接平均球坐标速度分量。

raw→fine 默认安全续跑，并使用 `--source-hash stat` 避免为每个大型 ATHDF 额外完整读盘；需要归档级强校验时可显式使用 `--source-hash sha256`。fine 的主要价值是复用同一次原始读取来尝试不同 coarse shape/mode；确认 coarse 可追溯且守恒后是否归档 fine，应根据磁盘成本决定。

当前二维 `GalDataSet`、CachePT、`SFRConvNet`、训练入口和 Results notebook 均不能直接处理三维产品，也不应为了三维流程重新耦合到二维 `galaxy_data.py`。后续应建立独立的三维 dataset/cache 层和 `SFRConvNet3D`，并保持二维类、cache schema 和 checkpoint 行为不变。目前只有一条 `fullFB_inflow` 时间序列，随机 snapshot split 只能作为乐观的同序列插值参考，正式评估需要 time-block/contiguous split 和时间隔离。

实现文件包括 `athdf_reader.py`、`galaxy_data_3d.py`、`coarsening_spatial_3D.py`、`macer_sfr_operator_3d.py` 以及两个统一 build 入口。完整接口、公式、验证数值、已知限制和后续训练设计见 [3DSFRConvnet.md](3DSFRConvnet.md)，该文档是三维任务的权威说明。

## 12. AI 修改项目时的最低工作规范

1. 先阅读任务相关 `.md`、当前源码和测试，再提出或实施修改；
2. 物理语义不确定时先检查数据属性、源码和 metadata，不根据变量名猜测；
3. 不覆盖用户已有数据或混合不同契约；测试使用独立目录和少量 snapshot；
4. 修改粗粒化、SFR 算子或训练目标时，同步更新 metadata、测试、脚本文档和研究计划；
5. 保留 direct-SFR 接口，同时维护 signed-closure 路线；
6. 对新算法做守恒性、数值稳定性、标签泄漏和对抗性边界检查；
7. 代码注释与项目文档主要使用中文，专有名词和公式可保留英文；
8. 不把尚未训练、仅 smoke-test 或仅存在 checkpoint 的实验写入 `Train_history.md`；
9. 不把二维结论自动外推到三维，也不把 filtered-HR 结论自动外推到 GIZMO；
10. 所有重要结论应注明证据来自论文、数据 README、源码、metadata、测试还是经验推断。
