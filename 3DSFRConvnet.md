# 3DSFRConvnet：MACER3D 矮星系 SFR 模型设计与实现记录

> 文档用途：统一记录 MACER3D 矮星系数据读取、守恒降采样、物理 baseline、3D CNN、训练与结果分析的设计、实现状态、验证结果和待确认问题。  
> 建立日期：2026-09-04  
> 当前状态：ATHDF 选择性读取器和字段配置已经完成；三维守恒降采样、3D CNN、训练缓存和结果分析尚未实现。  
> 维护约定：以后凡是对矮星系三维训练流程做出影响数据语义、接口、文件格式、模型结构、训练目标、划分方式或评估口径的重大修改，都必须同步更新本文档末尾的“变更日志”。

---

## 1. 项目目标与当前路线

本项目的长期目标不是只拟合某一条模拟的总 SFR 历史，而是从高分辨率 MACER 模拟中学习低分辨率状态下缺失的恒星形成闭合关系，并最终嵌入宇宙学模拟。

针对 MACER3D 矮星系数据曾讨论过两条路线：

1. 对三维数据做方位角平均，将其变成类似 MACER2D 的二维轴对称数据，再与盘星系数据共同训练二维模型；
2. 保留完整三维结构，在现有 `SFRConvNet` 思路上构建独立的 `3DSFRConvNet`，先训练矮星系数据，未来再接入三维盘星系数据。

当前实施路线为方案二。方案一可以作为信息损失和模型复杂度的对照实验，但不能替代三维主流程，因为方位角平均会消除非轴对称结构、局部团块、湍流以及方位角方向上的反馈传播。

需要进一步明确最终输出层级：

- **阶段性工程目标**：先完成“单个 snapshot → 瞬时总 SFR”的三维标量回归，用于贯通读取、降采样、缓存、训练和分析流程；
- **最终物理目标**：模型应输出 coarse cell 上的局部 SFR 密度或闭合修正场，再对全域积分得到总 SFR。只有局部输出才适合嵌入宇宙学模拟的逐流体单元恒星形成模块。

因此，标量模型只能视为流程验证和全局诊断模型，不能直接等同于最终可部署的局部闭合模型。

---

## 2. 当前可用数据与已核验事实

### 2.1 数据来源

数据位于：

```text
E:/科研项目与原始数据/ai+MACERsimulation/MACNet-v1/RawData/DwarfGalaxy
```

数据说明列出了以下模拟类型：

- `fullFB`：AGN feedback + SN feedback，无 cosmological inflow；
- `SNonly`：仅 SN feedback，无 cosmological inflow；
- `fullFB_inflow`：AGN feedback + SN feedback，有 cosmological inflow；
- `SNonly_inflow`：仅 SN feedback，有 cosmological inflow。

截至 2026-09-04，实际挂载并完成检查的是 `fullFB_inflow`，共有 250 个 `.athdf` snapshot，编号为 `00050` 至 `00299`。当前不能把论文中存在的其他模拟视为已经可训练的数据。

### 2.2 单个 snapshot 的实际结构

已检查文件：

```text
fullFB_inflow/yuan18.out1.00051.athdf
```

实际结构为：

- 文件大小：226,716,288 bytes，约 216.21 MiB；
- 坐标：`spherical_polar`；
- 全局网格：`(256, 64, 128)`，项目内部解释为 `(r, theta, phi)`；
- 总 cell 数：2,097,152；
- 实际 `MeshBlockSize`：`(16, 16, 16)`；
- MeshBlock 数量：512；
- `MaxLevel = 0`，当前数据没有 AMR refinement；
- `hydro` 数据集形状：`(27, 512, 16, 16, 16)`；
- 原始 dtype：`float32`。

`input.txt` 中写有 `MeshBlockSize=(32,32,32)`，与实际 ATHDF 属性不一致。原作者已经确认实际模拟输出使用的是 `(16,16,16)`。所有读取和重建仍必须以每个 ATHDF 文件属性为准。

径向方向为近似对数网格，范围约为 0.007–75 kpc；方位角覆盖完整的 (2\pi)。论文、`input.txt` 和实际 cell faces 对极角范围的描述可能有差异，因此几何运算一律以每个 ATHDF 文件中的 `x1f/x2f/x3f` 为权威来源。

### 2.3 27 个原始字段

实际 `VariableNames` 为：

```text
rho, press, vel1, vel2, vel3,
ism_frac, cgm_frac, sw_frac, snia_frac, snii_frac,
agnwc_frac, agnwh_frac, agnj_frac,
metallicity, star, newstar, nova, nova_delay,
agn_flux, cool_rate, heat_rate_agn, heat_rate_uvb,
rhodotsf, rhodotev, divB, powstar, coldmass
```

其中没有直接命名为下列语义的字段：

- gravitational potential / `phi`；
- gravitational acceleration / `gravity`；
- cooling time / `tcool`。

### 2.4 初始化 snapshot 风险

`00050` 中多个 user output variable 为全零，而 `00051` 开始出现非零值。它很可能是初始化输出或尚未更新派生量的特殊 snapshot。正式生成训练数据前必须自动检查并记录：

- `rhodotsf` 是否全零；
- `cool_rate` 是否全零；
- 其他 baseline 必要字段是否已经完成第一次更新。

不能只因为目标为零就一概删除，因为真实淬灭阶段也可能有零 SFR；应依据“恰好是初始编号 + 多个派生字段共同全零”的条件识别初始化输出。

### 2.5 原作者确认的 MACER3D 代码单位

原作者给出的基本代码单位为：

| 物理量 | 1 code unit 对应的 cgs 值 |
|---|---:|
| mass | `4.972750e40 g`，约 `2.5008e7 Msun` |
| length | `3.085678e21 cm`，即约 `1 kpc` |
| time | `3.155815e16 s`，即约 `1.000017 Gyr` |
| density | `1.692563e-24 g/cm^3` |
| velocity | `9.777753e4 cm/s` |
| pressure | `1.618166e-14 erg/cm^3` |
| magnetic field | `4.509376e-7 G` |

原作者同时给出以下 code-unit constants：

```text
dyne = 6.490465e-30
erg = 2.103416e-51
Gconst = 1.124766e2
Msun = 4.000000e-8
mH = 3.364939e-65
Lsun = 2.540227e-1
Myr = 1.000000e-3
kB = 2.904099e-67
c = 3.066067e5
electron charge e = 1.405799e-45
```

由 mass/time 单位可得，一个 code mass per code time 约等于：

$$
0.0250075\ M_\odot\,\mathrm{yr}^{-1}.
$$

这与项目当前使用的 `0.025` 换算因子一致。该结论成立的前提仍是
`rhodotsf` 的单位确实为 code density / code time；这一字段语义需作者进一步明确确认。

---

## 3. 字段角色与数据泄露边界

当前最小字段集合已经固定为三类。

### 3.1 神经网络输入

```text
rho, press, vel1, vel2, vel3
```

对应：

- `rho`：气体密度；
- `press`：气体压力；
- `vel1`：球坐标径向速度 (v_r)；
- `vel2`：极角速度 (v_\theta)；
- `vel3`：方位角速度 (v_\phi)。

### 3.2 仅供物理 baseline 使用

```text
cool_rate
```

`cool_rate` 需要随 snapshot 读取和降采样，但默认不拼接进模型输入张量 `X`。它用于在 residual/closure 模式中计算 MACER-LR baseline。

### 3.3 监督目标

```text
rhodotsf
```

`rhodotsf` 表示瞬时 SFR 密度。它只能用于构造标签，不得进入模型输入、条件变量、归一化统计量或任何部署时不可获得的特征。

总瞬时 SFR 应由体积积分得到：

\[
\mathrm{SFR}_{\rm total}
=
\sum_i \dot\rho_{\star,i}V_i.
\]

若采用当前 MACER 单位换算，则：

\[
\mathrm{SFR}[M_\odot\,\mathrm{yr}^{-1}]
=0.025\sum_i \mathrm{rhodotsf}_iV_i.
\]

该积分已在真实 snapshot 上与直接读取数组的结果交叉验证：`00051` 约为 (0.86\ M_\odot\,\mathrm{yr}^{-1})，`00299` 约为 (2.44\ M_\odot\,\mathrm{yr}^{-1})。后续仍需用 `yuan18.hst` 或作者提供的总 SFR 日志做独立确认。

### 3.4 暂不作为输入的字段

`star`、`newstar`、`rhodotsf` 必须从模型输入中排除，以避免把恒星形成结果或其高度相关代理泄露给模型。

各类 `*_frac` 是物质来源的被动标量。它们不是 MACER 恒星形成公式的基本输入，而且可能让模型直接识别模拟反馈类型，形成不具备部署性的 shortcut，因此第一版不使用。

`metallicity` 的处理不能永久定为“不需要”：

- 若直接使用 snapshot 已保存的 `cool_rate`，金属丰度对高分辨率冷却的影响已经部分包含在其中；
- 若要在 coarse 状态上严格重新查询 MACER3D cooling table，则需要 `metallicity`，并可能还需要 `agn_flux`、redshift 以及对 AGN/UVB heating 的一致处理；
- 宇宙学模拟通常可以提供 metallicity，因此它是合理的后续消融变量。

第一版保持五通道最小输入；待 baseline 语义确认后，再分别做 `+metallicity`、`+agn_flux` 等受控消融实验，不能一次加入全部 27 个字段。

### 3.5 压力、内能密度与温度关系：作者确认结果

ATHDF 输出配置为 `variable=prim,uov`，实际 primitive variables 为
`rho, press, vel1, vel2, vel3`，因此文件直接保存的是压力，而不是 energy。
在 `gamma=5/3` 下，与二维 `gas_energy` 对应的派生量定义为内能密度：

$$
e_{\rm int}=\frac{p}{\gamma-1}=1.5p.
$$

比内能与不含引力势的流体总能量密度分别为：

$$
u_{\rm int}=\frac{e_{\rm int}}{\rho},
\qquad
E_{\rm hydro}=e_{\rm int}+\frac12\rho|\boldsymbol v|^2.
$$

它们是三个不同物理量，processed 数据不得都使用含糊的 `gas_energy`
名称。三维流程计划使用明确名称 `internal_energy_density`；若未来保存总能量，
则另用 `resolved_hydro_energy_density`，并说明它不包含引力势能和降采样后丢失的
未解析速度弥散能。

原作者提供的平均分子量和温度函数为：

```cpp
Real GetMeanMolWeight(Real Z_Zsun)
{
  Real Z_frac = Constants::solar_metallicity * Z_Zsun;
  Real mu = 1.0 / (2. * H_frac
      + 3. / 4. * (1. - H_frac - Z_frac)
      + Z_frac / 2.);
  return mu * punit->hydrogen_mass_code;
}

Real GetTemp(Real press, Real rho, Real Z_Zsun)
{
  Real mean_mol_weight = GetMeanMolWeight(Z_Zsun);
  return press * mean_mol_weight
      / (rho * punit->k_boltzmann_code);
}
```

其中 `H_frac=0.7`，当前实现假定完全电离。作者注明未来才会加入更完整的
metallicity/temperature-dependent mean molecular weight。

因此精确关系应写为：

$$
T_C=
\frac{p_C}{\rho_C}
\frac{\mu(Z_C)m_{\rm H,code}}{k_{\rm B,code}},
$$

而不是把二维常数 93 写死。使用作者给出的
`mH_code=3.364939e-65`、`kB_code=2.904099e-67` 和
`mu≈0.61633` 时，系数约为：

$$
\frac{\mu m_{\rm H,code}}{k_{\rm B,code}}
\simeq71.41.
$$

这与此前从真实数据的 cooling floor 和 star-formation ceiling 独立反推的
约 71.4 完全一致，从而确认 `T=93e/rho` 不能直接移植到当前 MACER3D 数据。

温度应在密度、压力和 metallicity 完成 coarse aggregation 后重新计算，
不能直接体积平均 fine temperature。由于 `Z_Zsun` 进入平均分子量函数，
即使 metallicity 不作为神经网络输入，预处理仍需读取它来派生温度。

当前仍需确认两点：

1. `Constants::solar_metallicity` 的精确数值；
2. ATHDF 中 `metallicity` 保存的是绝对金属质量分数 `Z_frac`，还是太阳单位
   `Z_Zsun`。从初始区约 `0.00134` 与输入的 `0.1 Zsun` 对照看，前者更可能，
   但在作者确认前不写死转换。

`yuan18.hst` 有 32 列但没有表头，不能单独用于确认局部温度或能量定义。

---

## 4. 已实现：严格 ATHDF 选择性读取器

实现文件：

```text
raw_data_processing/athdf_reader.py
```

### 4.1 功能边界

读取器只负责：

1. 验证 ATHDF 文件结构；
2. 根据 `DatasetNames/NumVariables/VariableNames` 动态建立字段位置映射；
3. 根据 `LogicalLocations` 重建全局均匀层级网格；
4. 将 Athena++ block 数组从 `(phi, theta, r)` 显式转置为项目统一的 `(r, theta, phi)`；
5. 根据真实 cell faces 计算每个球坐标 cell 的精确体积；
6. 按需读取指定字段，避免一次加载全部 27 个变量。

它当前不负责：

- 三维降采样；
- 计算 cooling time；
- 重建引力势；
- 构造 MACER baseline；
- 生成训练缓存；
- 猜测不明确字段的物理语义。

### 4.2 默认字段常量

读取器中已经把三类字段拆开：

```python
DEFAULT_MODEL_INPUT_FIELDS = (
    "rho", "press", "vel1", "vel2", "vel3"
)
DEFAULT_BASELINE_FIELDS = ("cool_rate",)
DEFAULT_TARGET_FIELD = "rhodotsf"
```

`DEFAULT_REQUIRED_FIELDS` 是生成一份完整训练样本时需要从原文件读取的并集，不代表所有字段都会送入神经网络。

### 4.3 结构验证

当前读取器会拒绝：

- 非 `spherical_polar` 坐标；
- 缺少必要属性或坐标数组的文件；
- `RootGridSize` 不能被 `MeshBlockSize` 整除的文件；
- 重复、缺失或越界的 `LogicalLocations`；
- `DatasetNames/NumVariables/VariableNames` 不一致；
- 数据集形状与属性声明不一致；
- 非有限字段值；
- 非严格递增的 cell faces；
- 非完整 (2\pi) 方位角区域；
- `MaxLevel != 0` 的 AMR 文件。

AMR 文件被明确拒绝，而不是静默按均匀网格误读。未来如出现 AMR 数据，需要单独实现 conservative restriction。

### 4.4 精确球坐标体积

每个 cell 的体积为：

\[
V_{ijk}
=
\frac{r_{i+1}^3-r_i^3}{3}
\left(\cos\theta_j-\cos\theta_{j+1}\right)
\left(\phi_{k+1}-\phi_k\right).
\]

体积使用 `float64` 保存，并验证逐 cell 体积总和等于整个球坐标区域的解析体积。坐标中心只用于位置描述，不承担体积恢复职责。

### 4.5 命令行检查方式

只检查元数据：

```powershell
python raw_data_processing/athdf_reader.py "E:\path\to\snapshot.athdf"
```

读取指定字段：

```powershell
python raw_data_processing/athdf_reader.py "E:\path\to\snapshot.athdf" `
  --fields rho press vel1 vel2 vel3 cool_rate rhodotsf
```

---

## 5. 已实现：`HdfraConfig.json` 中的矮星系配置

虽然文件名沿用了二维数据时期的 `HdfraConfig.json`，当前已经新增 `dwarf_galaxy` 配置段，记录：

- `format = athdf`；
- 球坐标系和物理轴顺序；
- Athena++ block 数组顺序；
- 七个必要字段的 correspondence；
- `model_inputs/baseline_only/target` 三类角色；
- cell faces 与体积来源；
- target 的体积积分语义和单位换算；
- baseline 所需引力势的组成和来源状态。

需要注意：当前二维 `galaxy_data.py` 仍只处理原来的 HDFRA 数据。配置段的加入不代表现有二维预处理器已经能够处理 ATHDF。未来三维预处理器必须显式读取这一配置。

---

## 6. 引力势与 MACER SFR baseline

### 6.1 论文中的恒星形成关系

MACER3D 使用：

\[
\dot\rho_{\rm SF}
=
\eta_{\rm SF}\frac{\rho}{\tau_{\rm SF}},
\qquad
\tau_{\rm SF}=\max(\tau_{\rm cool},\tau_{\rm dyn}),
\]

\[
\tau_{\rm dyn}=\min(\tau_{\rm ff},\tau_{\rm rot}),
\qquad
\tau_{\rm ff}=\sqrt{\frac{3\pi}{32G\rho}},
\]

\[
\tau_{\rm rot}=\sqrt{\frac{r}{\partial\Phi/\partial r}}.
\]

矮星系背景势由四部分组成：

\[
\Phi
=
\Phi_{\rm BH}
+\Phi_{\rm disk,*}
+\Phi_{\rm bulge}
+\Phi_{\rm DM}.
\]

论文明确说明当前忽略气体自引力。

### 6.2 常规训练 snapshot 不含引力；另有专用 `grav` 文件

常规的 27-field 训练 snapshot 中没有引力势或引力加速度。`vel3` 是气体方位角速度，不能代替背景势给出的圆轨道时间尺度；`star` 即使代表恒星密度，也不包含黑洞和暗物质晕的完整贡献。

原作者另行提供了根目录文件：

```text
Fid_yuan18.out1.00000.athdf
```

该文件包含第 28 个变量 `grav`。实测 metadata 为：

- `Time=2.0`；
- 全局网格仍为 `(256,64,128)`；
- `grav` 重建后形状为 `(256,64,128)`；
- 全部数值为负，范围约为 `-92886` 至 `-54.8` code acceleration；
- 对 phi 完全不变，但随 r 和 theta 变化；
- 行为与指向中心的总径向引力加速度一致。

这个专用文件的 MeshBlock 划分是 `(64,32,64)`，不同于训练 snapshot 的
`(16,16,16)`，但全局网格相同。读取时仍必须经 `LogicalLocations` 重建，
不能按 block 直接与训练文件逐块配对。

因此：

- 直接读取 `rhodotsf` 作为训练真值不需要重新计算引力势，因为模拟已经在产生该字段时使用了自身的恒星形成模块；
- 从 coarse 状态独立计算 MACER-LR baseline 时，必须获得或重建 (partial\Phi/\partial r)。

### 6.3 作者提供的径向引力加速度实现

根目录 `grav.txt` 给出了 `GetTotalAcceleration(r,theta,m_bh)`。它返回四项
径向加速度之和：

```text
acc_tot = acc_bh + acc_bulge + acc_stellar_disk_r + acc_halo
```

其中：

$$
a_{\rm BH}=-\frac{GM_{\rm BH}}{r^2},
$$

$$
a_{\rm bulge}=-\frac{GM_{\rm bulge}}{(h_{\rm bulge}+r)^2},
$$

恒星盘项通过 `R=r*sin(theta)`、`z=r*cos(theta)` 和 `abulge` 构造，之后乘
`ratio_gs`；暗物质晕项使用 `d0DM` 与 `rsDM` 的 NFW 形式。

`grav.txt` 同时给出了 stellar-disk theta acceleration，但
`GetTotalAcceleration` 和专用 `grav` 字段目前表现为总径向加速度。对论文
rotation time 所需的径向势梯度，有：

$$
\frac{\partial\Phi}{\partial r}=-a_r=|\mathrm{grav}|.
$$

2026-09-10：用户转述作者确认专用 `grav` 为静态场，可跨时间复用。
当前离线算子直接读取该场，并逐轴核对目标 snapshot 的中心和边界。
解析重建仍缺少 `ratio_gs`，但不再阻塞该读取路径。
注意：多 snapshot 的局部 SFR 残差显示，恒星形成模块可能另有随时间变化的
中心质量贡献；这与背景场静态是否矛盾，需要作者说明，见第 18 节。

Direct-SFR 降采样和训练不需要 `grav`，因此这些问题不阻塞本轮 Direct 数据
生成；它们只阻塞后续严格 residual/closure baseline。

### 6.4 可用于解析重建的信息

`RawData/DwarfGalaxy/input.txt` 给出了以下初始条件参数：

- 恒星盘：`Mstar, hstar, zstar`；
- 核球：`Mbulge, hbulge`；
- NFW 暗物质晕：`d0DM, rsDM`；
- 中心黑洞初始质量关系：`eps_bh_mass`。

但这些参数本身仍不足以保证与模拟完全一致。严格重建还需要：

1. MACER3D 源代码中实际采用的势函数或加速度函数；
2. 非球对称恒星盘势的数值实现和径向投影定义；
3. 每个 snapshot 对应的黑洞质量；
4. 所有单位和软化/内边界处理。

如果背景恒星、核球和暗物质势固定，可以在固定网格上预计算其加速度；黑洞项则根据时间匹配的 (M_{\rm BH}(t)) 更新。当前 `yuan18.hst` 没有可见表头，因此不能在未确认列语义时直接指定某一列为黑洞质量。

配置中已明确记录：引力势不在 ATHDF 中，必须结合精确模拟实现和初始条件重建。

### 6.5 公式中的 2-pi 差异

当前二维 `raw_data_processing/macer_sfr_operator.py` 的 `paper_physical` 模式使用：

\[
\tau_{\rm rot}=2\pi\sqrt{\frac{r}{|a_r|}}.
\]

但目前读取的矮星系论文公式没有 (2\pi)。这会造成 (2\pi) 倍的时间尺度差异，并改变由 rotation limiter 控制的 cell。

在实现三维 baseline 前必须向数据提供者确认：

- 源代码中是否包含 (2\pi)；
- 论文是否省略了该因子；
- `rhodotsf` 实际由哪一种实现产生。

2026-09-10：独立三维算子的未降采样验证强烈支持带 `2pi` 的旋转时标；
不直接复用二维效率、密度阈值或有限时间步函数。数值证据见第 18 节。

### 6.6 `cool_rate` 与严格 LR baseline 的区别

当前计划读取 `cool_rate` 计算 baseline，但必须区分两个口径：

1. **数据兼容近似**：先将 HR snapshot 中已有的 `cool_rate` 守恒/加权降采样，再由 coarse `rho/press/cool_rate` 计算 cooling time；
2. **严格 LR 物理 baseline**：先降采样状态，再在 coarse 状态上重新调用 cooling table 得到新的 cooling rate 和 cooling time。

由于 cooling 是非线性的，通常有：

\[
\mathcal R[\Lambda(U_{\rm HR})]
\ne
\Lambda(\mathcal R[U_{\rm HR}]).
\]

第一种方法更容易实现，但不是严格意义上的 (mathcal S_\alpha(\mathcal R_\alpha U))。第二种方法更符合闭合建模定义，却需要 cooling table、metallicity、AGN flux、redshift 和 heating 处理等更多信息。

原作者已经确认 `cool_rate` 是单位体积冷却率，因此若以后需要对它做空间
降采样，应采用体积加权平均，以守恒 coarse cell 内的总 cooling power。
2026-09-10：用户授权先采用 `t_cool=press/((gamma-1)*cool_rate)`。
离线算子把输入解释为正的 code cooling power density，拒绝负值，零值对应
无限冷却时间。该假设在当前高分辨率样本上与记录 SFR 相容，但恒星形成区
主要由动力学时标主导，因此尚不能凭 SFR 吻合独立确认冷却功率的完整定义。
Direct 模式仍不要求该字段作为网络输入。

---

## 7. 可配置的三维守恒降采样

2026-09-10：Direct/Residual 空间块降采样已实现；最新接口与运行结果见第 20 节。
本节保留物理设计，早期接口草案以第 20 节为准。

### 7.1 初始目标分辨率

当前从原网格：

\[
(256,64,128)
\]

默认降到：

\[
(32,16,32).
\]

三个方向的整数合并因子分别为：

\[
(8,4,4).
\]

每个 coarse cell 对应 128 个 fine cell，总 voxel 数从 2,097,152 降到
16,384，也就是减少 128 倍。用户可以手动指定 `(16,16,32)`，此时合并因子
为 `(16,4,4)`，每个 coarse cell 对应 256 个 fine cell。

目标大小必须成为命令行参数和缓存 metadata，例如：

```text
--coarse-shape 16 16 32
```

第一版只接受各方向可整除的目标大小；如未来允许不可整除尺寸，必须显式定义非等长索引分区，不能使用普通插值伪装成守恒降采样。

### 7.2 球坐标下的分区

降采样是在原生 `(r,theta,phi)` 网格索引上合并连续 cell：

- coarse 径向边界取 fine `x1f` 的分组边界；
- coarse 极角边界取 fine `x2f` 的分组边界；
- coarse 方位角边界取 fine `x3f` 的分组边界；
- coarse volume 等于成员 fine volume 的和，并同时与 coarse faces 的解析球坐标体积交叉验证。

这种做法不会把球网格当成横平竖直的直角坐标网格。由于原网格径向为对数分布，等索引分组并不意味着相等的物理长度或体积；这是保留原生球坐标分层的结果，而不是 bug。

如果科学目标要求近似恒定物理尺度或恒定质量，则需要另行实现固定物理尺度或质量自适应算子，不能把它与固定索引分组混称为同一种分辨率。

### 7.3 标量场的降采样语义

设 coarse cell (C) 包含 fine cells (i\in C)。

体积为：

\[
V_C=\sum_{i\in C}V_i.
\]

密度为：

\[
\rho_C=\frac{\sum_i\rho_iV_i}{V_C},
\]

从而严格保持气体质量：

\[
\rho_CV_C=\sum_i\rho_iV_i.
\]

压力采用体积加权平均：

\[
p_C=\frac{\sum_i p_iV_i}{V_C}.
\]

因为 `gamma` 是常数，这同时严格保持由压力定义的总内能：

\[
e_{{\rm int},C}=\frac{p_C}{\gamma-1}
=\frac{\sum_i e_{{\rm int},i}V_i}{V_C}.
\]

压力本身不是守恒量，但 `sum(e_int*V)` 是本流程必须验证的 extensive
quantity。降采样后的 resolved kinetic energy 通常小于 fine kinetic energy；
两者之差代表未解析速度弥散，不能伪称被当前 Direct 数据产品守恒。

目标 SFR 密度应采用体积加权平均：

\[
\dot\rho_{\star,C}
=
\frac{\sum_i\dot\rho_{\star,i}V_i}{V_C},
\]

保证降采样前后总 SFR 完全一致。

为计算 coarse mean molecular weight，需要把 metallicity 当作质量分数做
气体质量加权：

\[
Z_C=\frac{\sum_i\rho_iV_iZ_i}{\sum_i\rho_iV_i}.
\]

随后由 coarse `rho/press/metallicity` 重新计算温度。metallicity 可以保存在
processed HDF5 中作为派生量依赖，但 Direct 网络默认不使用它。

原作者已确认 `cool_rate` 是单位体积冷却率；未来如启用，应体积加权以守恒
总 cooling power。本轮 Direct 模式明确不读取、不降采样也不输出
`cool_rate/cooling_time`。

### 7.4 速度不能直接逐球坐标分量平均

`vel1/vel2/vel3` 是每个 fine cell 局部球坐标基底上的分量。不同 ((\theta,\phi)) cell 的基底方向不同，因此直接对 (v_r,v_\theta,v_\phi) 分量做质量加权，并不严格守恒三维笛卡尔动量。

正确做法为：

1. 将每个 fine cell 的球坐标速度转换成笛卡尔速度；
2. 计算并加和笛卡尔动量 (
ho_iV_i\boldsymbol v_i)；
3. 除以 coarse gas mass 得到 coarse 笛卡尔速度；
4. 选择直接保存三通道笛卡尔速度，或者再投影到 coarse cell 中心的球坐标基底。

守恒运算与最终存储表示需要分开处理：守恒运算必须在笛卡尔动量上完成，
但同一个 coarse 速度矢量可以无损地保存为笛卡尔或 coarse-center 球坐标分量。

当前推荐 processed cache 同时保存：

- `momentum_x/momentum_y/momentum_z`：守恒审计用的 extensive quantities；
- `velocity_x/velocity_y/velocity_z`：由 coarse momentum 除以 coarse mass；
- `vel1/vel2/vel3`：将同一 coarse 笛卡尔速度投影到 coarse cell 中心基底所得，供物理分析和输入消融使用。

两套速度不是两次独立平均的结果，而是同一个守恒 coarse 矢量的两种坐标表示。
第一版网络究竟使用哪一套，应作为显式配置和 paired ablation，而不能写死在降采样器中。

方案A：保存 vx, vy, vz
优点：
- 与守恒得到的笛卡尔动量直接对应；
- 全局基底固定，没有局部球坐标基底随位置转动的问题；
- 更接近未来 GIZMO 等通常采用的笛卡尔速度；
- 容易验证三个方向的总动量守恒；
- 不依赖如何定义 coarse cell 的代表性球坐标基底。
缺点：
- 盘旋转在笛卡尔分量中会随 \(\phi\) 呈正弦/余弦变化；
- 普通 CNN 需要额外学习这种坐标依赖；
- vx/vy/vz 不如 \(v_r/v_\phi\) 直观，径向流入、流出和旋转不容易直接解释；
- 模型会依赖星系在全局坐标系中的朝向，未来部署时可能需要统一角动量轴或加入旋转增强。

方案B：保存 coarse vel1, vel2, vel3
这里不是直接平均原始球坐标分量，而是把守恒得到的 coarse 笛卡尔速度重新投影到 coarse cell 中心：
\[
(v_x,v_y,v_z)_C
\longrightarrow
(v_r,v_\theta,v_\phi)_C.
\]优点：
- \(v_r\) 直接表示流入/流出；
- \(v_\phi\) 直接表示盘旋转；
- 对接近轴对称的旋转盘，空间图像通常比笛卡尔分量简单；
- 与原始 ATHDF 的字段命名和物理分析习惯一致；
- 如果投影使用一致的 coarse 中心，它与笛卡尔速度包含相同的三个自由度，不会额外损失信息。
缺点：
- 每个 cell 的三个分量属于不同的局部基底；
- 普通 Conv3d 会把它们当作普通标量通道，并不知道基底随 \((\theta,\phi)\) 改变；
- 接近极区时球坐标基底更敏感；
- 部署到笛卡尔宇宙学模拟时，需要先确定星系中心和旋转轴，再进行坐标转换；
- 如果误用“直接平均球坐标分量”，就会破坏动量守恒。



### 7.5 每个 snapshot 的守恒审计

预处理必须自动计算并保存：

- fine/coarse 总体积及相对误差；
- fine/coarse 总气体质量及相对误差；
- fine/coarse 三个笛卡尔总动量分量及相对/绝对误差；
- fine/coarse 总瞬时 SFR 及相对误差；
- 若语义确认，fine/coarse 总冷却功率；
- NaN/Inf 数量；
- 正体积、非负密度等物理合法性检查；
- coarse faces 与逐 fine volume 求和的一致性。

守恒误差超过阈值时必须拒绝写入训练缓存，而不是只输出 warning。

### 7.6 计划中的代码接口与处理流程

本轮计划新增：

```text
raw_data_processing/coarsening_spatial_3D.py
```

核心函数接受 `CoarseningInput` 或等价的纯数组契约，并返回现有
`CoarseningResult`，避免数值模块反向依赖 `GalData`。拟定接口为：

```text
coarsen_spatial_3d(data, new_shape=(32,16,32), ...)
```

`galaxy_data.py` 负责：

1. 识别 `dwarf_galaxy`/`athdf` 数据源；
2. 调用 `athdf_reader.py` 读取 `rho/press/vel1/vel2/vel3/metallicity/rhodotsf`；
3. 将字段映射为明确的三维 processed names；
4. 通过统一 `GalData.coarsen()` dispatch 调用 `spatial_3d`；
5. 把结果写到 `BaseConfig.data_dir` 下对应星系目录；
6. 保存来源、单位、几何、字段角色和守恒审计 metadata。

命令行 `--coarse-shape` 需要由当前只接受两个整数改为按数据维数验证：

- 二维数据接受两个整数并保持既有行为；
- 三维数据接受三个整数；
- 三维默认值为 `(32,16,32)`；
- `--coarse-shape 16 16 32` 允许手动覆盖；
- 同一条命令若混合处理二维和三维数据，不允许用一个有歧义的 shape 静默套用。

三维 Direct 每个 snapshot 的计算顺序为：

```text
inspect/read ATHDF
  -> validate fields, axes and source geometry
  -> build exact fine cell volume from faces
  -> construct fine Cartesian mass/momentum
  -> non-overlapping index-block reduction
  -> derive coarse rho, pressure and internal-energy density
  -> mass-average metallicity
  -> recompute coarse temperature
  -> derive Cartesian velocity and optional spherical projection
  -> volume-average rhodotsf and integrate total SFR
  -> run mandatory conservation audit
  -> atomically write one processed HDF5
```

为避免单个原始 snapshot 约 216 MiB 带来的额外内存和磁盘压力，应逐
snapshot 流式处理，处理完立即释放 fine arrays。是否额外保存完整 fine HDF5
副本仍需决定；对于已有 ATHDF 原文件，默认只保存 coarse processed HDF5
更节省空间。

---

## 8. 降采样尺寸对计算量的影响

原始网格有 2,097,152 个 voxel，默认 `(32,16,32)` 有 16,384 个 voxel，
手动选择 `(16,16,32)` 时有 8,192 个 voxel。对于五个 `float32` 输入通道：

- 原始单样本输入约为 40 MiB；
- `(32,16,32)` 单样本输入约为 320 KiB；
- `(16,16,32)` 单样本输入约为 160 KiB；
- 两种 coarse shape 仅输入张量的内存分别减少约 128 倍和 256 倍。

对相同通道数、kernel 和层数，3D convolution 的主要 FLOPs 与 activation
memory 近似正比于 voxel 数，因此粗化会显著提升吞吐量并允许更大的 batch
size。但实际训练加速不会严格等于 128 或 256 倍，因为还包括数据读取、
网络头部、GPU kernel 启动、优化器状态和 I/O 等固定开销。

3D convolution 的参数量主要由：

\[
C_{\rm in}C_{\rm out}k_rk_\theta k_\phi
\]

决定，通常与空间尺寸无关；空间尺寸主要影响计算量和中间 activation 内存。

选择更小目标网格的代价包括：

- 更强的信息损失；
- SFR 团块被稀释；
- 阈值型恒星形成关系更难由 coarse 均值恢复；
- 更大的真实 closure；
- 小尺度 AGN/SN 结构被抹平。

后续至少应比较多个可整除尺寸，例如 `(16,16,32)`、`(32,16,32)` 和 `(32,32,64)`，并使用同一时间划分做 paired comparison。

---

## 9. 待实现：训练样本与缓存格式

每个 processed snapshot 至少需要保存：

### 9.1 输入与几何

- `x`：严格只含模型输入通道；
- `channel_names`；
- coarse `r/theta/phi` centers 与 faces；
- `cell_volume`；
- `target_shape` 和各方向合并因子；
- 原始 snapshot 路径、编号、时间和模拟名称；
- `sample_uid` 与 `parent_uid`。

### 9.2 监督量与 baseline

- `sfr_density_true`：守恒粗化后的局部 SFR 密度；
- `target_sfr_physical`：积分后的瞬时总 SFR；
- `macer_sfr_baseline_density`：若 baseline 已可靠实现；
- `macer_sfr_baseline`：baseline 总 SFR；
- `closure_true_density = sfr_density_true - macer_sfr_baseline_density`；
- `closure_true_physical = target_sfr_physical - macer_sfr_baseline`。

### 9.3 学习契约

metadata 必须明确：

```text
baseline_is_model_input = false
baseline_is_output_composition = true   # signed_closure 模式
target_semantics = instantaneous_sfr
target_reduction = volume_integral
```

还必须记录：

- direct-SFR 或 signed-closure 训练目标；
- target/closure transform 与 scale；
- 输入归一化统计量只由 train set 计算；
- baseline 版本、引力势版本、cooling-time 版本；
- 守恒审计结果；
- 是否排除了初始化 snapshot。

不能把 `cool_rate`、baseline、`rhodotsf` 或全数据集统计量意外拼接进 `x`。

---

## 10. 待实现：3DSFRConvNet

### 10.1 第一版结构

可以从二维 `SFRConvNet` 的残差结构出发，将以下模块替换为三维版本：

- `Conv2d → Conv3d`；
- `BatchNorm2d → BatchNorm3d`，或在小 batch 下优先考虑 `GroupNorm`；
- `Dropout2d → Dropout3d`；
- `AdaptiveAvgPool2d → AdaptiveAvgPool3d`。

建议第一版使用较小通道数并限制下采样层数。`r=16` 已经很短，连续 stride-2 很容易过早压缩为空间单点。

### 10.2 球坐标不是普通均匀直角网格

普通 `Conv3d` 默认相邻 index 具有类似物理间距和平移等价性，但当前网格：

- (r) 方向近似对数；
- 相同 (\Delta\phi) 对应的物理弧长随 (r\sin\theta) 改变；
- (phi) 方向周期；
- (	heta) 方向接近极区时几何退化，但本模拟截去了真正的两极。

因此至少需要：

- (phi) 方向 circular padding；
- (r) 和 (	heta) 方向与模拟边界条件一致的 padding；
- 提供坐标、cell volume 或局部尺度作为独立 positional/geometric encoding；
- 不把坐标通道混同为流体物理输入字段。

后续可比较普通 3D CNN、坐标调制 CNN 和局部尺度感知卷积，但第一版应保持结构简单，先验证数据语义和闭合关系。

### 10.3 标量输出与场输出

标量版本可通过全局池化后输出一个瞬时总 SFR 或总 closure；它适合快速检查，但可能只学习整体气体质量和平均密度。

场输出版本应保持 coarse 空间尺寸，输出每个 cell 的：

\[
\widehat{\dot\rho}_{\star,C}
\quad\text{或}\quad
\widehat{\mathcal C}_{C}.
\]

训练损失应同时考虑：

- cell-wise 误差；
- 体积积分后的总 SFR 误差；
- 最终 (B+\widehat C) 的非负约束；
- 不同半径/密度区域的极端不均衡。

最终部署模型优先采用场输出或局部 patch 模型。

---

## 11. 训练目标：direct 与 residual/closure

### 11.1 Direct 模式

\[
\widehat Y=F_\theta(X).
\]

优点是可以在引力势尚未重建时先完成端到端训练；缺点是物理可解释性和跨分辨率外推能力可能较弱。

### 11.2 Signed-closure 模式

在 coarse 状态上计算物理 baseline：

\[
B_\alpha=\mathcal S_\alpha(X_\alpha),
\]

真实闭合项：

\[
C_\alpha^{\rm true}=Y_\alpha-B_\alpha,
\]

模型学习：

\[
\widehat C_\alpha=F_\theta(X_\alpha),
\]

最终预测：

\[
\widehat Y_\alpha=B_\alpha+\widehat C_\alpha.
\]

`cool_rate` 可以参与 (B_\alpha) 的计算，但不进入 (F_\theta) 的输入。训练和推理代码必须显式执行 `baseline + predicted_closure`，不能让模型隐式吸收 baseline。

在引力和 cooling 语义未确认前，可先实现 direct 模式作为工程 smoke test，但不应把它当作最终 residual 模型结论。

---

## 12. 数据划分与泛化评估

相邻 snapshot 间隔约 2 Myr，流体状态高度相关。对单条模拟随机打散后做 60%/20%/20% 划分，会让训练集和测试集包含几乎相邻的时间状态，产生严重乐观偏差。

### 12.1 当前只有一个 run 时

至少同时报告两种结果：

1. 随机划分：仅作为与旧实验兼容的上限参考；
2. 时间块划分：作为主要结果，例如前 60% train、中间 20% validation、后 20% test，并在边界保留时间 gap。

如果恒星形成演化明显非平稳，还应使用滚动时间验证，避免结论只依赖某一个时间边界。

### 12.2 将来有多个反馈配置时

需要分层评估：

- 每个 run 内时间块划分；
- 留一模拟配置验证，例如 hold out `SNonly_inflow`；
- 留一物理过程或参数范围外推；
- 不同分辨率 holdout。

同一原始 snapshot 生成的不同降采样尺寸或数据增强版本必须通过 `parent_uid` 绑定在同一个 split，避免跨分辨率泄露。

任何 oversampling 只能在 split 完成后应用于 train set，validation/test 不能过采样。

### 12.3 样本量限制

当前只有一条 250-snapshot 模拟，不能据此证明模型对其他矮星系、其他反馈配置或三维盘星系具有泛化能力。即使采用 patch 训练，patch 也不是独立模拟样本；所有来自同一 snapshot 的 patch 必须属于同一个 split。

---

## 13. 输入变换、损失与训练稳定性

建议的初始变换：

- `rho`、`press`：正值 log transform 后使用 train-only mean/std；
- 速度：保留符号，使用标准化或 signed-asinh；
- SFR/closure：支持零值；direct SFR 可用 `log1p` 类变换，signed closure 使用可逆 signed-asinh；
- 所有 transform 参数只用 train set 拟合并写入 metadata。

由于 3D 模型 batch size 可能很小，第一版优先考虑 `GroupNorm`，不要默认 BatchNorm 的 running statistics 在 batch=1–4 时可靠。

评估不能只看全局 (R^2)。至少包括：

- MAE、RMSE、相对误差和 log-space 误差；
- 总 SFR 时间序列；
- 累积新生恒星质量；
- 零 SFR/低 SFR/爆发阶段分段指标；
- baseline、prediction 和 true 的三方比较；
- cell-wise 径向、极角和方位角残差；
- 总 SFR 积分误差与局部场误差；
- 负的最终 SFR 比例；
- 不同时间块、不同 run、不同分辨率上的指标。

---

## 14. 结果分析流程的设计要求

三维结果分析不能只沿用二维图片。至少需要生成：

- 指定 (r/\theta/\phi) 切片上的 true、baseline、prediction、residual；
- 方位角平均后的 (r-\theta) 对照图；
- 径向 SFR profile；
- 体积或质量加权的误差分布；
- 总瞬时 SFR history；
- cumulative formed stellar mass；
- 按 feedback phase、SFR burst/quiescent phase 分组的统计；
- 守恒审计和 inference metadata 摘要。

所有全局 SFR 图必须从局部场和 `cell_volume` 重新积分得到，并与缓存中的标量结果交叉验证，避免分析代码再次混淆瞬时 SFR 和历史累积 SFR。

---

## 15. 测试与验收标准

已实现测试位于：

```text
tests/test_athdf_reader.py
```

当前覆盖：

- 跨 HDF dataset 的变量映射；
- 打乱 MeshBlock 顺序后的正确重建；
- `(phi,theta,r) → (r,theta,phi)` 转置；
- 精确球坐标 cell volume；
- 缺字段拒绝；
- AMR 拒绝；
- 模型输入、baseline-only 和 target 的角色隔离；
- `HdfraConfig.json` 与读取器默认字段的一致性。

2026-09-04 在 `D:/conda_envs/physics_ai/python.exe` 环境运行完整测试套件，结果为 87 tests passed。

三维完整流程在被视为可训练前还必须增加：

1. 真实 snapshot 的 metadata smoke test；
2. 多 snapshot 字段集合、形状和坐标一致性测试；
3. 标量与矢量降采样守恒测试；
4. (phi) 周期边界测试；
5. target 体积积分测试；
6. 初始化 snapshot 识别测试；
7. baseline 与源 MACER3D 实现的逐 cell 对照；
8. 训练缓存无标签泄露测试；
9. split 的时间/parent UID 无重叠测试；
10. 3D model 单 batch forward/backward、checkpoint round-trip 和结果分析一致性测试。

---

## 16. 分阶段实施顺序

### 阶段 A：数据语义确认

2026-09-05 更新后的状态：

- 已取得基本 code units、`GetMeanMolWeight` 和 `GetTemp` 源代码；
- 已确认 `cool_rate` 是单位体积冷却率；其符号、完整单位和是否为 net cooling 留待 residual 阶段确认；
- 已取得背景径向加速度源代码和专用 `grav` ATHDF；
- 仍需确认 `Constants::solar_metallicity` 以及 ATHDF `metallicity` 的存储口径；
- 仍需获取 `yuan18.hst` 的列名；
- 仍需确认 `rhodotsf` 的精确单位、时间层和 finite-step/stochastic 语义；
- 仍需确认 `00050` 是否为应排除的初始化 snapshot；
- residual 阶段还需确认专用 `grav` 是否可跨时间复用、`ratio_gs`、时间匹配的黑洞质量及 rotation time 的 2-pi 定义。

以下为最初建立文档时的历史待确认清单，以上述更新状态为准：

- 向数据提供者确认 `cool_rate` 的单位、符号和是否为 net cooling；
- 获取 MACER3D 温度换算的精确源代码常数；实数据已否定直接使用二维的系数 93；
- 获取 `yuan18.hst` 的列名；
- 确认时间匹配的黑洞质量字段；
- 获取背景势/加速度源代码；
- 确认 (	au_{\rm rot}) 是否包含 (2\pi)；
- 确认 `rhodotsf` 的代码单位和有限时间步 limiter。

### 阶段 B：守恒降采样器

- 支持可配置 `target_shape`；
- 使用真实 faces 和 volume；
- 以笛卡尔动量方式处理速度；
- 输出 target、baseline 必要字段和完整 metadata；
- 对每个 snapshot 强制执行守恒审计。

### 阶段 C：Direct 3D 标量模型

- 构建最小 `3DSFRConvNet`；
- 打通 processed HDF5 → CachePT → train → checkpoint → result analysis；
- 仅用于工程验证和基本可学习性检查。

### 阶段 D：MACER-LR baseline

- 重建 coarse 引力加速度；
- 明确 coarse cooling-time 口径；
- 与 MACER3D 源代码逐 cell 对照；
- 保存 baseline 和 limiter diagnostics。

### 阶段 E：Signed-closure 模型

- 学习 (C=Y-B)；
- 显式构造 (B+\widehat C)；
- 同时比较 direct、baseline-only 和 closure-corrected prediction。

### 阶段 F：局部场输出与部署接口

- 由总 SFR 标量回归升级为局部 SFR 密度/closure 场；
- 使用 patch 或 fully convolutional 训练；
- 加入宇宙学模拟可获得的分辨率、cell volume、局部尺度等条件；
- 定义与目标宇宙学代码的数据交换和单位接口。

### 阶段 G：泛化实验

- 时间块 holdout；
- run/反馈配置 holdout；
- 多分辨率 holdout；
- `metallicity/agn_flux` 等受控消融；
- 与方位角平均二维方案做 paired comparison。

---

## 17. 当前阻塞项与决策清单

| 项目 | 当前状态 | 未解决问题 | 对后续的影响 |
|---|---|---|---|
| ATHDF 读取 | 已完成 | 当前仅支持 `MaxLevel=0` | 可开始非 AMR 数据预处理 |
| 字段角色 | 基本完成 | metallicity 仅作温度依赖还是也保存为候选输入 | 不阻塞最小 direct 模型 |
| code units | 作者已提供 | 需写入正式配置和 HDF5 metadata | 不阻塞数值降采样 |
| 热力学派生量 | 公式已确认 | 缺 `solar_metallicity` 精确值与 ATHDF metallicity 口径 | 阻塞最终温度实现 |
| 三维守恒降采样 | Direct/Residual 空间块已实现，真实样本两种 shape 通过 | 仅支持正密度、无 AMR、整除的原生球坐标网格 | 可生成 fine/coarse；尚未接入三维训练 |
| target | 基本确认 | `rhodotsf` 精确单位、时间层、limiter/stochastic 语义未由作者确认 | 可保守降采样原值，但阻塞最终物理解读 |
| 引力势 | 用户确认静态 grav 可复用，已用于离线算子 | 局部残差暗示 SFR 模块可能另用增长后的 BH 质量 | 不阻塞当前总 SFR 复现；局部精确率待确认 |
| rotation time | 18 个 snapshot 的验证支持带 2pi | 尚未取得三维恒星形成函数源码 | 已有数据支持的默认实现 |
| cooling time | 已按用户授权实现 p/((gamma-1)cool_rate) | SF 区几乎无冷却主导，不能充分识别该定义；LR 冷却表仍未接入 | 可离线求值，不等于严格 LR 重算 |
| 3D 模型 | 未实现 | 标量或场输出的阶段安排 | 不应直接照搬全部 2D 层数 |
| 泛化数据 | 仅发现 `fullFB_inflow` | 其他 run 尚未挂载 | 无法验证跨物理配置泛化 |

### 17.1 是否已经可以开始实现 Direct 降采样器

已经具备的条件：

- ATHDF 结构、字段动态映射和全局网格重建已经验证；
- fine cell volume 可以由真实 faces 精确计算；
- `rho/press/vel1/vel2/vel3/rhodotsf/metallicity` 均可读取；
- `gamma=5/3` 和 `internal_energy_density=press/(gamma-1)` 已明确；
- 温度函数和全部基本 code units 已取得；
- `cool_rate` 与 `grav` 均可在 Direct 阶段排除；
- 默认 `(32,16,32)` 和手动 `(16,16,32)` 均能整除原网格；
- 质量、内能、笛卡尔动量、金属质量、总体积和总 SFR 的守恒规则已经明确；
- 现有 `CoarseningInput/CoarseningResult` 与 `GalData.coarsen()` 可以扩展为统一三维 dispatch。

在“遇到语义不确定立即停止”的约束下，正式写程序前仍需确认：

1. `Constants::solar_metallicity` 的精确数值；
2. ATHDF `metallicity` 是绝对质量分数还是 `Z/Zsun`；
3. `rhodotsf` 是否确为 code density / code time、是否为当前时刻瞬时率，以及是否已经包含 finite-step/stochastic limiter；
4. `00050` 是否应作为初始化输出排除；
5. 用户是否同意 processed HDF5 同时保存 Cartesian 和 coarse-center spherical 两套速度表示，但默认模型输入仍由配置选择；
6. 用户是否同意默认不复制完整 fine HDF5，只保存 coarse processed HDF5，以避免不必要的磁盘占用。

第 1–4 项属于应继续向模拟作者确认的物理/数据语义；第 5–6 项属于项目接口决策。
以上为早期决策记录。2026-09-10 用户已授权保存三套速度相关字段、通过拆分
builder 保存 fine/coarse，并采用质量加权有效径向引力。温度和数密度沿用已验证
SFR3D 公式；metallicity 绝对质量分数仍标注为数据支持的口径。实现状态见第 20 节。

---

## 18. 2026-09-10 三维原始分辨率 SFR 验证结果

本节为最新离线算子状态；前文的 Direct/coarse 开发计划并不因此视为已完成。

实现：`raw_data_processing/macer_sfr_operator_3d.py`。采用连续时间公式：

```text
mu = 1 / (2*X + 0.75*(1-X-Z) + 0.5*Z), X=0.7
T = press/rho * mu*mH_code/kB_code
n = rho / (mu*mH_code*length_unit_cm^3)  [cm^-3]
t_cool = press / ((gamma-1)*cool_rate)
t_ff = sqrt(3*pi/(32*G_code*rho))
t_rot = rotation_factor * sqrt(r/abs(grav))
t_SF = max(t_cool, min(t_ff, t_rot))
rhodot_model = etaSF*rho/t_SF, if T < temp_sf and n > nrho_sf
```

其中 `gamma=1.666666666667, etaSF=0.03, temp_sf=40000, nrho_sf=1`
读取挂载数据中的 `input.txt`。`rotation_factor` 可指定 1 或 2pi，默认后者。
`G_code=112.4766, mH_code=3.364939e-65, kB_code=2.904099e-67,
length_unit_cm=3.085678e21` 来自作者。金属丰度采用数据支持的绝对质量分数
解释。密度阈值使用总自由粒子数密度；不采用二维 `1.02*rho`，也不采用氢核数密度。

静态 `grav` 来源于根目录参考 ATHDF，已对所有扫描 snapshot 核对网格中心与
faces。SFR 参考值为各文件 `rhodotsf*cell_volume` 的积分，换算采用项目统一的
0.025 Msun/yr 每 code rate，未从标签估计或拟合 eta。

结果位于 `data/sfr3d_validation_20260910/`：

- `scan.json`：250 个 snapshot 的原始总 SFR、时刻和非零格点数。
- `validation.json`：18 个选择样本、选样理由、源哈希、参数、五个变体和局部诊断。
- `report.md`：逐 snapshot 的总 SFR 与误差。
- `rotation_diagnostics.json`：00051、00200、00299 的局部旋转残差与无表头 hst 数值对照。

一个全零 snapshot 单列；其余 17 个覆盖 0.862757–2.701754 Msun/yr。
带 2pi 时，最大积分相对绝对误差为 3.23062e-6（0.000323062%）；最大场
积分 L1 误差 `sum(abs(pred-true)*V)/sum(true*V)` 为 3.87606e-6
（0.000387606%），每个选样 snapshot 的正 SFR 格点集合均完全一致。
不带 2pi 时，正参考样本的总 SFR 高估 8.8673%–76.6515%。
改用氢核数密度导致低估 7.7406%–47.5189%；改用 rho/mH 导致低估
4.0993%–25.7306%。仅把温度替换为固定 71.41*p/rho（数密度仍保留 mu(Z)），
最大积分相对误差约 0.00151244%。

局部限制：不能用全局积分精度宣称每格都已精确复现。00299 的活跃格点中，
局部相对误差最大约 5.8573%；这些偏差主要发生在旋转时标主导区。
该 snapshot 的旋转主导区域仅贡献总 SFR 的 0.03543%；局部误差超过 1%
的格点合计仅贡献总 SFR 的 0.01037%。

对远离时标切换边界的旋转主导格点，用记录率反推有效加速度，仅作诊断：

```text
g_eff = r * (2*pi*rhodotsf/(etaSF*rho))^2
delta_M = (g_eff - abs(grav))*r^2/G_code
```

反推增量在同一 snapshot 内接近常数，00051、00200、00299 的中位数约为
24776、80700、132369 Msun。最近时刻 hst 的第 27 列分别约为
24790.52、80706.0、132374.6，第 20 列减第 27 列约为 875000 Msun，恰好
等于 input.txt 的 `eps_bh_mass*Mstar`。这强烈提示 SFR 模块可能使用了
增长后的中心黑洞质量，即使输出的背景 grav 是静态的。由于 hst 无表头，
尚不将第 20/27 列解释设为正式接口，也不把监督值反推的质量用于 baseline。
需要作者明确：SFR 的旋转时间是否另行使用当前 M_BH；hst 第 20、27 列的确切含义。

hst 第 30 列在上述三个匹配时刻也与原 snapshot 的 SFR 积分吻合到输出精度，
但列语义仍未正式确认。因此正式误差表以 snapshot 的 rhodotsf 积分为参考。

当前诊断中恒星形成几乎均受动力学时标限制，良好的 SFR 复现不足以独立验证
冷却公式、低温 cooling table 或 coarse 冷却重算。原始 HR 验证不构成 LR
闭合或跨星系预测能力证明。尚未引入三维有限时间步修正。

复现命令（项目根目录）：

```powershell
python raw_data_processing/validate_macer_sfr_3d.py --raw-dir "E:\科研项目与原始数据\ai+MACERsimulation\MACNet-v1\RawData\DwarfGalaxy\fullFB_inflow" --input "E:\科研项目与原始数据\ai+MACERsimulation\MACNet-v1\RawData\DwarfGalaxy\input.txt" --output data/sfr3d_validation_20260910
python raw_data_processing/diagnose_sfr3d_rotation.py --raw-dir "E:\科研项目与原始数据\ai+MACERsimulation\MACNet-v1\RawData\DwarfGalaxy\fullFB_inflow" --input "E:\科研项目与原始数据\ai+MACERsimulation\MACNet-v1\RawData\DwarfGalaxy\input.txt" --output data/sfr3d_validation_20260910/rotation_diagnostics.json
```

## 19. 变更日志

### 2026-09-10：实现 Direct/Residual 三维空间降采样及拆分构建入口

- 新增 `coarsening_spatial_3D.py`：精确 faces 几何、非重叠索引块聚合、笛卡尔动量守恒、温度重算和自动审计。
- 新增 `galaxy_data_3d.py`：三维字段与独立监督组、稳定 metadata/UID、原子 HDF5 保存及读回校验。
- 新增 `build_snapshots_3d.py`；扩展 `build_fine_from_raw.py`、`build_coarse_from_fine.py`，保留二维入口行为。
- `GalData.read_athdf()` 和 `GalData.load_h5()` 返回三维对象，其 `.coarsen()` 调用三维算子。
- 增加 RawDataConfig 的 dwarf run；旧 parse_log_file 和二维 galaxy_data 主程序跳过 ATHDF，避免误解析日志。
- 模式、源哈希、physics 参数进入产品身份；不同签名的三维产品不允许同名覆盖。
- 验证 00051、00254、00299，两种 coarse shape × Direct/Residual 共 12 个产品，最大守恒 scaled relative error 为 5.40774e-16。
- 新增九个三维空间/文件/入口测试，覆盖标签隔离、冗余字段损坏、几何错配、守恒及禁止 coarse→coarse。
- 相关回归测试共 45 项通过：包含二维 builder、空间算子、metadata、ATHDF 和二维/三维 SFR 算子。
- 未接入三维模型、训练缓存或二维 Results notebook。三维文件若送入旧 GalDataSet，会明确拒绝。

### 2026-09-10：实现并验证独立三维 SFR 算子

- 新增 `raw_data_processing/macer_sfr_operator_3d.py`，输入不含 `rhodotsf`。
- 新增 `validate_macer_sfr_3d.py`：扫描 250 个 ATHDF，按正 SFR 分位数、时间覆盖及最快上升/下降前后选择 18 个样本，比较 5 个公式变体。
- 新增 `diagnose_sfr3d_rotation.py`：诊断局部旋转残差；监督值反推的质量仅用于分析，不进入正式算子。
- 更新 dwarf baseline 配置，记录静态 grav、冷却公式和独立算子接口；尚未接入模型训练或三维降采样。
- 新增 6 个三维单元测试；连同二维算子和 ATHDF 读取器共 23 个相关测试通过。
- 结果与复现入口见第 18 节。新增算子不覆盖二维实现。

### 2026-09-05：记录原作者回复并完善 Direct 三维降采样设计

- 记录 MACER3D 的 mass/length/time/density/velocity/pressure 等 code units 和 code-unit constants。
- 记录作者提供的 `GetMeanMolWeight` 与 `GetTemp`，确认温度应由 coarse `rho/press/metallicity` 重算，数值系数约为 71.4，而不是二维的 93。
- 确认与二维 `gas_energy` 对应的三维派生量为 `internal_energy_density=press/(gamma-1)`。
- 确认 `cool_rate` 是单位体积冷却率；Direct 阶段不读取、不降采样该字段。
- 确认训练数据实际 MeshBlock 为 `(16,16,16)`。
- 检查 `grav.txt` 和 `Fid_yuan18.out1.00000.athdf`：后者包含额外 `grav` 字段，可作为径向引力加速度来源；其跨时间复用规则仍需确认。
- 将三维默认目标 shape 修正为 `(32,16,32)`，保留手动指定 `(16,16,32)` 的接口。
- 增补 `coarsening_spatial_3D.py`、`GalData.coarsen()` dispatch、逐 snapshot 流式输出和 Direct 守恒审计的拟定接口；本轮未编写程序。

### 2026-09-04：核验 MACER3D 热力学关系与速度存储策略

- 由 `prim,uov` 输出类型和 `gamma=5/3` 确认 ATHDF 直接保存压力；与二维 `gas_energy` 对应的派生量应为 `internal_energy_density=p/(gamma-1)`。
- 对 11 个真实 snapshot 的 cooling floor 和 star-formation temperature ceiling 进行双边界检查，否定直接复用 `T=93e/rho`。
- 当前实数据支持 `T≈71.4*press/rho≈47.6*internal_energy_density/rho`；精确常数仍等待 MACER3D 源代码确认。
- 明确速度降采样始终先守恒加和笛卡尔动量；processed cache 计划同时保存笛卡尔速度和由同一 coarse 矢量投影得到的球坐标速度。

### 2026-09-04：建立三维训练设计文档

- 新建本文档，整合矮星系数据结构、字段语义、ATHDF 读取、守恒降采样、baseline、3D CNN、训练划分和结果分析方案。
- 记录当前只实际挂载 `fullFB_inflow` 的事实。
- 明确 `cool_rate` 是 baseline-only 字段，不是神经网络输入。
- 明确 `rhodotsf` 是监督目标，不能进入输入。
- 记录引力势不在 ATHDF 中，必须由模拟实现和初始条件重建。
- 记录论文与现有二维 operator 在 (	au_{\rm rot}) 上的 (2\pi) 差异。
- 将球坐标速度的笛卡尔动量守恒列为三维降采样的强制要求。

### 2026-09-04：实现 ATHDF 选择性读取器

- 新增 `raw_data_processing/athdf_reader.py`。
- 支持动态字段映射、MeshBlock 全局重建、轴顺序转换和精确球坐标体积。
- 默认模型输入设为 `rho/press/vel1/vel2/vel3`。
- 将 `cool_rate` 与模型输入分离，将 `rhodotsf` 与全部输入分离。
- 对非零 AMR level 采取显式拒绝策略。

### 2026-09-04：增加矮星系配置与测试

- 在 `.config/HdfraConfig.json` 新增 `dwarf_galaxy`。
- 加入字段角色、几何、target 和 baseline 引力势依赖说明。
- 在 `tests/test_athdf_reader.py` 增加配置—读取器一致性测试。
- 完整项目测试结果：87 项通过。

---

## 20. 三维空间降采样的实际接口与验证

### 20.1 构建流程与模式

```text
ATHDF + input.txt [+ 静态参考 grav]
  -> build_fine_from_raw.py
  -> data/<galaxy>/<fine_dirname>/*.h5
  -> build_coarse_from_fine.py --coarsener spatial_3d
  -> data/<galaxy>/coarse_spatial3d_<shape>_<mode>/*.h5
```

当前 run 生成名为 `dwarf_galaxy_fullFB_inflow`。无须先执行 parse_log_file；时间
读取 ATHDF metadata，输出索引保留原文件编号。原始零 SFR 帧仍保存，不自动删除；
Residual fine 中 SFR 与 cooling_rate 同时全零时记录 initialization_candidate，
该标记仅供诊断，不能单独据此判断物理淬灭或训练纳入规则。

`--prediction-mode direct` 为三维默认。Direct fine 仅读取 rho、press、vel1/2/3、
metallicity、rhodotsf 与坐标。Residual fine 另读 cool_rate 和静态 grav；
这份 fine 可复用于两类 coarse，Direct fine 则不能构建 Residual coarse。
旧二维处理的物理逻辑不随该三维模式开关改变。

```powershell
python raw_data_processing/build_fine_from_raw.py --galaxy dwarf_galaxy_fullFB_inflow --prediction-mode residual --fine-dirname fine_residual
python raw_data_processing/build_coarse_from_fine.py --galaxy dwarf_galaxy_fullFB_inflow --coarsener spatial_3d --prediction-mode residual --fine-dirname fine_residual
python raw_data_processing/build_coarse_from_fine.py --galaxy dwarf_galaxy_fullFB_inflow --coarsener spatial_3d --prediction-mode direct --fine-dirname fine_residual --coarse-shape 16 16 32
```

fine/coarse 两个命令均支持重复的 `--snapshot-index` 与 `--max-snapshots`。
默认三维 shape `(32,16,32)`；可指定 `(16,16,32)`，必须逐轴整除。
三维只有空间块算子，拒绝 R_m 参数。不允许 coarse 覆盖 fine 目录、coarse 再次
降采样或不同产品签名覆盖同名文件。相同签名允许原子替换重新生成。

### 20.2 字段规则

| 字段 | coarse 定义 | 文件角色 |
|---|---|---|
| cell_volume | fine 体积求和；同时验证 coarse faces 解析体积 | geometry |
| density、gas_mass | 总质量/总体积；单元总质量 | snapshot |
| pressure、gas_energy | 压强体积加权；内能密度 p/(gamma−1) | snapshot |
| metallicity | 金属质量/气体质量 | snapshot，辅助字段 |
| temperature | 由 coarse p、rho、Z 与作者 mu(Z) 公式重算 | snapshot，辅助字段 |
| momentum_x/y/z | 固定笛卡尔基底的单元积分动量，直接求和 | snapshot，辅助字段 |
| velocity_x/y/z | coarse 动量/质量 | snapshot，候选速度表示 |
| vel1/2/3 | 同一笛卡尔速度在 coarse 中心球坐标基底的投影 | snapshot，默认速度输入 |
| sfr_density_true | fine rhodotsf 体积加权 | sfr_supervision，禁止作为输入 |
| cooling_rate_density | fine cool_rate 体积加权 | Residual baseline-only |
| cooling_time | coarse gas_energy/cooling_rate_density；零冷却率为 +inf | Residual baseline-only |
| radial_acceleration | fine 径向 grav 的质量加权标量 | Residual baseline-only |

每个速度字段的 HDF5 dataset attrs 都包含中文说明。momentum 不是动量密度；
两套速度表示对应同一个矢量，不能把九个通道当作九个独立物理自由度。
密度、压强、vel1/2/3 标注为默认输入，其余物理字段按角色区分。
所有物理量在文件中保持作者代码单位；温度为 K；积分 SFR 转 Msun/yr 乘 0.025。

体积仅从真实边界或 fine 权威体积求和获得。coarse r 中心为 sqrt(r_in*r_out)，
theta/phi 中心为边界中点；中心不承担体积恢复职责。

自动审计总体积、质量、内能、金属质量、三个笛卡尔动量分量、SFR；Residual
另审计冷却功率和质量加权径向标量积分。动量近抵消时误差以 fine 各单元绝对量
之和缩放，避免净动量接近零导致相对误差失真。未解析动能单独记录，不转入热能。

### 20.3 Residual 监督与局限

同一个已验证的 SFR3D 算子作用于 coarse density、pressure、metallicity、
cooling_rate 和有效径向引力，在监督组保存局部 baseline、局部 signed closure
以及 canonical `Y_alpha/B_alpha/C_alpha_true`。Y 来自原始 rhodotsf 的守恒积分，
B 来自 coarse 状态重算，C=Y−B；baseline 函数接口不接受真实 SFR。

有效径向引力保持静态 fine grav 的质量加权标量积分，但不等于完整引力矢量的
守恒聚合，也不一定等于 coarse 中心加速度。冷却率来自 HR 数据的体积平均，
不代表在 coarse 状态重新查询 cooling table。以上近似写入 metadata。
后续若修改黑洞质量口径、冷却表或温度关系，应使用新目录重建产品。

三维数据采用 `macnet_3d_format=1`，与二维 format-version 6 区分。当前
`GalDataSet`/CNN/Results notebook 仍是二维流程，不能直接训练三维产品。
Python 使用 `sample = GalData().load_h5(path)` 或 `GalData3D.load_h5(path)`，
必须接收返回对象；三维字段不走二维的 T=93e/rho、坐标转置和 KS baseline。

### 20.4 真实数据验证

选取低 SFR 的 00051、高 SFR 的 00254 和末期 00299。生成三个 Residual fine、
一个独立 Direct fine，以及两个 shape × 两个模式 × 三个 snapshot 的 12 个 coarse。
审计均通过，所有加和量的最大 scaled relative error 约 5.41e-16。

| snapshot | 守恒 True (Msun/yr) | B:32×16×32 | B:16×16×32 |
|---|---:|---:|---:|
| 00051 | 0.862757335 | 0.797669279 | 0.647725029 |
| 00254 | 2.701753484 | 2.193384856 | 2.016161159 |
| 00299 | 2.437168153 | 2.011702403 | 1.863385733 |

Direct/Residual 的相同 shape 上物理聚合和 True 一致。baseline 与 True 不相等
是 coarse 状态进入非线性 SFR 算子的结果，不是质量/SFR 守恒失败。
验证文件位于 `data/dwarf_galaxy_fullFB_inflow/`：`fine_smoke_3d`（约 628 MiB）、
`fine_direct_smoke_3d`（约 175 MiB）和四个带 shape/mode 的 coarse 目录（合计约 20 MiB）。
本轮没有批量处理全部 250 个原始文件。

### 2026-09-10：大型三维文件的续跑与源身份策略

- `build_fine_from_raw.py` 新增默认安全续跑。已有输出只有在来源 snapshot、模式、
  物理参数、输入文件和引力来源契约一致时才跳过；损坏或不匹配的同名文件会停止。
- 三维 ATHDF 默认使用 `--source-hash stat`，记录文件名、字节数和纳秒修改时间，
  避免 `read_athdf` 后再完整读取约 216 MiB 源文件一次。`--source-hash sha256` 保留为
  显式强校验选项，适合新目录的归档级构建，但必然增加一次完整磁盘读取。
- `GalData3D.from_raw` 不再隐式计算 SHA-256；强哈希只能由批处理入口明确传入。
- 三维 fine→coarse provenance 同样改为 stat 文件身份；coarse 的物理来源仍由
  `source_sample_uid` 约束。
- 已实际检查 `fine_sfr_v6/00050.h5` 至 `00052.h5`：均可只读 metadata 后安全跳过，
  三个文件合计续跑约 1.3 秒，未重新载入 snapshot 场。
- 新增 stat 不调用 SHA 与 SHA 显式 opt-in 测试；相关三维测试共 17 项通过。

## 21. 后续更新本文档时的最低要求

每次重大修改至少记录：

1. 修改日期；
2. 修改的文件和主要接口；
3. 数据/字段/单位语义是否变化；
4. 是否影响旧缓存和 checkpoint 兼容性；
5. 新增或修改的测试；
6. 实际验证结果；
7. 尚未解决的风险；
8. 若改变实验结论，说明旧结论为何不再适用。

代码行为与本文档发生冲突时，不应默认相信任意一方；必须通过真实数据、单元测试和物理定义重新核对，并在修复后同步更新两者。
