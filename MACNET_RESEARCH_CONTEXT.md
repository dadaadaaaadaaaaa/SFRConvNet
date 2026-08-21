# MACNet：用物理 AI 将 MACER 小尺度反馈嵌入宇宙学模拟

> 文档用途：这是供后续 AI、代码协作者和课题组成员读取的项目上下文与研究路线说明。
>
> 当前状态：研究方向设计阶段；已有 MACER2D HR 数据和 CNN 基线，但跨数据分布泛化较差。
>
> 最近更新：2026-08-03。

---

## 1. 项目背景与最终目标

本项目来自 MACER 课题组。课题组已经积累了大量 MACER2D 和部分 MACER3D 高分辨率模拟数据。MACER 能够在星系核附近解析比典型宇宙学模拟更小的尺度，并给出黑洞吸积、AGN 风/喷流、恒星形成及其反馈等结果。

典型宇宙学模拟无法直接解析 MACER 所覆盖的小尺度，因此通常使用解析或经验性的亚网格模型。项目的长期目标是：

> 使用神经网络、神经算子或其他物理 AI，从宇宙学模拟可解析的低分辨率气体状态推断未解析的小尺度物理，并把 MACER 所提供的高分辨率反馈知识高效、稳定、守恒地嵌入宇宙学模拟。

最终希望替代或修正的量可能包括：

- 恒星形成率（SFR）；
- 新形成恒星质量；
- 黑洞吸积率（BHAR）；
- AGN 风/喷流的质量、动量和能量注入；
- 未解析冷致密气体比例、湍流能、冷却率等中间闭合量。

当前主要工作聚焦于：

\[
\text{低分辨率气体物理场}
\longrightarrow
\text{SFR 或新形成恒星质量}.
\]

黑洞吸积率和完整 AGN 反馈是后续扩展目标。

---

## 2. 已有数据

单个 MACER snapshot 中目前已知或可能使用的 HDF 数据包括：

| HDF dataset | Python 变量 | 含义 |
|---|---|---|
| Data-Set-2 | `v1` | 径向速度 |
| Data-Set-3 | `v2` | \(\theta\) 方向速度 |
| Data-Set-4 | `v3` | \(\phi\) 方向速度 |
| Data-Set-5 | `d` | 气体数密度 |
| Data-Set-6 | `T` | 气体温度 |
| Data-Set-10 | `e` | 气体内能密度；当前代码中的兼容字段名为 `gas_energy` |
| Data-Set-11 | `sfr` | 恒星形成率或恒星形成率密度，精确定义和单位待核验 |
| Data-Set-12 | `mnewstar` | 新形成恒星质量 |
| Data-Set-13 | `dstar` | 恒星密度 |
| Data-Set-14 | `Mstar_i` | 初始恒星质量 |
| Data-Set-15 | `vol` | 网格单元体积 |
| Data-Set-16 | `mgas` | 每个网格单元中的气体质量 |
| Data-Set-17 | `coolt` | 气体冷却时间 |
| Data-Set-19 | `abun_wind` | AGN 风 tracer/丰度 |
| Data-Set-20 | `abun_jet` | AGN 喷流 tracer/丰度 |
| Data-Set-21 | `abun_cold` | 冷模宇宙学流入 tracer/丰度 |
| Data-Set-22 | `abun_hot` | 热模宇宙学流入 tracer/丰度 |
| Data-Set-23 | `abun_init` | 初始气体 tracer/丰度 |

这些变量并不都适合作为最终部署模型的输入。任何在目标宇宙学代码运行时不可获得的变量，只能用于辅助监督、诊断或教师模型，不能成为生产模型的必需输入。

---

## 3. 已完成的工作与当前问题

### 3.1 已完成的基线

当前工作流程为：

1. 读取 MACER2D 的高分辨率气体场；
2. 人为降采样得到低分辨率输入；
3. 使用 CNN 学习

   \[
   \rho_{\rm LR}(\boldsymbol x),
   T_{\rm LR}(\boldsymbol x),
   \boldsymbol v_{\rm LR}(\boldsymbol x)
   \longrightarrow
   \mathrm{SFR}
   \]

   或

   \[
   \rho_{\rm LR},T_{\rm LR},\boldsymbol v_{\rm LR}
   \longrightarrow
   \mathrm{BHAR}.
   \]

### 3.2 观察到的问题

CNN 在训练分布上的拟合可能可接受，但泛化能力较差。可能原因不能简单归结为网络不够复杂，而应优先检查：

- LR 到目标是否本身是一对多的非确定映射；
- 简单 resize 是否破坏质量、动量和能量守恒；
- 降采样后的输入是否等价于真实 LR 模拟输出；
- SFR 阈值是否导致不连续和严重类别不平衡；
- snapshot 是否被随机打散，造成训练集和测试集时间泄漏；
- 不同模拟、初始条件、黑洞质量和反馈参数之间是否发生分布偏移；
- MACER2D 轴对称结构是否让模型学习了不能迁移到 3D 的模式；
- 训练标签是否只是现有亚网格公式的输出，而不是所需的粗粒化 HR 物理量；
- tracer、恒星场或未来时刻信息是否构成输入泄漏；
- 单 snapshot 是否缺少决定 SFR/BHAR 所需的历史信息。

---

## 4. 原始研究设想

原始设想是：MACER 与宇宙学模拟在气体动力学上大体求解相同的连续性、动量和能量方程，主要区别之一是分辨率。因此可以训练神经算子学习

\[
\mathcal F_\theta:
\boldsymbol U_{\rm LR}
\longrightarrow
\boldsymbol U_{\rm HR},
\]

其中

\[
\boldsymbol U=(\rho,T,\boldsymbol v,e,\ldots).
\]

随后再学习

\[
\boldsymbol U_{\rm HR}
\longrightarrow
\mathrm{SFR/BHAR}.
\]

完整路径为

\[
\boxed{
\boldsymbol U_{\rm LR}
\longrightarrow
\boldsymbol U_{\rm HR}
\longrightarrow
\mathrm{SFR/BHAR}
}.
\]

对这一设想的主要担心是：神经算子可能只针对训练时的固定分辨率对逼近目标，不一定能在新的分辨率上保持物理正确性。因此考虑加入 PDE residual、守恒律和其他物理约束。

这个担心是成立的，但还需要进一步修正问题的数学定义。

---

## 5. 核心物理修正：这首先是闭合问题，不是唯一超分辨率问题

### 5.1 filtered HR 不等于真实 LR 模拟

设 HR 解为

\[
\boldsymbol U_{\rm HR}
=(\rho,\rho\boldsymbol v,E,\ldots),
\]

滤波或粗粒化算子为 \(\mathcal C_\Delta\)，则人工降采样产生

\[
\overline{\boldsymbol U}_\Delta
=\mathcal C_\Delta(\boldsymbol U_{\rm HR}).
\]

但真实低分辨率求解器产生的是

\[
\boldsymbol U_{\rm LR}^{\rm sim}.
\]

一般而言：

\[
\boxed{
\boldsymbol U_{\rm LR}^{\rm sim}
\neq
\mathcal C_\Delta(\boldsymbol U_{\rm HR}^{\rm sim})
}.
\]

造成差异的因素包括：

- 非线性方程在过滤后产生未闭合项；
- HR 与 LR 的数值扩散和截断误差不同；
- 网格、SPH kernel、moving mesh、AMR 和 Riemann solver 不同；
- 引力软化、时间步和反馈注入尺度不同；
- 冷却、恒星形成、吸积和反馈源项具有强非线性；
- MACER 球坐标/轴对称几何与三维宇宙学环境不同。

因此，仅对 HR snapshot 做插值或平均，只能构造 filtered-HR 数据，不能完整代表真实宇宙学 LR 数据。

### 5.2 可压缩流体过滤后出现亚网格项

定义 Favre 滤波速度

\[
\widetilde{\boldsymbol v}
=
\frac{\overline{\rho\boldsymbol v}}{\bar\rho}.
\]

连续性方程近似保持原形式：

\[
\frac{\partial\bar\rho}{\partial t}
+\nabla\cdot(\bar\rho\widetilde{\boldsymbol v})
=\overline{S_\rho}.
\]

动量方程会出现亚网格应力

\[
\tau_{ij}^{\rm SGS}
=
\overline{\rho v_i v_j}
-\bar\rho\widetilde v_i\widetilde v_j.
\]

粗粒化动量方程中需要额外处理

\[
-\nabla_j\tau_{ij}^{\rm SGS}.
\]

能量方程还会产生未解析湍流能、压力功和热通量。冷却项通常满足

\[
\overline{\rho^2\Lambda(T,Z)}
\neq
\bar\rho^2\Lambda(\bar T,\bar Z).
\]

所以分辨率改变时，不只是同一连续解换了一组采样点，而是有效闭合本身也发生了变化。

### 5.3 LR 到 HR 一般不可唯一反演

不同 HR 子网格结构可能具有相同的 LR 平均量。例如：

- 情形 A：整个粗网格内的气体都低于恒星形成密度阈值；
- 情形 B：少量气体很冷、很致密，其余气体稀薄、炽热；
- 两者可以具有相同的平均密度、温度和速度；
- 但 A 的 SFR 为零，B 的 SFR 非零。

因此

\[
\boldsymbol U_{\rm LR}\to\boldsymbol U_{\rm HR}
\]

通常是一对多问题。确定性网络倾向于输出条件平均，而条件平均 HR 场会抹平小尺度间歇性、团块和爆发性。

若目标是非线性函数 \(f\)，还存在

\[
\mathbb E[f(\boldsymbol U_{\rm HR})
\mid\boldsymbol U_{\rm LR}]
\neq
f\!\left(
\mathbb E[\boldsymbol U_{\rm HR}
\mid\boldsymbol U_{\rm LR}]
\right).
\]

所以即使网络准确恢复了“平均 HR 场”，再将其送入阈值型 SFR 公式，也不一定得到正确的平均 SFR。

### 5.4 神经算子的分辨率不变性不等于物理粗粒化不变性

FNO、DeepONet 等神经算子学习函数空间之间的映射，并可能在同一连续问题的不同数值采样上评估。但需要区分：

1. 同一底层函数在不同数量网格点上的表示；
2. 改变物理滤波尺度 \(\Delta\)，从而改变未解析物理和闭合项。

本项目主要属于第二类问题。因此应学习带显式尺度参数的模型：

\[
\mathcal F_\theta[
\boldsymbol U_{\rm LR};
\Delta x,\Delta t,
\mathcal G_\Delta,
\text{solver/geometry}
].
\]

在固定 LR-HR 分辨率对上训练成功，不能证明模型可以跨物理分辨率泛化。

---

## 6. 修正后的研究目标

推荐把项目核心重新定义为：

> 学习一个分辨率条件化、概率性、带时间记忆、满足守恒约束的亚网格闭合算子，使其从宇宙学 LR 状态预测 MACER 所代表的未解析物理统计量或反馈源项。

统一形式可写成

\[
p_\theta\!\left(
\boldsymbol Y^{t+1}
\mid
\boldsymbol U_{\rm LR}^{t-k:t},
\Delta x,\Delta t,
\mathcal G_\Delta,
\boldsymbol\Theta_{\rm global}
\right),
\]

其中 \(\boldsymbol Y\) 可以包括：

\[
\boldsymbol Y=
(\Delta M_\star,
\dot M_{\rm BH},
\Delta M_{\rm wind},
\Delta\boldsymbol p_{\rm fb},
\Delta E_{\rm fb},
f_{\rm cold,dense},
K_{\rm SGS},\ldots).
\]

这里的“神经算子”不再被理解为唯一反演完整 HR 场，而是一个从粗粒化状态函数到闭合量、源项或条件分布的算子。

---

## 7. 是否必须使用 LR→HR→SFR/BHAR

结论：不必须，而且对于在线宇宙学模拟通常不是最合适的主路线。

| 路线 | 优点 | 主要问题 | 定位 |
|---|---|---|---|
| 确定性 LR→完整 HR→目标 | 场结构直观、便于可视化 | HR 非唯一；误差累积；计算昂贵；目标错位 | 不推荐作为生产主线 |
| 随机 LR→HR→目标 | 可表达多个可能的 HR 实现 | 训练和在线采样昂贵 | 独立科学支线 |
| LR→SFR/BHAR | 直接、部署成本低 | 解释性有限；必须表达不确定性 | 推荐生产路线之一 |
| LR→亚网格统计量→目标 | 物理可解释，较容易守恒和跨尺度 | 需要选择足够的统计量 | 最推荐 |
| 多任务：直接目标 + HR 统计辅助 | 兼顾目标精度与物理表征 | 训练设计更复杂 | 推荐最终方案 |

推荐的总体结构为：

\[
\boldsymbol U_{\rm LR}^{t-k:t}
\longrightarrow
\boldsymbol z_{\rm unresolved}
\longrightarrow
p(\mathrm{SFR/BHAR}),
\]

其中未解析状态 \(\boldsymbol z_{\rm unresolved}\) 可以包含：

- 冷致密气体质量分数；
- 子网格密度或温度 PDF 的参数；
- clumping factor；
- 亚网格湍流速度弥散或湍流能；
- 有效冷却时间；
- 有效自由落体时间；
- 未解析角动量或径向内流；
- 吸积盘储库状态。

完整 HR 场重建可作为辅助监督，不要求在宇宙学代码的每个时间步显式生成。

---

## 8. SFR 子项目的推荐定义

### 8.1 首先确认标签代表什么

MACER 文献中的常见恒星形成处方为

\[
\dot\rho_{\rm SFR}
=
\frac{\eta_{\rm SF}\rho}{\tau_{\rm SF}},
\]

并在满足密度和温度阈值时形成恒星，例如

\[
n>1\ {\rm cm^{-3}},
\qquad
T<4\times10^4\ {\rm K},
\]

且常取 \(\eta_{\rm SF}=0.01\)。具体公式、时间尺度和阈值必须以当前 MACER 数据对应的代码版本为准。

如果 `sfr` 本身完全由当前 snapshot 中的 \(\rho,T\) 和既有解析公式生成，那么训练网络复现 `sfr` 主要是在模拟已有亚网格公式，而不是从 HR 数据发现新的闭合关系。

本项目更有意义的监督目标应为粗网格内的 HR 总恒星形成量：

\[
\dot M_{\star,C}^{\rm HR}
=
\int_C \dot\rho_{\star,\rm HR}\,{\rm d}V,
\]

或者一个时间间隔内实际形成的恒星质量

\[
\Delta M_{\star,C}^{\rm HR}.
\]

### 8.2 推荐的可解释闭合形式

让网络预测

\[
f_{\rm cd}
=
\frac{M_{\rm cold,dense}}{M_{\rm gas}},
\qquad
t_{\rm eff}>0,
\]

然后通过确定性物理层计算

\[
\boxed{
\dot M_\star
=
\eta_{\rm SF}
\frac{f_{\rm cd}M_{\rm gas}}{t_{\rm eff}}
}.
\]

约束方法：

- 使用 sigmoid 保证 \(0\le f_{\rm cd}\le1\)；
- 使用 softplus 或指数变换保证 \(t_{\rm eff}>0\)；
- 强制 \(\Delta M_\star\le M_{\rm gas,available}\)；
- 输出 cell-integrated \(\Delta M_\star\)，再由代码转换为源项。

另一种稳健方案是对已有模型做乘性残差修正：

\[
\dot M_\star^{\rm AI}
=
\dot M_\star^{\rm base}
\exp\left[
\delta_\theta(
\boldsymbol U_{\rm LR},
\Delta,
\text{history})
\right].
\]

这允许模型在分布外或失效时退回解析基线。

### 8.3 SFR 应采用概率模型

SFR 数据通常包含大量零值和少量强爆发值，不适合只用普通 MSE。推荐 hurdle/zero-inflated 模型：

\[
p(\dot M_\star\mid X)
=
p_0(X)\,\delta(\dot M_\star)
+[1-p_0(X)]
p_+(\dot M_\star\mid X).
\]

模型分别预测：

1. 当前粗网格是否发生恒星形成；
2. 在发生恒星形成的条件下，正 SFR 或 \(\Delta M_\star\) 的分布。

正值部分可以使用 lognormal、Gamma、mixture density 或分位数回归。

---

## 9. BHAR/AGN 反馈子项目的推荐定义

BHAR 不应直接照搬 SFR 的 snapshot 模型。MACER 中的 BHAR 与内边界流量、自由落体延迟、吸积盘储库、黏滞时间和盘风损失有关。

可首先从解析场计算某半径上的内流率：

\[
\dot M_{\rm in}(r)
=
-\int
\rho\,\min(v_r,0)
r^2\,{\rm d}\Omega.
\]

随后学习从粗尺度内流到真实 BHAR 和反馈输出的动态闭合：

\[
p_\theta\!\left(
\dot M_{\rm BH}^{t+1},
M_{\rm disc}^{t+1},
\dot M_{\rm wind}^{t+1}
\mid
\boldsymbol U_{\rm LR}^{t-k:t},
M_{\rm disc}^{t},
\Delta
\right).
\]

适合的模型包括：

- GRU/LSTM 基线；
- state-space model；
- Neural ODE；
- 带隐状态的神经算子；
- operator encoder + 动态标量输出头。

单 snapshot 到 BHAR 的性能上限可能受到不可辨识性限制，因此必须比较单帧与时间序列模型。

---

## 10. 数据构造规范

### 10.1 保守粗粒化

禁止把普通图像 resize 作为唯一降采样方法。

粗网格密度使用体积加权：

\[
\bar\rho_C
=
\frac{\sum_{i\in C}\rho_iV_i}
{\sum_{i\in C}V_i}.
\]

速度使用质量加权/Favre 滤波：

\[
\widetilde{\boldsymbol v}_C
=
\frac{\sum_{i\in C}
\rho_i\boldsymbol v_iV_i}
{\sum_{i\in C}\rho_iV_i}.
\]

`Data-Set-10` 并不是总能量，而是气体内能密度。粗粒化时先积分得到单元内能，再重建粗网格内能密度：

\[
E_{{\rm int},C}
=
\sum_{i\in C}e_iV_i,
\qquad
\bar e_C
=
\frac{E_{{\rm int},C}}
{\sum_{i\in C}V_i}.
\]

不应从该字段中扣除解析动能；那种操作只适用于代码实际保存总能量密度的情形。温度不是独立粗粒化变量，而应始终由

\[
T_C=93\frac{\bar e_C}{\bar\rho_C}
\]

重新计算。resolved kinetic energy 可作为单独诊断，但不能与 `Data-Set-10` 混为总能量守恒。

SFR 标签使用 cell-integrated 形式：

\[
\dot M_{\star,C}
=
\sum_{i\in C}
\dot\rho_{\star,i}V_i.
\]

### 10.2 三层数据体系

#### 第一层：filtered-HR

- 对同一 HR 数据使用多个分辨率比，例如 2、4、8、16；
- 使用保守体积/质量加权；
- 模拟目标求解器的 kernel、mesh 或 averaging operator；
- 用于模型开发和快速消融。

#### 第二层：相同初始条件的真实多分辨率重运行

- HR 与 LR 分别运行；
- 比较真实 LR 与 filtered-HR 的差异；
- 量化数值求解器、时间演化和源项造成的 domain gap；
- 用于判断模型是否学到了真实闭合，而不仅是插值逆映射。

#### 第三层：目标宇宙学代码数据

- 运行少量受控 cosmological zoom-in 或局部 patch；
- 确认宇宙学 LR 输入与 MACER 降采样输入的分布差异；
- 进行多保真训练、微调或 domain adaptation；
- 最终完成在线耦合验证。

### 10.3 数据划分

不得随机打散所有 snapshot 后再划分训练/测试集。推荐按以下层级切分：

- 完整 simulation run；
- 初始条件；
- 黑洞质量或星系质量；
- AGN 反馈参数；
- 外部气体供给；
- 分辨率；
- MACER2D 与 MACER3D。

核心测试包括 leave-one-simulation-out 和 leave-one-resolution-out。

### 10.4 2D 到 3D 的处理

MACER2D 到 MACER3D 不是单纯增加一个空间维度，而是从轴对称动力学转向可能包含非轴对称不稳定性、湍流级联和角动量输运的三维系统。

推荐：

- 使用大量 MACER2D 数据预训练；
- 使用较少 MACER3D 数据做多保真校正；
- 单独报告 2D 内部泛化与 2D→3D 迁移；
- 不把 2D→3D 失败错误解释为单纯的分辨率问题。

---

## 11. 输入变量和导出物理特征

### 11.1 数据一致性检查

应检查：

\[
M_{\rm gas,C}
\simeq
\rho_CV_C
\]

在考虑数密度到质量密度、平均分子量和单位变换后是否成立。

检查 `T` 与 `e` 是否满足对应状态方程。检查

\[
M_{\rm newstar}
\simeq
\dot\rho_{\rm SFR}V\Delta t
\]

或该数据版本实际采用的时间积分关系。

### 11.2 推荐导出特征

在原始场之外，可计算：

\[
t_{\rm ff}
=
\sqrt{\frac{3\pi}{32G\rho}},
\qquad
\frac{t_{\rm cool}}{t_{\rm ff}},
\]

\[
\mathcal M
=
\frac{|\boldsymbol v-\boldsymbol v_{\rm bulk}|}{c_s},
\qquad
\nabla\cdot\boldsymbol v,
\qquad
|\nabla\times\boldsymbol v|,
\]

以及在数据允许时计算：

- Jeans 长度及 \(\Delta/\lambda_{\rm J}\)；
- 旋转/自由落体时间；
- 局部角动量；
- 径向质量流率；
- 局部势阱深度或逃逸速度；
- 金属丰度；
- 冷却长度；
- 邻域密度/温度 PDF 的低阶矩。

球坐标中的梯度、散度和旋度必须包含正确的球坐标度规因子。

### 11.3 输入泄漏原则

若 `abun_wind`、`abun_jet` 等 tracer 是目标反馈事件的直接结果，它们可能泄露答案。原则是：

- 部署时存在且因果上可用：可以输入；
- 只在 MACER 中存在：只能辅助监督；
- 包含未来状态：禁止输入；
- 定义与目标宇宙学代码不同：需要建立明确转换后才能使用。

---

## 12. 模型架构建议

### 12.1 强基线优先

在引入大型神经算子前，必须建立：

1. 原 MACER 解析 SFR/BHAR 公式；
2. 线性/广义线性模型；
3. MLP 或 XGBoost；
4. 现有 CNN；
5. CNN/图网络 + 时间编码；
6. 概率性 direct closure；
7. 可解释 latent closure。

神经算子必须在真正 OOD 和在线测试中超过这些基线，而不是只在随机 snapshot 测试上降低 MSE。

### 12.2 神经算子的适用位置

如果输入和输出都是空间函数，例如

\[
\boldsymbol U_{\rm LR}(\boldsymbol x)
\to
f_{\rm cold,dense}(\boldsymbol x),
K_{\rm SGS}(\boldsymbol x),
S_E(\boldsymbol x),
\]

神经算子具有明确意义。

如果输出只是一个星系总 SFR 或单一 BHAR 标量，神经算子编码器可以使用，但不应假设它必然优于更简单的空间编码器加标量输出头。

### 12.3 几何选择

- MACER2D 规则球坐标网格：可尝试 geometry-aware FNO、UNO、DeepONet；
- 不规则网格、moving mesh、SPH：优先图神经算子；
- 三维旋转协变问题：考虑 E(3)-equivariant 图网络；
- 多尺度径向 profile：DeepONet 或 profile encoder + state-space model 可能比笛卡尔 FNO 更合适。

架构应把求解器适配层与物理闭合核心分离，使 MACER 球网格输入和宇宙学 particle/cell 输入能映射到统一的局部无量纲物理表征。

---

## 13. 物理约束设计

### 13.1 优先使用硬约束

- \(\rho,T,\mathrm{SFR},\mathrm{BHAR}\ge0\)；
- \(0\le f_{\rm cold,dense}\le1\)；
- \(\Delta M_\star+\Delta M_{\rm BH}\le M_{\rm gas,available}\)；
- 反馈注入的质量、动量和能量预算成套输出；
- 使用相对黑洞或局部质心速度，避免依赖绝对惯性系；
- 使用无量纲输入；
- 显式输入 \(\Delta x,\Delta t\) 和滤波定义；
- 使用已知对称性和坐标几何；
- 对 OOD 输入提供解析 fallback。

### 13.2 PDE 约束必须作用于过滤方程

不应要求 LR 场满足没有 SGS 项的 HR Euler 方程。推荐约束形式为

\[
\mathcal R_{\rm filtered}
+\nabla\cdot\tau_\theta^{\rm SGS}
-S_\theta
=0.
\]

如果直接把未过滤 Euler residual 压到零，模型可能被迫消除原本需要学习的亚网格物理。

### 13.3 推荐损失函数

一个可能的总损失为

\[
\mathcal L
=
\mathcal L_{\rm target}
+\lambda_{\rm stat}\mathcal L_{\rm HR-stat}
+\lambda_{\rm cons}\mathcal L_{\rm conservation}
+\lambda_{\rm roll}\mathcal L_{\rm rollout}
+\lambda_{\rm spec}\mathcal L_{\rm spectrum}
+\lambda_{\rm cal}\mathcal L_{\rm calibration}.
\]

其中：

- `target`：SFR/BHAR 或反馈源项；
- `HR-stat`：冷致密气体比例、PDF 矩、功率谱等；
- `conservation`：积分质量、动量、能量预算；
- `rollout`：多步时间演化误差；
- `spectrum`：时域或空间谱统计；
- `calibration`：概率输出是否校准。

对于跨多个数量级的速率，应比较 log-space、相对误差、分位数和概率似然，不能只报告全场 L2。

---

## 14. 推荐项目流程

### Phase 0：固定科学问题和部署契约

需要明确：

- 第一阶段目标是 SFR、\(\Delta M_\star\) 还是 BHAR；
- 输出是 cell-local、patch-integrated 还是 galaxy-integrated；
- 目标宇宙学代码及其数据结构；
- 目标分辨率范围；
- 推理时间间隔；
- 运行时可获得的输入变量；
- 是否要求输出完整反馈质量、动量和能量源项。

目标代码尚未固定时，不应把某一种滤波核或网络几何当作最终标准。若强调完整开源和便于修改，可优先评估 SWIFT 生态；若课题组已有 AREPO/TNG 工作流，则可以为保持科学连续性而优先沿现有工作流验证。最终选择应由代码可修改性、已有初始条件/分析工具和在线耦合成本共同决定。

### Phase 1：数据和标签审计

交付物：

- 数据字典和单位表；
- 坐标/网格说明；
- `d-mgas-vol` 一致性报告；
- `T-e` 状态方程报告；
- `sfr-mnewstar-dt` 关系报告；
- tracer 可部署性清单；
- 时间对齐和缺失值报告；
- 模拟级 train/validation/test 清单。

### Phase 2：构建保守多分辨率数据集

交付物：

- 统一 conservative coarse-graining 模块；
- 多个绝对 \(\Delta\) 和分辨率比；
- 原始场、粗粒化场和粗粒化目标；
- 滤波核与目标求解器 kernel 的元数据；
- 少量真实 LR 重运行对。

### Phase 3：建立强基线

比较：

- 解析处方；
- MLP/XGBoost；
- CNN；
- 时序 CNN/GNN；
- direct probabilistic closure；
- latent physical closure。

此阶段目标是识别泛化差究竟来自模型、数据、不可辨识性还是 domain shift。

### Phase 4：神经算子和多任务模型

建议优先顺序：

1. 分辨率条件化的 direct closure；
2. LR→未解析统计量→SFR；
3. 加入时间历史；
4. 加入 HR 统计辅助任务；
5. 图神经算子或 geometry-aware NO；
6. 最后再测试随机完整 HR 重建。

### Phase 5：严格 OOD 验证

必须至少包括：

- 未见分辨率；
- 未见 simulation run；
- 未见黑洞/星系参数；
- 未见反馈强度；
- 2D→3D；
- filtered-HR→真实 LR；
- MACER→目标宇宙学代码。

### Phase 6：在线 a posteriori 测试

顺序：

1. shadow mode：推理但不修改流体；
2. AI 仅修正解析模型；
3. 启用局部源项但限制幅度；
4. 启用完整反馈预算；
5. 长时间积分；
6. 小规模 cosmological zoom-in；
7. 与解析亚网格模型和 HR/MACER 结果比较。

### Phase 7：宇宙学 zoom-in 科学验证

最终不只比较网络误差，还应比较：

- 星系总 SFR 和恒星质量增长；
- BHAR 分布和 AGN duty cycle；
- 冷/热气体质量；
- 径向气体、温度和金属分布；
- 流入/流出率；
- 黑洞-星系标度关系；
- 长时间质量和能量预算；
- 不同分辨率下结果是否收敛。

---

## 15. 评价指标与通过标准

### 15.1 离线指标

- cell/patch/galaxy-integrated SFR 或 BHAR 误差；
- 对数误差和分位数误差；
- SFR=0 的分类性能；
- 概率校准、coverage 和 sharpness；
- 径向 profile；
- 冷致密气体质量；
- 空间功率谱；
- 时间自相关、功率谱和爆发持续时间；
- 质量、动量和能量守恒误差。

### 15.2 核心 OOD 指标

- leave-one-resolution-out；
- filtered-HR 与真实 LR 的性能差；
- leave-one-simulation-out；
- 2D→3D；
- 目标宇宙学代码输入上的性能。

### 15.3 在线指标

- 长时间数值稳定性；
- 是否造成非物理负密度/负能量；
- 是否缩短时间步到不可接受程度；
- 是否产生反馈能量漂移；
- 与解析模型相比的额外推理成本；
- 科学统计量是否优于或至少不劣于原亚网格模型。

### 15.4 Go/No-Go 判据

只有满足以下条件，才认为模型具有嵌入宇宙学模拟的价值：

1. 在按 simulation 划分的测试集上超过简单基线；
2. 在未见分辨率上仍有稳定提升；
3. 对真实 LR 而不只是 filtered-HR 有效；
4. 概率输出得到合理校准；
5. 在线积分稳定且守恒；
6. zoom-in 科学统计量显示明确改善。

如果模型只在训练过的固定分辨率对上恢复 HR 图像，不应宣称获得了通用亚网格闭合。

---

## 16. 当前建议的最小可行项目（MVP）

第一篇或第一阶段工作可收敛为：

> 使用 MACER2D 多分辨率数据，构建分辨率条件化、概率性且质量守恒的 SFR 闭合模型；比较 direct closure、latent cold-dense-gas closure 和 LR→HR→SFR 三条路线，并在未见模拟和未见分辨率上测试。

建议 MVP 输入：

- \(\log\rho\)；
- \(\log T\) 或内能中的一个主状态量；
- Favre 速度分量；
- cell 体积；
- 半径/坐标；
- \(t_{\rm cool}/t_{\rm ff}\)；
- Mach number；
- 散度和涡量；
- \(\Delta x,\Delta t\)；
- 最近若干 snapshot 的历史。

建议 MVP 输出：

- \(p(\mathrm{SFR}=0)\)；
- 正 \(\Delta M_\star\) 的概率分布；
- 冷致密气体质量分数作为辅助输出；
- 可选 clumping factor 或子网格密度方差。

建议 MVP 模型：

1. CNN/图编码器 + probabilistic head；
2. 同一编码器 + latent physical head；
3. resolution-conditioned neural operator；
4. 确定性 super-resolution 仅作为比较基线。

主要实验矩阵：

| 训练条件 | 测试条件 | 目的 |
|---|---|---|
| 固定分辨率 | 同分辨率新 snapshot | 基础插值能力 |
| 多分辨率 | 留出一个分辨率 | 尺度外推 |
| 多个 run | 留出完整 run | 物理参数泛化 |
| filtered-HR | 真实 LR 重运行 | domain gap |
| MACER2D | MACER3D | 维度/对称性迁移 |

---

## 17. 尚未解决、需要课题组确认的问题

后续 AI 或协作者不能自行假定以下问题已经解决：

1. 当前 `sfr` 的精确定义、单位和计算公式是什么？
2. `mnewstar` 对应哪个时间区间，是否与当前 snapshot 对齐？
3. 当前数据中的 `d` 是哪一种数密度，平均分子量如何处理？
4. 已确认 `Data-Set-10` 的 `e` 是气体内能密度；仍需在迁移外部数据时核对其代码单位。
5. MACER2D/3D 的具体网格、边界条件和 snapshot cadence 是什么？
6. 现有 CNN 的训练/验证划分方式是什么？
7. 所谓“泛化差”是跨 snapshot、跨 simulation、跨分辨率还是跨参数？
8. 是否已有相同初始条件的真实多分辨率 MACER 运行？
9. 第一阶段目标宇宙学代码是什么？
10. 目标代码运行时能否提供 `coolt`、金属丰度、恒星势和 tracer？
11. 最终需要局部 SFR source、星系总 SFR，还是生成 star particles 的概率？
12. 对 BHAR，是否可以获得连续时间序列和内边界流量/盘储库变量？
13. MACER3D 数据量是否足以作为独立测试集或微调集？

---

## 18. 给后续 AI/协作者的工作约束

处理本项目时，应遵循以下原则：

- 不把“相同流体方程”误解为 LR 可以唯一恢复 HR；
- 不把神经算子的 discretization invariance 解释为物理分辨率不变性；
- 不用普通图像 resize 替代保守粗粒化；
- 不把 filtered-HR 验证等同于真实 LR 模拟验证；
- 不随机打散强时间相关 snapshot 作为主要数据划分；
- 不在没有时间信息的情况下默认 snapshot 足以确定 BHAR；
- 不将部署时不可获得的 tracer 作为生产模型必需输入；
- 不只报告像素 MSE；
- 不把未经在线 rollout 验证的模型称为可嵌入宇宙学模拟；
- 不因模型被称为 PINO/NO/physics AI 就默认其满足物理；
- 优先使用硬守恒结构，再使用软 PDE loss；
- 明确区分第一性原理方程、过滤后闭合项、MACER 亚网格处方和 AI 经验模型。

---

## 19. 关键文献

### MACER

1. Zhu et al., *On the dominant role of wind in the quasar feedback mode in the late-stage evolution of massive elliptical galaxies*, MNRAS 525, 4840：
   <https://academic.oup.com/mnras/article/525/4/4840/7259172>
2. Yuan et al., MACER 主要框架与 AGN 反馈模型：
   <https://arxiv.org/abs/1807.05488>
3. MACER 项目文献入口：
   <https://macer-project.github.io/>

### 神经算子

4. Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*：
   <https://arxiv.org/abs/2010.08895>
5. Lu et al., *DeepONet: Learning nonlinear operators*：
   <https://arxiv.org/abs/1910.03193>
6. Finzi et al., E(3)-equivariant neural networks：
   <https://arxiv.org/abs/2101.03164>

### 粗粒化和亚网格闭合

7. Schmidt, *Large Eddy Simulations in Astrophysics*：
   <https://link.springer.com/article/10.1007/lrca-2015-2>
8. *Multiscale Neural Operator: Learning Fast and Grid-independent PDE Solvers*：
   <https://arxiv.org/abs/2207.11417>
9. *Learning Subgrid-scale Models with Neural Ordinary Differential Equations*：
   <https://arxiv.org/abs/2212.09967>

### 与黑洞反馈闭合接近的近期工作

10. *From Black Hole to Galaxy: Neural Operator — Framework for Accretion and Feedback Dynamics*，较新的预印本，结论需谨慎看待：
    <https://arxiv.org/abs/2512.01576>

---

## 20. 一句话项目定义

> MACNet 的目标不是从宇宙学 LR 场唯一“画出”MACER HR 场，而是利用 MACER 多尺度数据学习一个分辨率条件化、概率性、带记忆且守恒的闭合算子，从而在宇宙学模拟中预测未解析的恒星形成、黑洞吸积和 AGN 反馈预算。
