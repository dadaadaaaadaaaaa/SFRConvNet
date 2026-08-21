# MACNet：空间尺度与质量尺度粗粒化算子的比较研究方案

> 文档用途：供后续课题组成员、代码协作者和 AI 助手共同读取，作为实现、训练、评估与结论表述的统一依据。  
> 当前日期：2026-08-16  
> 当前阶段：已有 MACER2D 数据、守恒的 \(\mathcal R_h\)、axisymmetric \(\mathcal R_m\)-v0/v1 和 direct CNN 训练结果；已完成 30-snapshot 数值审计及 v0/v1 配对比较；尚未实现完整 MACER-LR SFR baseline 与 residual-closure 训练，也尚无真实 GIZMO snapshot。

---

## 1. 研究背景

本项目希望利用高分辨率 MACER 模拟所包含的星系尺度小尺度物理信息，构建能够在低分辨率宇宙学模拟中使用的恒星形成率（SFR）闭合模型。目标代码已确定为 GIZMO，计划使用其 MFM 求解器。

MACER 与 GIZMO 的离散方式不同：

- 当前 MACER2D 数据位于规则的球极坐标网格上，数组形状为

  \[
  (n_\theta,n_r)=(72,280),
  \]

  其中 \(r\) 近似对数均匀，\(\theta\) 均匀；
- GIZMO/MFM 使用随流体运动的近似固定质量流体单元，其局部空间尺度随密度变化。

因此，需要比较至少两类人工粗粒化算子：

1. 固定空间/固定索引块粗粒化 \(\mathcal R_h\)；
2. 固定质量预算对应的自适应尺度粗粒化 \(\mathcal R_m\)。

比较的科学问题不是“规则网格和不规则网格哪个更高级”，而是：

> 哪一种粗粒化方式更接近未来 GIZMO 输入所经历的信息损失，并使“LR 物理基线 + 学习到的闭合修正”在未见分辨率和未见模拟上最准确、最稳定？闭合项本身的幅度只是诊断量，不是最终优化目标。

---

## 2. 核心数学定义

令 HR-MACER 状态为

\[
\boldsymbol U_{\rm HR}
=
(\rho,\rho\boldsymbol v,E,\rho_*,\ldots).
\]

MACER 在 HR 状态上的恒星形成算子记为

\[
\dot\rho_{\star,\rm HR}
=
\mathcal S_{\rm HR}(\boldsymbol U_{\rm HR}).
\]

在粗分辨率状态上实现的同一物理处方记为 \(\mathcal S_\alpha\)。单独使用下标是必要的，因为梯度、阈值、自由落体时间以及其他派生量应在 LR 表示上重新计算，不能默认 \(\mathcal S_{\rm HR}\) 与 \(\mathcal S_\alpha\) 的离散实现完全相同。

对于任意粗粒化算子 \(\mathcal R_\alpha\)，定义：

### 2.1 粗粒化输入

\[
\boldsymbol X_\alpha
=
\mathcal R_\alpha(\boldsymbol U_{\rm HR}).
\]

### 2.2 正确的 HR 粗粒化目标

\[
Y_\alpha
=
\mathcal R_\alpha
\mathcal S_{\rm HR}(\boldsymbol U_{\rm HR}).
\]

这里 \(Y_\alpha\) 是与 LR 网格对应的场级真值，而不是 CNN 的输入。

### 2.3 在粗粒化状态上直接应用物理公式得到的基线

\[
B_\alpha
=
\mathcal S_\alpha
\left[
\mathcal R_\alpha(\boldsymbol U_{\rm HR})
\right]
=
\mathcal S_\alpha(\boldsymbol X_\alpha).
\]

这就是需要实现的 MACER-LR 物理基线。它必须由 LR 状态重新计算，不能由 HR SFR 反推。

### 2.4 真实闭合项、CNN 学习对象与修正后预测

\[
\boxed{
\mathcal C_\alpha^{\rm true}
=
\mathcal R_\alpha\mathcal S_{\rm HR}(\boldsymbol U_{\rm HR})
-
\mathcal S_\alpha
\left[
\mathcal R_\alpha(\boldsymbol U_{\rm HR})
\right]
}
\]

即

\[
\mathcal C_\alpha^{\rm true}=Y_\alpha-B_\alpha.
\]

CNN 或神经算子学习的对象不是 \(Y_\alpha\) 本身，而是闭合项的预测器：

\[
\boxed{
\widehat{\mathcal C}_\theta
=
\mathcal F_\theta
(\boldsymbol X_\alpha,\boldsymbol q_\alpha)
\approx
\mathcal C_\alpha^{\rm true}
}
\]

计划中的修正后预测为

\[
\boxed{
\widehat Y_\alpha
=
B_\alpha
+
\widehat{\mathcal C}_\theta
(\boldsymbol X_\alpha,\boldsymbol q_\alpha)
}
\]

其中 \(\boldsymbol q_\alpha\) 是分辨率与算子元数据，例如：

\[
\boldsymbol q_h=(h,\Delta r,r\Delta\theta,V^{1/3}),
\]

或

\[
\boldsymbol q_m=(M_{\rm filter},H_{\rm eff},N_{\rm eff},V^{1/3}).
\]

HR 状态、HR SFR 和 \(\mathcal C_\alpha^{\rm true}\) 只用于训练标签构造；部署时模型只能获得 LR 状态、分辨率/算子元数据以及可在 LR 上计算的物理基线。可选地可把 \(B_\alpha\) 本身作为额外条件输入，但它无论是否进入网络，都必须作为最终预测的显式加法基线。

---

## 3. 对研究目标的一项必要修正：区分三个目标

原表述容易把“闭合项幅度小”和“最终模型预测准确”混为同一个目标。两者有关，但不等价。本项目必须区分粗粒化诊断、网络训练和最终模型选择三个层级。

### 3.1 \(\mathcal C_\alpha^{\rm true}\) 是数据决定的标签，不是可被 CNN 变小的量

固定 \(\mathcal R_\alpha\)、\(\mathcal S_{\rm HR}\)、\(\mathcal S_\alpha\) 和 HR snapshot 后，

\[
\mathcal C_\alpha^{\rm true}
=Y_\alpha-B_\alpha
\]

已经由数据唯一确定。CNN 训练不能改变它，只能让

\[
\widehat{\mathcal C}_\theta
\]

逼近它。因此 CNN 的直接学习目标是

\[
\boxed{
\min_\theta
\mathcal L_{\rm closure}
=
\min_\theta
\left\|
\widehat{\mathcal C}_\theta
(\boldsymbol X_\alpha,\boldsymbol q_\alpha)
-
\mathcal C_\alpha^{\rm true}
\right\|^2
}
\]

而不是

\[
\min_\theta\left\|\mathcal C_\alpha^{\rm true}\right\|.
\]

### 3.2 最终优化目标是修正后预测逼近 HR 粗粒化真值

真正用于科学评价与部署的量是

\[
\widehat Y_\alpha
=
B_\alpha
+
\widehat{\mathcal C}_\theta.
\]

最终目标为

\[
\boxed{
\min_\theta
\mathcal L_{\rm final}
=
\min_\theta
\left\|
B_\alpha
+
\widehat{\mathcal C}_\theta
-
Y_\alpha
\right\|^2
}
\]

因为 \(\mathcal C_\alpha^{\rm true}=Y_\alpha-B_\alpha\)，在普通平方误差下，\(\mathcal L_{\rm closure}\) 与 \(\mathcal L_{\rm final}\) 数学等价。但第二种写法更准确地表达了项目目的：不是追求闭合修正本身等于零，而是追求物理基线经过修正后等于真值。

### 3.3 当前全局 SFR 任务的精确定义

引入从局部 SFR 场到 snapshot 总 SFR 的归约算子 \(\mathcal A\)。如果 Data-Set-11 是 SFR 密度，\(\mathcal A\) 是体积积分；如果它已经是每格 SFR 贡献，\(\mathcal A\) 是直接求和。当前配置使用 reduction=sum，但仍须从 MACER 输出代码确认 Data-Set-11 的单位语义。

当前任务的真值为

\[
Y_{\rm true}^{\rm global}
=
\mathcal A
\left[
\mathcal R_\alpha
\mathcal S_{\rm HR}(\boldsymbol U_{\rm HR})
\right].
\]

全局 MACER-LR baseline 为

\[
B_\alpha^{\rm global}
=
\mathcal A
\left[
\mathcal S_\alpha(\boldsymbol X_\alpha)
\right].
\]

全局闭合标签为

\[
C_{\alpha,\rm global}^{\rm true}
=
Y_{\rm true}^{\rm global}
-
B_\alpha^{\rm global}.
\]

在当前 snapshot 级 CNN 路线中，网络应输出一个标量残差：

\[
\widehat C_{\theta,\rm global}
=
\mathcal F_\theta
(\boldsymbol X_\alpha,\boldsymbol q_\alpha).
\]

最终预测为

\[
\boxed{
\widehat Y_{\rm global}
=
B_\alpha^{\rm global}
+
\widehat C_{\theta,\rm global}
}
\]

并优化

\[
\boxed{
\mathcal L_{\rm global}
=
\ell
\left(
B_\alpha^{\rm global}
+
\widehat C_{\theta,\rm global},
Y_{\rm true}^{\rm global}
\right).
}
\]

只有在未来保存并监督逐 cell 或逐 patch 的 SFR source 后，网络才学习场级 \(\widehat{\mathcal C}_\theta(\boldsymbol x)\)。不能把当前全局标量监督描述成已经学习了完整的局部闭合场。

### 3.4 \(\|\mathcal C_\alpha^{\rm true}\|\) 的正确地位

在比较 \(\mathcal R_h\) 与 \(\mathcal R_m\) 时，闭合项幅度仍然有诊断价值，但不是最终优化目标，也不能单独决定哪个算子更好。特别是当粗粒化很弱时，

\[
\mathcal R_\alpha\rightarrow I
\quad\Rightarrow\quad
\mathcal C_\alpha^{\rm true}\rightarrow0,
\]

这并不能证明该算子适合目标宇宙学分辨率。另一方面，幅度较大但结构稳定的闭合项，可能比幅度较小却高度随机的闭合项更容易预测。

因此，在匹配计算预算、信息预算或有效分辨率后，算子选择应按以下优先级评价：

1. 修正后预测 \(B_\alpha+\widehat{\mathcal C}_\theta\) 对真值 \(Y_\alpha\) 的误差；
2. 相比未修正物理基线 \(B_\alpha\) 的提升，以及是否优于 direct CNN；
3. 在未见分辨率、未见 MACER run 和未来 GIZMO snapshot 上的泛化；
4. 闭合残差是否稳定、可预测，而不仅是平均幅度是否较小；
5. 对核函数、网格相位、边界处理和分辨率元数据的鲁棒性；
6. 守恒性、计算开销以及与未来 GIZMO 输入分布的距离；
7. \(\|\mathcal C_\alpha^{\rm true}\|\) 的幅度，作为解释上述结果的辅助诊断量。

### 3.5 当前代码与目标方案的区别

上述公式描述的是计划实现的 residual-closure 模型。当前 MACNET-v1 仍主要执行

\[
\widehat Y_{\rm direct}
=
\mathcal F_\theta(\boldsymbol X_\alpha)
\]

的全局 SFR 直接预测；现有 sfr_ks/FiLM 不等价于显式的 MACER-LR baseline 加法残差。只有在实现并保存

\[
B_\alpha^{\rm global}
=
\mathcal A[\mathcal S_\alpha(\boldsymbol X_\alpha)]
\]

并将训练标签改为

\[
C_{\alpha,\rm global}^{\rm true}
=Y_{\rm true}^{\rm global}-B_\alpha^{\rm global}
\]

后，才能称为完整的 MACER-LR 闭合学习。

---

## 4. 当前代码与数据的真实状态

### 4.1 已实现的 \(\mathcal R_h\)

当前 [`raw_data_processing/galaxy_data.py`](raw_data_processing/galaxy_data.py) 中的 `GalData.rescale()` 已实现一套守恒的固定索引块粗粒化算子。

它不是普通图像均值池化，而是使用 MACER `Data-Set-15` 提供的真实单元体积，进行以下操作：

#### 质量

\[
M_C=\sum_{k\in C}\rho_kV_k,
\qquad
\rho_C=\frac{M_C}{V_C},
\qquad
V_C=\sum_{k\in C}V_k.
\]

#### 气体内能密度

`Data-Set-10` 的 `e` 是气体内能密度。代码为兼容旧缓存仍使用字段名
`gas_energy`，但其单元积分量是气体内能 $E_{{\rm int},k}=e_kV_k$：

\[
e_C
=
\frac{\sum_{k\in C}e_kV_k}{V_C}.
\]

#### 速度与动量

\[
\boldsymbol P_C
=
\sum_{k\in C}M_k\boldsymbol v_k,
\qquad
\boldsymbol v_C
=
\frac{\boldsymbol P_C}{M_C}.
\]

#### 初始恒星密度

\[
\rho_{*,C}
=
\frac{\sum_{k\in C}\rho_{*,k}V_k}{V_C}.
\]

温度由粗粒化后的 `gas_energy` 和 `density` 重新计算：

\[
T_C=93\frac{e_C}{\rho_C}.
\]

代码还会自动检查总体积、气体质量、由内能密度积分得到的气体内能、三个方向的动量及初始恒星质量是否在给定数值容差内守恒。`Data-Set-10` 不是总能量，也不是单元内能。

### 4.2 当前固定粗化尺度

预处理入口目前固定执行：

```python
file_coarse = file.rescale(new_shape=(8, 8))
```

即

\[
72\times280\rightarrow8\times8,
\]

每个 coarse cell 包含

\[
9\times35=315
\]

个 fine cells。

该算子更准确地应称为：

> 在均匀 \(\theta\) 与近似均匀 \(\log r\) 计算坐标中的守恒固定分块算子。

目前代码已经支持任何能够整除原数组维度的 `new_shape`，但主数据管线只生成了 \(8\times8\) 数据。

### 4.3 当前监督目标是全局标量 SFR

`.config/HdfraConfig.json` 当前把 `Data-Set-11` 定义为 SFR 标签，并采用：

```json
"reduction": "sum"
```

所以现有 CNN 预测的是每个 snapshot 的全局标量：

\[
Y^{\rm global}
=
\sum_k {\rm SFR}_k.
\]

当前代码没有把局部 SFR 场作为模型输出，也没有把局部 SFR 场随 \(\mathcal R_h\) 一起保存到 coarse HDF5。

因此，现阶段适用的闭合定义首先是全局形式：

\[
\mathcal C_\alpha^{\rm global}
=
Y_{\rm HR}^{\rm global}
-B_\alpha^{\rm global}.
\]

未来若转为逐 cell 或逐 patch 的 SFR source 预测，才使用场级形式：

\[
\mathcal C_{\alpha,i}
=
Y_{\alpha,i}-B_{\alpha,i}.
\]

### 4.4 当前物理基线不是 MACER-LR SFR

现有训练管线计算并保存的是全局 Kennicutt-Schmidt 基线 `sfr_ks`。它可作为 FiLM 条件输入，但当前 `SFRConvNet` 的输出头仍直接预测 SFR，而不是显式预测

\[
Y-B.
\]

所以必须区分：

- 已实现：KS baseline、可选 FiLM、CNN 直接预测；
- 尚未实现：把完整或明确近似的 MACER SFR 公式应用于 \(\mathcal R_\alpha\boldsymbol U_{\rm HR}\)，得到 \(B_\alpha\)；
- 尚未实现：真正的残差训练

  \[
  \widehat Y=B_\alpha+\widehat{\mathcal C}_\theta.
  \]

在 MACER-LR baseline 尚未实现前，可以先比较不同粗粒化输入下的 direct CNN 泛化，但不能声称已经比较了 \(\mathcal C_h\) 与 \(\mathcal C_m\) 的完整物理闭合误差。

### 4.5 当前泛化评估基础

项目已经具备 leave-one-galaxy-out 数据划分与分析流程。后续 \(\mathcal R_m\) 与 \(\mathcal R_h\) 的比较应复用同一组 simulation-level 划分，避免相邻 snapshot 泄漏。

---

## 5. 两种算子的本质区别

### 5.1 固定空间/固定块算子 \(\mathcal R_h\)

固定空间尺度滤波可抽象写成

\[
\overline{\boldsymbol U}_h(\boldsymbol x)
=
\int G_h(\boldsymbol x-\boldsymbol y)
\boldsymbol U(\boldsymbol y)\,{\rm d}V.
\]

对于当前规则分块实现，所有 coarse cells 在 \((\theta,\log r)\) 计算坐标中包含相同数量的 fine cells，但物理长度、体积和质量并不相同。

它的主要优点是：

- 实现简单；
- 输出天然是规则矩阵；
- 守恒关系清晰；
- 训练稳定；
- 容易构造严格嵌套的多分辨率层级。

主要限制是：

- 不是 GIZMO/MFM 的近似固定质量分辨方式；
- 相同的索引块在不同半径对应不同物理尺度；
- 固定分块可能在高密度与低密度区消除不同质量的信息；
- 当前 \(8\times8\) 分块非常粗，且径向与角向粗化程度不对称。

### 5.2 固定质量预算算子 \(\mathcal R_m\)

质量尺度粗粒化要求局部滤波支持域内包含近似固定质量：

\[
\int_{\Omega(\boldsymbol x,H)}
\rho(\boldsymbol y)\,{\rm d}V
\simeq
M_{\rm filter}.
\]

对于真正的局部三维球形支持域，在密度近似均匀时：

\[
H(\boldsymbol x)
\propto
\left(
\frac{M_{\rm filter}}{\rho(\boldsymbol x)}
\right)^{1/3}.
\]

因此：

- 高密度区域使用较小空间尺度；
- 低密度区域使用较大空间尺度；
- 它改变的不只是网格形状，而是“小尺度信息随密度被消除的方式”。

这更接近 GIZMO/MFM 的自适应解析能力，但并不保证一定产生更小的闭合误差或更好的泛化。

但上式 **不能直接用于当前 axisymmetric \(\mathcal R_m\)**。MACER2D cell 表示完整环带；远离轴线且 \(H\ll R\) 时，子午面圆形邻域旋转后近似为环面管：

\[
M_{\rm enc}\sim 2\pi^2R\rho H^2,
\qquad
H\propto
\left(\frac{M_{\rm filter}}{R\rho}\right)^{1/2}.
\]

靠近对称轴时才逐渐转向三维尺度行为。因此必须把“轴对称整环质量滤波”和“未来局部三维质量滤波”的尺度律、质量语义与实验结论分开。

---

## 6. 第一版 \(\mathcal R_m\) 的推荐定义

### 6.1 不采用“先生成不规则等质量块，再重新光栅化”作为主实现

该方案可以守恒，但存在以下问题：

- 等质量区域的空间分区不唯一；
- 区域形状不规则，难以直接输入 CNN；
- 再映射到规则网格后，最终规则单元不再等质量；
- 容易发生“质量聚合 + 规则光栅化”的双重滤波；
- 训练结果可能依赖区域生长算法。

因此第一版推荐直接在共同规则输出网格上实现“自适应核的守恒沉积”。

### 6.2 源单元自适应核

令原始 HR fine cell 为 \(k\)，其广延量包括：

\[
M_k=\rho_kV_k,
\]

\[
\boldsymbol P_k=M_k\boldsymbol v_k,
\]

\[
E_k=e_kV_k,
\]

以及初始恒星质量

\[
M_{*,k}=\rho_{*,k}V_k.
\]

若未来构造局部 SFR 标签，还包括

\[
\dot M_{\star,k}.
\]

对每个源单元求自适应支持尺度 \(H_k\)，使

\[
M_{\rm enc}(\boldsymbol x_k,H_k)
=
\sum_{l\in\Omega(\boldsymbol x_k,H_k)}M_l
\simeq M_{\rm filter}.
\]

第一版可以采用单调累计质量与二分/邻居排序求解 \(H_k\)。

### 6.3 守恒沉积到规则矩阵

设输出规则单元为 \(B_i\)。定义源单元 \(k\) 对目标单元 \(i\) 的未归一权重

\[
\widetilde f_{ik}
=
\int_{B_i}
W(\boldsymbol x-\boldsymbol x_k,H_k)
\,{\rm d}V.
\]

当前实现不能只做每个源单元的列归一化。仅满足

\[
\sum_i f_{ik}=1
\]

虽然能保证广延量总和守恒，却不能保证常数密度场经过“自适应核 + 规则输出格”后仍保持常数。实际采用双边缘守恒输运。令非负稀疏核为 \(K_{ik}=\widetilde f_{ik}\)，求

\[
T_{ik}=a_iK_{ik}b_k
\]

使其同时满足

\[
\sum_iT_{ik}=V_k,
\qquad
\sum_kT_{ik}=V_i.
\]

再定义源单元广延量的输运分数

\[
f_{ik}=\frac{T_{ik}}{V_k},
\qquad
\sum_if_{ik}=1.
\]

将广延量沉积为：

\[
M_i=\sum_k f_{ik}M_k,
\]

\[
\boldsymbol P_i=\sum_k f_{ik}\boldsymbol P_k,
\]

\[
E_i=\sum_k f_{ik}E_k,
\]

\[
M_{*,i}=\sum_k f_{ik}M_{*,k}.
\]

于是自动满足：

\[
\sum_iM_i=\sum_kM_k,
\]

\[
\sum_i\boldsymbol P_i=\sum_k\boldsymbol P_k,
\]

\[
\sum_iE_i=\sum_kE_k.
\]

最后由目标网格真实体积 \(V_i\) 构造强度量：

\[
\rho_i=\frac{M_i}{V_i},
\qquad
\boldsymbol v_i=\frac{\boldsymbol P_i}{M_i},
\qquad
e_i=\frac{E_i}{V_i}.
\]

温度仍由粗粒化后的能量与密度重新计算。

该方法的含义是：

> \(M_{\rm filter}\) 控制小尺度信息被平滑的质量尺度；规则输出网格只负责向 CNN 提供矩阵表示。最终规则像素不要求等质量。

### 6.4 核函数的分阶段选择

axisymmetric Rm-v0 使用易解释、紧支撑的核：

- top-hat；或
- cubic spline。

axisymmetric Rm-v1 固定使用光滑的 Wendland \(C^2\) 核：

\[
W(q)=(1-q)^4(1+4q),\qquad 0\le q<1.
\]

后续再使用与目标 GIZMO `Config.sh` 中 `KERNEL_FUNCTION` 一致的精确核，并从相同版本源码的 `kernel.h` 读取定义。

### 6.5 \(M_{\rm filter}\) 的定义

需要区分：

#### 单元质量尺度

\[
M_{\rm filter}=m_{\rm target}.
\]

它描述单个等质量流体单元的等效空间尺度。

#### MFM 邻域/核尺度

\[
M_{\rm filter}=N_{\rm ngb}m_{\rm target}.
\]

它描述 GIZMO 进行局部梯度和通量重建时使用的邻域质量尺度。

注意：MFM 并没有把 \(N_{\rm ngb}\) 个单元合并成一个单元。因此第二个定义只是“局部数值邻域尺度”，不能被称为 GIZMO 单元质量分辨率。

在没有真实 GIZMO 配置与 snapshot 的第一阶段，建议直接把

\[
M_{\rm filter}
\]

作为算子控制量，不急于将其解释为某个确定的 GIZMO particle mass。

---

## 7. MACER2D 质量尺度的特殊限制

MACER2D 的一个 \((r,\theta)\) cell 代表绕对称轴旋转一周的三维环带，其 `Data-Set-15` 体积和对应质量是完整轴对称环带的体积与质量。

GIZMO 的一个气体单元则是局部三维质量元。因此：

\[
M_k^{\rm MACER2D\ ring}
\]

不能直接等同于

\[
m_a^{\rm GIZMO\ local\ cell}.
\]

第一阶段必须明确采用以下两种层级之一。

### 7.1 v0：轴对称质量滤波实验

直接在 MACER2D 原生网格拓扑和真实环带体积上定义累计质量。这可以用于比较自适应质量滤波与固定块滤波的统计和机器学习差异，但结论只能写成：

> 在 MACER2D 轴对称质量度量下的 \(\mathcal R_m\) 实验。

不能直接声称该 \(M_{\rm filter}\) 等于某个 GIZMO particle mass。

### 7.2 `local_3d_rm_v0`：未来三维局部球形质量滤波

将 MACER2D 解解释为轴对称三维连续场：

\[
\rho(r,\theta,\phi)=\rho(r,\theta),
\]

然后在每个空间点的局部三维球形支持域内求：

\[
\int_{|\boldsymbol y-\boldsymbol x|<H}
\rho(\boldsymbol y)\,{\rm d}^3y
=M_{\rm filter}.
\]

这种定义更接近 GIZMO 的局域三维邻域，但仍无法恢复真实的非轴对称结构。

推荐顺序是先完成轴对称方法，验证完整数据与训练管线；若 \(\mathcal R_m\) 显示出价值，再实现 `local_3d_rm_v0` 或直接转向 MACER3D 数据。本节方法尚未实现，不能与下节的轴对称加速版混用名称。

### 7.3 `axisymmetric_rm_v1`：当前轴对称加速与光滑核版本

本轮已经实现的 v1 仍然使用 MACER2D 整环质量与子午面 \((R,z)\) 距离，并没有变成三维局部球形滤波。它相对 v0 的变化是数值算法而不是质量几何：

1. 对固定网格几何缓存 cKDTree 邻居；
2. 向量化查询初始 KNN，仅对未达到 \(M_{\rm filter}\) 的 source 自适应扩大邻居数；
3. 分离固定质量尺度 \(H_{m,k}\) 与输运图尺度 \(H_{g,k}=\eta_kH_{m,k}\)；
4. 使用 Wendland \(C^2\) 光滑核；
5. 检查 native edge 与体积边缘的显式可行解；
6. 在 coarse 有向图上检查强连通分量，局部扩大相关支撑，并删除不能参与守恒环流的跨 SCC 单向边；
7. 用 64 维对偶 L-BFGS 产生 warm start，再用有限次 IPFP polish 达到边缘容差；
8. 对 KNN 数量、图修复轮数、求解器迭代数和 wall time 设置硬停止条件。

因此统一命名为：

- `axisymmetric_rm_v0`：top-hat/cubic-spline + 普通 IPFP；
- `axisymmetric_rm_v1`：Wendland \(C^2\) + 图审查 + L-BFGS/IPFP；
- `local_3d_rm_v0`：尚未实现的三维局部球形方案。

---

## 8. \(\mathcal R_h\) 与 \(\mathcal R_m\) 的公平比较原则

### 8.1 相同物理区域

两种算子必须作用于相同的 HR snapshot、相同的径向范围和相同边界。

### 8.2 相同输出张量形状

第一轮可让两种算子都输出相同矩阵形状，便于复用相同 CNN，例如：

\[
8\times8.
\]

但 \(8\times8\) 非常粗，可能使目标规则网格本身成为主导滤波器，从而掩盖 \(\mathcal R_m\) 的自适应核差异。因此科学主实验还应增加更细的共同输出网格，例如：

\[
36\times140,
\quad
18\times56,
\quad
12\times28,
\quad
8\times20.
\]

具体形状需结合模型架构和计算成本确定。

### 8.3 相同守恒要求

两种算子都必须验证：

\[
\epsilon_Q
=
\frac{|Q_{\rm coarse}-Q_{\rm HR}|}
{\max(|Q_{\rm HR}|,Q_{\rm floor})}.
\]

至少对以下量检查：

- 总体积（对固定覆盖域）；
- 气体质量；
- 三方向总动量；
- 气体能量；
- 初始恒星质量；
- 若输出局部 SFR 标签，则总 SFR。

### 8.4 匹配有效信息预算

不能把一个很细的 \(\mathcal R_m\) 与一个极粗的 \(\mathcal R_h\) 比较。应至少用以下指标匹配或分层报告：

- 每个输出位置的有效 fine-cell 数量 \(N_{\rm eff}\)；
- 中位有效空间尺度；
- 中位滤波质量；
- 密度场方差保留率；
- 功率谱截止尺度；
- 压缩率和输出自由度。

可定义有效贡献单元数：

\[
N_{{\rm eff},i}
=
\frac{
\left(\sum_k f_{ik}M_k\right)^2
}{
\sum_k(f_{ik}M_k)^2
}.
\]

### 8.5 相同训练协议

两种算子必须使用：

- 相同 snapshot 清单；
- 相同训练/验证/测试 run；
- 相同目标变换；
- 相同 CNN 架构和参数量；
- 相同优化器、训练轮数和随机种子集合；
- 相同输入通道集合；
- 相同超参数搜索预算。

---

## 9. 训练任务的分阶段定义

### 9.1 阶段 A：复现当前 direct CNN

首先仅替换输入粗粒化算子：

\[
\boldsymbol X_h
\longrightarrow
Y_{\rm HR}^{\rm global},
\]

以及

\[
\boldsymbol X_m
\longrightarrow
Y_{\rm HR}^{\rm global}.
\]

该阶段回答：

> 在相同 CNN 与全局 SFR 标签下，质量自适应粗粒化是否比规则固定块保留了更有利于预测 SFR 的信息？

它不直接回答完整闭合项是否更小，因为尚缺真正的 \(B_\alpha\)。

### 9.2 阶段 B：计算算子相关物理基线

对每一种粗粒化状态分别计算：

\[
B_h=\mathcal S_{\rm LR}(\boldsymbol X_h),
\]

\[
B_m=\mathcal S_{\rm LR}(\boldsymbol X_m).
\]

这里的 \(\mathcal S_{\rm LR}\) 应优先为可复现的 MACER SFR 公式。如果冷却时间、引力势梯度或其他输入尚不完整，应：

1. 补充计算/输出所需物理量；或
2. 定义并明确标记一个 MACER-like 近似基线；
3. 不能把当前 KS baseline 当作完整 MACER-LR baseline。

随后得到：

\[
\mathcal C_h=Y_{\rm HR}-B_h,
\qquad
\mathcal C_m=Y_{\rm HR}-B_m.
\]

### 9.3 阶段 C：残差学习

训练相同模型分别学习：

\[
\widehat{\mathcal C}_h
=
F_\theta(\boldsymbol X_h,\boldsymbol q_h,B_h),
\]

\[
\widehat{\mathcal C}_m
=
F_\theta(\boldsymbol X_m,\boldsymbol q_m,B_m).
\]

最终：

\[
\widehat Y_h=B_h+\widehat{\mathcal C}_h,
\]

\[
\widehat Y_m=B_m+\widehat{\mathcal C}_m.
\]

SFR 非负且跨越多个数量级。实际实现可采用对数残差：

\[
c_\alpha^{\log}
=
\log(Y+\epsilon)-\log(B_\alpha+\epsilon).
\]

但必须处理 \(B_\alpha=0\)、\(Y>0\) 的情形，不能使用严格的纯乘法修正使零永远保持为零。

还必须区分两种互不等价的训练定义：

- 若学习物理空间加法闭合 \(C_\alpha^{\rm true}=Y-B_\alpha\)，则重建是
  \[
  \hat Y=B_\alpha+\hat C_\theta,
  \]
  而残差可能为负，不能直接复用只接受非负 SFR 的变换；可用标准化、signed-\(\operatorname{asinh}\) 等变换。
- 若学习上述 log-ratio，则它不是物理可加闭合项，正确重建是
  \[
  \hat Y=(B_\alpha+\epsilon)
  \exp(\hat c_\theta^{\log})-\epsilon,
  \]
  不能再写成 \(B_\alpha+\hat c_\theta^{\log}\)。

### 9.4 阶段 D：场级或 patch 级 SFR 闭合

在确认 `Data-Set-11` 的逐 cell 语义和单位后，构造：

\[
\dot M_{\star,i}^{\rm target}
=
\sum_k f_{ik}\dot M_{\star,k}.
\]

然后训练：

\[
\boldsymbol X_{\alpha,\mathcal N(i)}
\longrightarrow
\dot M_{\star,i}^{\rm target}.
\]

这比全局标量 SFR 更接近未来在线嵌入 GIZMO 所需的局部 source closure，但不应在标签语义未核验前贸然实施。

---

## 10. 多分辨率数据集设计

### 10.1 \(\mathcal R_h\) 分辨率序列

当前只有 \(8\times8\) 主数据。应增加多个规则层级，例如：

\[
36\times140,
\quad
18\times56,
\quad
12\times28,
\quad
8\times20,
\quad
8\times8.
\]

其中 \(8\times8\) 主要作为极端粗粒化压力测试。

### 10.2 \(\mathcal R_m\) 质量序列

因为当前没有真实 GIZMO snapshot，且 MACER2D 环带质量不能直接解释为局域粒子质量，v0 阶段不应武断固定为

\[
10^4-10^6M_\odot.
\]

建议先根据 MACER 数据自身的质量统计定义一组可覆盖范围，例如：

\[
M_{\rm filter}^{(s)}
\in
\{M_1,M_2,M_3,M_4,M_5\},
\]

使其对应不同的：

- 中位有效 fine-cell 数；
- 中位物理支持尺度；
- 密度方差保留率；
- 与 \(\mathcal R_h\) 各级的大致信息预算。

待目标 GIZMO 配置确定后，再把序列校准到实际：

\[
m_{\rm gas},
\qquad
N_{\rm ngb}m_{\rm gas}.
\]

### 10.3 分辨率条件输入

对 \(\mathcal R_h\)，至少保存：

\[
\Delta r_i,
\quad
r_i\Delta\theta_i,
\quad
V_i^{1/3},
\quad
k_r,k_\theta.
\]

对 \(\mathcal R_m\)，至少保存：

\[
M_{\rm filter},
\quad
H_{{\rm eff},i},
\quad
N_{{\rm eff},i},
\quad
M_i,
\quad
V_i^{1/3}.
\]

这些量应作为模型输入或条件元数据，不能假设网络仅凭密度纹理自行识别分辨率。

---

## 11. 评估矩阵

### 11.1 算子层诊断

对每个 snapshot、每个尺度分别记录：

- 守恒误差；
- \(H_{\rm eff}\) 或 coarse-cell 尺度分布；
- \(N_{\rm eff}\) 分布；
- \(\rho,T,u,\boldsymbol v\) 的一维与联合 PDF；
- 冷致密气体质量比例；
- 超过 MACER SFR 阈值的质量比例；
- 密度/温度梯度统计；
- 空间功率谱或结构函数；
- 与 HR 场的重建/滤波差异。

### 11.2 基线闭合误差

对全局 SFR 可使用：

\[
e_{\rm base,log}
=
\left|
\log(B_\alpha+\epsilon)
-
\log(Y+\epsilon)
\right|.
\]

以及归一残差：

\[
e_{\rm base,rel}
=
\frac{|Y-B_\alpha|}{Y+\epsilon}.
\]

比较 \(\mathcal R_h\) 与 \(\mathcal R_m\) 在相同信息预算下的误差分布，而不是只比较平均值。

### 11.3 模型误差

至少报告：

- log-space MAE/RMSE；
- 物理空间 MAE；
- \(R^2\)；
- 高/低 SFR 分位区间误差；
- SFR 时间序列峰值、持续时间和相位误差；
- 累计新生恒星质量误差；
- 模型相对物理基线的 skill score。

可定义：

\[
{\rm Skill}_\alpha
=
1-
\frac{{\rm Error}(\widehat Y_\alpha,Y)}
{{\rm Error}(B_\alpha,Y)}.
\]

### 11.4 泛化实验

必须至少包括：

1. **同 run、未见时间段**；
2. **leave-one-galaxy/run-out**；
3. **leave-one-resolution-out**；
4. **分辨率插值**：训练粗细两端、测试中间尺度；
5. **分辨率外推**：训练较细尺度、测试更粗尺度；
6. **核敏感性**：top-hat 训练或测试与 cubic-spline 对照；
7. **网格相位敏感性**：轻微移动规则输出网格原点或边界；
8. **跨算子测试**：在 \(\mathcal R_h\) 训练、\(\mathcal R_m\) 测试及反向测试，用于诊断模型是否绑定某个粗粒化纹理。

注意：跨算子测试是鲁棒性诊断，不要求模型必然在完全不同的算子上无条件泛化。

---

## 12. 预期效果与竞争性假设

### 12.1 \(\mathcal R_m\) 可能更好的原因

MACER SFR 对冷、致密气体和非线性时间尺度敏感。质量自适应滤波在高密度区具有更小的空间支持尺度，可能比固定空间块更好地保留：

- 密度峰；
- 冷气体团块；
- SFR 阈值以上的气体质量；
- 盘面附近的局部结构；
- 与 MFM 实际解析能力相似的密度依赖尺度。

因此可能出现：

\[
|\mathcal C_m|<|\mathcal C_h|
\]

以及更好的未见分辨率泛化。

### 12.2 \(\mathcal R_h\) 可能更好的原因

\(\mathcal R_h\) 具有固定拓扑、严格嵌套层级和较低的算子噪声。\(\mathcal R_m\) 的 \(H\) 会随密度变化，并可能在密度结构变化时发生非平滑跳变；其结果还依赖核函数、质量预算和边界处理。

因此 \(\mathcal R_h\) 可能：

- 更容易被 CNN 学习；
- 在 MACER 内部具有更稳定的统计分布；
- 对不同 snapshot 的几何对齐更一致；
- 在没有真实 GIZMO 数据校准时反而更少引入人为假设。

### 12.3 不预设胜者

目前不能提前得出“\(\mathcal R_m\) 一定更好”的结论。合理的竞争性假设是：

- **H1**：\(\mathcal R_m\) 在匹配信息预算下减少物理闭合误差，并提高 OOD 性能；
- **H2**：两者闭合误差接近，但 \(\mathcal R_h\) 更稳定、实现更简单；
- **H3**：\(\mathcal R_m\) 在 MACER 内部有效，但因 MACER2D 轴对称质量解释或核选择产生额外 domain gap；
- **H4**：粗粒化算子的影响小于标签不可辨识性、时间历史缺失或跨 run 物理参数差异。

---

## 13. 具体实施流程

### Phase 0：冻结当前 \(\mathcal R_h\) 基准

1. 保存当前 `galaxy_data.py` 的版本/commit；
2. 记录 \(72\times280\rightarrow8\times8\) 的现有训练结果；
3. 记录当前 input channels、target transform、split 和随机种子作为 legacy baseline；正式对照必须改用 time-block/run-block split，不能冻结已确认存在泄漏的随机 split；
4. 保存守恒验证报告；
5. 不覆盖已有 `coarse/` 数据。

### Phase 1：将粗粒化接口参数化

建议保留现有 `rescale()` 行为以维持兼容性，并新增清晰接口，例如：

```python
rescale_spatial(new_shape, ...)
rescale_mass_adaptive(
    new_shape,
    mass_filter,
    kernel,
    geometry_mode,
    boundary_mode,
    ...,
)
```

输出目录和文件元数据必须记录算子类型：

```text
operator = spatial_block | adaptive_mass
output_shape
mass_filter
kernel_name
kernel_support_convention
geometry_mode
boundary_mode
conservation_tolerance
code_version
```

### Phase 2：实现 \(\mathcal R_m\)-v0

1. 从 `density * cell_volume` 得到 fine-cell 质量；
2. 在原生 MACER2D 拓扑上建立邻域搜索；
3. 对每个源单元求满足累计质量约束的 \(H_k\)；
4. 计算到规则输出网格的核积分/近似权重；
5. 对稀疏核执行双边缘平衡，满足 \(\sum_iT_{ik}=V_k\) 与 \(\sum_kT_{ik}=V_i\)，再令 \(f_{ik}=T_{ik}/V_k\)；
6. 沉积质量、动量、能量与恒星质量；
7. 重新计算强度量与温度；
8. 运行与 `rescale()` 同等级的守恒检查；
9. 保存 \(H_{\rm eff}\)、\(N_{\rm eff}\)、实际沉积质量等诊断通道。

### Phase 3：单 snapshot 数值验证

至少测试：

- 常密度、常速度场是否保持常数；
- 单个密度峰是否按预期自适应平滑；
- 总质量、动量和能量是否守恒；
- \(M_{\rm filter}\) 增大时，\(H\) 是否整体单调增大；
- 高密度区的 \(H\) 是否小于低密度区；
- 边界单元是否出现不合理质量堆积；
- top-hat 与 cubic-spline 的差异是否可控；
- 重复运行是否完全可复现。

### Phase 4：生成 paired dataset

对同一个 HR snapshot 同时生成：

```text
HR snapshot ID
├── R_h(scale_1)
├── R_h(scale_2)
├── ...
├── R_m(mass_1)
├── R_m(mass_2)
└── ...
```

所有变体共享同一个 HR 全局 SFR 标签、run ID 和物理时间。

### Phase 5：direct CNN 对照实验

在不改模型任务的前提下，分别训练：

1. 当前 \(\mathcal R_h(8\times8)\)；
2. 多尺度 \(\mathcal R_h\)；
3. 单尺度 \(\mathcal R_m\)；
4. 多尺度/多质量 \(\mathcal R_m\)；
5. 混合 \(\mathcal R_h+\mathcal R_m\) 训练；
6. 明确输入分辨率条件和不输入分辨率条件的消融。

### Phase 6：实现 MACER-LR baseline 与残差模型

1. 核验 MACER SFR 所需输入；
2. 计算或补充 \(\tau_{\rm cool}\)、\(\tau_{\rm ff}\)、\(\tau_{\rm rot}\) 等；
3. 对每种 \(\mathcal R_\alpha\) 计算 \(B_\alpha\)；
4. 比较 \(\mathcal C_h\) 和 \(\mathcal C_m\)；
5. 将训练目标切换为闭合残差；
6. 与 direct CNN、KS-FiLM 和纯物理基线做完整消融。

### Phase 7：获得 GIZMO snapshot 后的校准

读取真实：

\[
\boldsymbol x_a,
m_a,
\rho_a,
H_a,
\boldsymbol v_a,
u_a.
\]

随后：

1. 统计实际质量分辨率分布 \(p(m_a)\)；
2. 统计 \(p(H_a,\rho_a,T_a,r)\)；
3. 校准 \(M_{\rm filter}\)、核函数与邻居数；
4. 验证 GIZMO 到共同网格的守恒和 round-trip 局部误差；
5. 比较 MACER \(\mathcal R_h/\mathcal R_m\) 与 GIZMO 输入的 domain shift；
6. 再决定最终部署采用规则 CNN、局部 patch、图网络还是神经算子。

---

## 14. 建议的数据与代码产物

### 14.1 HDF5 字段

每个 coarse snapshot 至少保存：

- `density`；
- `gas_energy`；
- `temperature`；
- `v1`, `v2`, `v3`；
- `initial_stellar_density`；
- `cell_volume`；
- `gas_mass`；
- 坐标轴及边界；
- SFR target metadata；
- operator metadata。

\(\mathcal R_m\) 额外保存：

- `effective_support_scale`；
- `effective_source_count`；
- `mass_filter`；
- `kernel_id/name`；
- `boundary_affected_fraction`；
- 每种守恒量的误差摘要。

### 14.2 目录建议

避免继续用含义不明确的单一 `coarse/`：

```text
processed/
├── fine/
├── coarse_rh/
│   ├── shape_36x140/
│   ├── shape_18x56/
│   └── shape_8x8/
└── coarse_rm/
    ├── mass_.../
    ├── mass_.../
    └── mass_.../
```

### 14.3 自动测试

新增测试应覆盖：

- 守恒量精度；
- 输出 shape；
- 轴和体积的一致性；
- 空字段、负密度和非有限值拒绝；
- 大/小 \(M_{\rm filter}\) 极限；
- 边界核归一化；
- 相同输入的确定性；
- `Data-Set-11` 不进入模型输入，防止标签泄漏。

---

## 15. 预期结论的表述边界

### 当前可以确认

1. 已有 \(\mathcal R_h\) 是基于真实体积的守恒固定分块，而不是普通 resize；
2. 已完成 \(72\times280\rightarrow8\times8\) 数据生成和 CNN 训练；
3. 当前目标是 snapshot 级全局 SFR；
4. 当前 KS baseline 和 CNN-FiLM 不等于 MACER-LR 残差闭合模型；
5. 当前没有真实 GIZMO snapshot，因此尚不能验证跨代码泛化。

### 第一阶段实验可以回答

1. 在 MACER2D 内部，\(\mathcal R_m\) 是否比 \(\mathcal R_h\) 保留更多与全局 SFR 有关的信息；
2. 哪一种算子下 direct CNN 的未见时间/run/分辨率性能更好；
3. \(\mathcal R_m\) 是否对核和质量预算过度敏感；
4. 在实现 MACER-LR baseline 后，哪一种算子的闭合误差更小、更可学习。

### 第一阶段不能回答

1. \(\mathcal R_m\) 是否精确复制真实 GIZMO/MFM 数值信息损失；
2. MACER2D 环带质量是否能直接对应 GIZMO particle mass；
3. 模型是否能泛化到真实 cosmological zoom-in；
4. 模型在线替代 GIZMO SFR 后是否稳定；
5. filtered-HR 的成功是否能代表真实 LR 重新运行的成功。

这些问题必须等待实际 GIZMO 配置、snapshot 和最终在线实验。

---

## 16. Go/No-Go 判据

只有满足以下条件，才认为 \(\mathcal R_m\) 相比 \(\mathcal R_h\) 具有继续投入的价值：

1. 在匹配信息预算下，物理基线闭合误差显著降低；或
2. 即使闭合项幅度相近，残差模型在未见 run/分辨率上稳定更好；
3. 提升在多个随机种子和多个 MACER run 上存在；
4. 提升不依赖一个极窄的 \(M_{\rm filter}\) 或单一核函数；
5. 守恒误差满足预设容差；
6. 算子复杂度和数据处理成本可接受；
7. 获得 GIZMO snapshot 后，\(\mathcal R_m\) 确实减小 MACER-to-GIZMO 输入分布差距。

如果 \(\mathcal R_m\) 只在训练分辨率上略好，但在未见分辨率、未见 run 或核扰动下失效，应优先保留更简单稳定的 \(\mathcal R_h\)。

---

## 17. 推荐的近期最小可行任务

近期不应同时修改粗粒化算子、网络架构、损失函数和数据划分。最小任务应为：

1. 冻结现有 \(8\times8\) \(\mathcal R_h\) CNN 结果；
2. 实现轴对称 \(\mathcal R_m\)-v0 的守恒沉积；
3. 让 \(\mathcal R_m\) 同样输出 \(8\times8\)，以直接复用当前 CNN；
4. 选择 3-5 个 \(M_{\rm filter}\)；
5. 在完全相同的数据划分和训练配置下重训；
6. 比较 direct CNN 的 in-domain、leave-one-run-out 和 leave-one-mass-out 性能；
7. 再在更细共同输出网格上复现实验，确认 \(8\times8\) 没有掩盖自适应滤波差异；
8. 同步推进 MACER-LR SFR baseline 的精确复现；
9. 最后才进入真正的闭合残差训练。

---

## 18. 一句话项目定义

> 本阶段不是预设质量粗粒化必然优于规则空间粗粒化，而是在 MACER HR 数据上构建两个守恒、分辨率可控的粗粒化算子 \(\mathcal R_h\) 与 \(\mathcal R_m\)，在匹配信息预算下比较它们造成的 SFR 非交换闭合误差、残差可学习性和 OOD 泛化，从而为未来 GIZMO/MFM 部署选择更合适的数据表示与闭合模型。

---

## 19. 轴对称 \(\mathcal R_m\)-v0 的当前实现

### 19.1 v0 的精确定义

当前数值实现位于 `raw_data_processing/coarsening_rm_v0.py`；
`raw_data_processing/galaxy_data.py` 中的 `GalData.rescale_mass_adaptive()` 只保留为向后兼容的薄接口。它不是“不规则等质量块再做几何相交”的实现，而是一个更适合当前规则 CNN 输出的两阶段算子：

1. 对每个 MACER fine cell \(k\)，在子午面坐标
   \[
   (R_k,z_k)=(r_k\sin\theta_k,r_k\cos\theta_k)
   \]
   中寻找最小支撑尺度 \(H_k\)，使邻域内的轴对称环带质量满足
   \[
   \sum_{l:d_{kl}\le H_k}m_l\ge M_{\rm filter}.
   \]
2. 根据 \(H_k\) 和核函数构造 fine source 到规则 output cell 的稀疏输运矩阵；
3. 用稀疏迭代比例拟合同时约束输运矩阵的行、列边缘：
   \[
   \sum_I T_{Ik}=V_k,
   \qquad
   \sum_kT_{Ik}=V_I.
   \]
4. 对质量、动量、由 `gas_energy*volume` 得到的气体内能、恒星质量等广延量做守恒沉积，再由沉积结果导出密度、内能密度、速度和温度。

上述双边缘约束的目的不仅是保证总质量等广延量守恒，还保证常数密度场不会由于自适应核和规则输出网格的组合而产生人为起伏。

这里的 \(M_{\rm filter}\) 是 **MACER2D 轴对称整环质量预算**。它控制每个 source cell 看见的邻域质量和有效空间尺度；它不是每个输出规则格的质量，也不能在尚无真实 snapshot 对照时直接解释为 GIZMO particle mass。

### 19.2 已实现文件

- `raw_data_processing/coarsening_types.py`：所有算子共用的纯数组输入/输出契约；
- `raw_data_processing/coarsening_common.py`：几何、规则 block sum、字段验证与守恒验证；
- `raw_data_processing/coarsening_spatial.py`：\(\mathcal R_h\)；
- `raw_data_processing/coarsening_rm_v0.py`：axisymmetric \(\mathcal R_m\)-v0；
- `raw_data_processing/coarsening_rm_v1.py`：axisymmetric \(\mathcal R_m\)-v1；
- `raw_data_processing/galaxy_data.py`：HDF I/O、对象组装、统一 `GalData.coarsen()` 分派及旧 API 包装；
- `raw_data_processing/build_rm_v0.py`：单文件或目录批处理命令；
- `raw_data_processing/estimate_rm_mass_filter.py`：由现有 \(\mathcal R_h\) block mass 估计探索用质量尺度；
- `raw_data_processing/audit_mass_coarsening.py`：抽样、超时保护与批量物理/数值审计；
- `raw_data_processing/visualize_coarsening_partitions.py`：软输运和硬化伪分区可视化；
- `raw_data_processing/compare_coarsening_audits.py`：v0/v1 配对比较；
- `tests/`：空间算子、v0、v1、审计、可视化和比较器的自动测试。

输出 HDF5 额外保存：

- `source_support_scale`：每个 fine source 的 \(H_k\)；
- `source_enclosed_mass`：确定 \(H_k\) 时实际包围的质量；
- `effective_support_scale`：每个规则输出格接收到的质量加权有效尺度；
- `effective_source_count`：每个规则输出格的有效 source 数；
- `dominant_destination` 与 `dominant_fraction`：软输运的 argmax 诊断及其权重；
- `source_transport_entropy`：每个 source 的输运熵；
- `transport_mass_by_destination_native`：coarse destination 与 source-native coarse cell 之间的质量流；
- 输运矩阵平衡误差与迭代次数。

### 19.3 选择第一组 \(M_{\rm filter}\)

为了让 \(\mathcal R_m\) 与已有 \(72\times280\to8\times8\) 的 \(\mathcal R_h\) 比较尽量公平，先计算每个规则 block 的真实气体质量：

\[
M_C^{(h)}=\sum_{k\in C}\rho_kV_k,
\qquad
M_0=\operatorname{median}_{C,t,\mathrm{run}}M_C^{(h)}.
\]

推荐第一轮使用

\[
M_{\rm filter}\in
\{0.25,0.5,1,2,4\}M_0.
\]

这只是匹配信息预算的实验定义。正式比较时还要同时报告有效 source 数和有效支撑尺度的分布，因为相同质量阈值在不同密度区域对应不同空间尺度。

估计命令：

```powershell
python raw_data_processing/estimate_rm_mass_filter.py `
  --input data/disk_galaxy_fiducial/fine `
  --shape 8 8 --limit 100
```

生成一个质量尺度的数据集：

```powershell
python raw_data_processing/build_rm_v0.py `
  --input data/disk_galaxy_fiducial/fine `
  --output data/disk_galaxy_fiducial/coarse_rm_v0_m0 `
  --shape 8 8 --mass-filter <上一步得到的M0> `
  --kernel top_hat --limit 1
```

应先用 `--limit 1` 做 smoke test，检查守恒误差和支撑尺度分布，再批量处理。

### 19.4 当前数据迁移与运行前提

仓库中现有 `data/.../fine/*.h5` 是旧的 processed 文件，未保存
`geometry/cell_volume` 和坐标边界，不能可靠构造守恒 \(\mathcal R_m\)。需要用当前
`galaxy_data.py` 从原始 HDFRA 重新生成 fine HDF5，使真实 `Data-Set-15` 体积和轴边界进入 processed 文件。

base 环境仍存在 SciPy 二进制不一致，因而另建了隔离环境 `macer_preprocess`：

- Python 3.12.13；
- NumPy 2.5.2；
- SciPy 1.18.0；
- `pyhdf.SD` 和 `scipy.spatial.cKDTree` 均已验证可用。

安全重建单个 snapshot 的入口为 `raw_data_processing/rebuild_fine_snapshot.py`。它默认拒绝覆盖已有文件：

```powershell
D:\conda_envs\macer_preprocess\python.exe `
  raw_data_processing/rebuild_fine_snapshot.py `
  --galaxy disk_galaxy_fiducial `
  --snapshot-index 0 `
  --output data/rm_v0_smoke/fine/00000.h5
```

2026-08-14 已完成 `hdfra.00000` 的隔离重建和验证：

- shape：\((72,280)\)；
- 新 HDF5 format version：4；
- `Data-Set-15` 与原始 HDF4 最大绝对差：0；
- 七个模型输入场与旧 fine HDF5 最大绝对差：0；
- 径向和角向坐标边界均严格递增；
- HDF5 round-trip 通过。

该样本对应的 \(8\times8\) 规则 block mass 中位数为

\[
M_0=1.2877608
\]

（代码质量单位）。使用 top-hat 核的真实 \(\mathcal R_m\)-v0 在默认 5000 次平衡迭代后尚未达到 \(10^{-6}\) 容差；提高到 30000 次上限后在 8700 次收敛，误差为 \(9.966\times10^{-7}\)。因此批处理命令应暂时显式加入：

```powershell
--balance-max-iterations 30000
```

该 smoke test 的总体积、质量、能量和三方向动量相对守恒误差均小于 \(5\times10^{-16}\)。后续仍需在多个 snapshot 和多个 \(M_{\rm filter}\) 上统计迭代数，判断是否需要改善稀疏核连通性，而不是永久依赖更高迭代上限。

### 19.5 v0 的科学边界

轴对称 v0 可以检验“质量自适应尺度是否比固定空间 block 更适合 MACER 内部闭合学习”，但不能单独证明：

- 它等价于 GIZMO/MFM 的邻域重构；
- MACER 整环质量等价于 GIZMO 三维粒子质量；
- filtered HR 等价于同一初始条件下真正重跑的 LR 解；
- 在 MACER 数据上更好的模型必然能迁移到宇宙学 zoom-in。

获得 GIZMO snapshot 后，应另外实现 particle-to-common-grid 沉积，并比较输入分布、有效尺度以及守恒量；不能把当前 v0 的成功直接当作跨代码迁移结论。

---

## 20. 2026-08-16 模块化重构与接口契约

### 20.1 当前代码结构

粗粒化数值算法已经从臃肿的 `galaxy_data.py` 中分离。现在的数据流为：

\[
\texttt{GalData/HDF}
\longrightarrow
\texttt{CoarseningInput}
\longrightarrow
\mathcal R_h\ \text{或}\ \mathcal R_m
\longrightarrow
\texttt{CoarseningResult}
\longrightarrow
\texttt{GalData/HDF}.
\]

所有算子必须返回：粗化后的物理场、真实输出体积、坐标中心与边界、算子 metadata、diagnostics 和分阶段运行时间。`GalData.coarsen()` 是统一入口；原有 `rescale()` 与 `rescale_mass_adaptive()` 仍可调用，分别包装 \(\mathcal R_h\) 和 axisymmetric Rm-v0。

规则空间算子的模块化版本与旧实现做了逐字段回归：物理场、坐标和 metadata 最大差均为 0。axisymmetric Rm-v0 在真实 \(72\times280\) smoke snapshot 上与拆分前实现逐字段及旧诊断量最大差也为 0；仍为 8700 次 IPFP、平衡误差 \(9.966\times10^{-7}\)。

预处理主入口仍是：

```powershell
python raw_data_processing/galaxy_data.py `
  --coarsener spatial_block|axisymmetric_rm_v0|axisymmetric_rm_v1 `
  --coarse-shape 8 8
```

Rm 算子还需显式给出 `--mass-filter`。不同算子默认进入不同 coarse 子目录，`GalDataSet.load_data()` 现在按文件名确定性排序，并拒绝静默混合不同 operator、kernel 或 \(M_{\rm filter}\)。

### 20.2 本轮修复的配置问题

`disk_galaxy.supplement` 的 concatenate 配置原来把 `ranges` 写成单一 `[start,end]`，而调度器需要 `[[start,end], ...]`；本地 `.config/RawDataConfig.json` 已修正。该目录被 `.gitignore` 忽略，迁移环境时仍需检查本地配置是否同步。

### 20.3 守恒声明的边界

当前算法严格审计：

- cell volume；
- gas mass；
- 由内能密度积分得到的气体内能 `gas_energy*volume`；
- 数值上的 \(\sum m v_r,\sum m v_\theta,\sum m v_\phi\)；
- 初始恒星质量。

但球坐标基矢随位置变化，后三个数值分量的总和不等于全局 Cartesian 动量。因此审计器额外记录

\[
P_z=\sum m(v_r\cos\theta-v_\theta\sin\theta),
\qquad
L_z=\sum mr\sin\theta\,v_\phi,
\]

但没有把它们误称为当前投影自动保证的守恒量。质量加权速度平均还会消除 resolved kinetic energy；审计器把该动能损失单列为 SGS/未解析动能诊断，不能声称物理总能量完整守恒。

---

## 21. 30-snapshot 分层审计设计

### 21.1 样本与重建

现有约 2.4 万个 processed fine HDF5 是旧格式，缺少 authoritative `Data-Set-15` 和坐标边界，不能用于守恒审计。本轮从三个 disk-galaxy run 各选 10 个 snapshot：

- 每个 run 划分五个时间区间；
- 每个区间覆盖高 SFR 与气体形态对照状态；
- 辅助特征包括总气体质量、密度分位数、冷气体比例、速度 RMS 和子午面形态；
- SFR 覆盖 0 到约 \(3.1\times10^3\,M_\odot\,{\rm yr}^{-1}\)。

30 个样本均由原始 HDF4 重新生成 format-v4 fine HDF5，并验证 round-trip、体积来源和 shape。清单与产物位于：

- `Results/coarsening_audit/selected_snapshots.csv`；
- `Results/coarsening_audit/rebuilt_manifest.csv`；
- `data/coarsening_audit/fine/<run>/`。

这些目录属于本地实验产物，受 `.gitignore` 管理。

### 21.2 审计器与超时

`audit_mass_coarsening.py` 对每个 snapshot/operator 使用独立 spawn 进程，并在每项结束后立即 checkpoint CSV。它记录：

\[
\epsilon_{\rm conservation},\quad
\epsilon_{\rm balance},\quad
N_{\rm iteration},\quad
H_{\rm eff},\quad
N_{\rm eff},\quad
t_{\rm runtime}.
\]

同时严格分开两组质量：

1. `source_enclosed_mass/Mfilter`：重叠支撑域是否达到质量阈值及离散超调；
2. `output_cell_mass/Mfilter`：规则 CNN 输出格质量的描述统计，不是“目标质量误差”。

审计还逐格保存 64 个输出质量、\(H_{\rm eff}\)、\(N_{\rm eff}\) 与 argmax 伪分区连通性。

对抗审查发现并修复了一个 Windows 多进程死锁：旧版本先 `join()` 子进程，再从 Queue 读取包含 64 个 cell 的结果；当结果超过管道缓冲时，子进程阻塞在 `put()`，所有成功任务都会被误判为超时。现在父进程先取结果，再回收子进程；真实单项回归通过。

---

## 22. axisymmetric Rm-v0 的 30-snapshot 结果

固定设置为

\[
72\times280\rightarrow8\times8,
\qquad
M_{\rm filter}=M_0=1.2877608,
\]

top-hat 核、平衡容差 \(10^{-6}\)、最多 30000 次 IPFP、每 snapshot 90 秒硬超时。

### 22.1 运行状态

| 状态 | 数量 |
|---|---:|
| 成功 | 23 |
| 30000 次后未收敛 | 6 |
| 90 秒超时 | 1 |

六个未收敛样本的终止平衡误差约为 \(1.8\times10^{-6}\)–\(7.8\times10^{-6}\)。成功样本中：

- wall time 中位数 21.08 s，5%–95% 为 11.04–42.11 s；
- IPFP 迭代中位数 4380，最大 23850；
- 最大 stored-extensive 守恒误差中位数 \(4.85\times10^{-16}\)，最大 \(1.39\times10^{-15}\)；
- 平衡误差均小于 \(10^{-6}\)。

因此当前 v0 的主要问题不是广延量沉积守恒，而是 top-hat 稀疏图上普通 IPFP 的 snapshot-dependent 收敛性。

### 22.2 质量统计

对成功样本，`source_enclosed_mass/Mfilter` 的 snapshot 中位数再取中位为 1.0145；其 p95 的跨样本中位数为 1.726。但单个巨大环带 cell 会产生不可避免的离散超调：每个 snapshot 最大值的中位数为 93.0，30 样本成功子集中的全局最大值为 242。这不是算法违反质量守恒，而是当一个原生整环 cell 本身大于质量预算时，不可能拆分该 source cell。

规则输出格质量完全不受等质量约束。成功样本中：

- `output_cell_mass/Mfilter` 的 snapshot 最大值中位数约 985；
- 质量 CV 中位数约 2.59；
- Gini 中位数约 0.857。

所以固定质量粗粒化的意义在于 **source 的自适应信息支撑尺度**，不是最终 \(8\times8\) CNN 像素等质量。

### 22.3 伪分区与可视化

真实 smoke 图保存在 `Results/coarsening_audit/smoke_v0_partition.png`，报告为同名 JSON。该样本的 argmax 后验硬化结果有 64 个 connected components、56 个非空 destination、5 个 destination 被分裂；coarse 有向输运图有 5 个 SCC。

全体成功 v0 样本的 snapshot 中位数为：

- argmax component 总数 57；
- 断裂 destination 数 3；
- 空 destination 数 11。

这些只说明 soft transport 的主导权重形态；四色着色只是显示工具，不能作为真实物理分区存在或连通的证据。

---

## 23. axisymmetric Rm-v1 的实现与 30-snapshot 对照

### 23.1 邻居上限的预审计修正

默认 `max_neighbors=4096` 在部分低密度 source 上不足以完整包围固定 \(M_{\rm filter}\)，程序按设计快速失败。代表样本测试表明，允许 **仅对未解决 source** 自适应扩展到全域上限 20160 后，实际最多查询 8192 个邻居，24.7 s 完成；质量定义并未改变。

因此正式 v1 审计使用：

- `max_neighbors=20160`；
- `max_neighbor_expansions=10`；
- 300 次 L-BFGS；
- 最多 500 次 IPFP polish；
- 80 s 内部 deadline 与 90 s 外部超时。

### 23.2 状态与守恒

| 状态 | v0 | v1 |
|---|---:|---:|
| 成功 | 23 | 25 |
| 未收敛 | 6 | 4 |
| 超时 | 1 | 1 |

v1 成功样本的 stored-extensive 守恒误差中位数为 \(5.46\times10^{-16}\)，最大 \(1.86\times10^{-15}\)。25 个成功样本的最终 coarse 图均为 1 个 SCC，且真实数据上没有触发跨 SCC 边裁剪；裁剪逻辑的正确性由单向边对抗测试覆盖，但尚未被这 30 个成功 snapshot 实际触发。

### 23.3 21 个双方成功样本的严格配对结果

只有相同 `sample_uid` 且 v0/v1 都成功的 21 个样本进入数值配对：

| 指标 | v0 中位数 | v1 中位数 | \(v1-v0\) 中位数 |
|---|---:|---:|---:|
| wall time [s] | 18.86 | 14.37 | -4.92 |
| balance iterations | 4370 | 450 | -3910 |
| balance error | \(9.91\times10^{-7}\) | \(8.97\times10^{-7}\) | \(-8.43\times10^{-8}\) |
| stored conservation error | \(4.82\times10^{-16}\) | \(6.95\times10^{-16}\) | \(2.07\times10^{-17}\) |
| output mass CV | 2.58603 | 2.58607 | 0.000123 |
| argmax 断裂 destination | 3 | 0 | -3 |
| argmax 空 destination | 11 | 2 | -7 |
| resolved kinetic loss fraction | 0.0928 | 0.0924 | -0.00069 |

中位 speedup

\[
\operatorname{median}(t_{v0}/t_{v1})=1.25,
\]

但 5%–95% 为 0.54–2.14，说明 v1 并非每个状态都更快。两者 `source_enclosed_mass/Mfilter` 的配对统计完全相同，因为 v1 没有改变固定质量核心尺度 \(H_m\)，只改变了 \(H_g\)、核形状与平衡器。规则输出质量 CV/Gini 也几乎不变。

失败状态必须单独保留：共同失败/超时 3 个，v1 成功而 v0 失败 4 个，v0 成功而 v1 失败 2 个。不能只报告 21 个成功配对而隐藏求解鲁棒性。

完整结果位于：

- `Results/coarsening_audit/v0_m0_30/`；
- `Results/coarsening_audit/v1_m0_30/`；
- `Results/coarsening_audit/comparison_v0_v1_m0_30/comparison_report.md`。

### 23.4 当前算法结论

在这一质量尺度与本地 CPU 上，v1 相对 v0 的证据是：

- 成功率略高；
- 中位运行时间更短，但不是一致加速；
- 平衡迭代大幅减少；
- 广延量守恒同样可靠；
- argmax 后验形态更少断裂、更少空 destination。

但 v1 **没有**证明：

- \(\mathcal R_m\) 比 \(\mathcal R_h\) 的 SFR 闭合更好；
- 光滑核产生了更物理的真实分区；
- 当前整环质量可直接对应 GIZMO particle mass；
- 对 \(M_0\) 的结果可泛化到其他质量尺度。

因此建议把 `axisymmetric_rm_v1` 作为下一轮 \(\mathcal R_m\) 数据生成的优先候选，同时保留 v0 作为可解释基准；最终 Go/No-Go 仍必须由 \(\mathcal R_h\) 对照、SFR baseline/closure 和 OOD 训练结果决定。

---

## 24. 计算硬件与训练管线的必要修正

### 24.1 四张 A100 对当前预处理的帮助有限

当前实现的主要耗时来自 CPU `cKDTree` 查询、Python/NumPy 稀疏核构造、SciPy sparse 运算和低维平衡器；代码没有 CUDA 路径。直接把同一代码放到 4\(\times\)A100 服务器不会自动获得显著 GPU 加速。更直接的提速顺序是：

1. 复用同一进程内的几何 KNN cache；
2. 多 snapshot 做 CPU 多进程并行，但控制内存与 I/O；
3. 预计算几何邻居并保存 geometry hash；
4. 只有在 profile 证明核构造/平衡是主要瓶颈后，才考虑 CuPy/PyTorch sparse 或批量 GPU 重写。

A100 对后续 CNN/神经算子训练会有明显价值，但这与当前 CPU 预处理是否提速是两个问题。

### 24.2 当前 Experiment 代码的高优先级风险

Step 0 审查时，旧训练是

\[
\widehat z_{\rm SFR}=F_\theta(X,z_{\rm KS}),
\]

其中 KS 只是 FiLM conditioner；它不是 \(B_\alpha+\hat C_\theta\) 的 residual closure。本轮 metadata 修正后，新 version-3 cache 的第一阶段模型契约改为

\[
\widehat z_{\rm SFR}
=F_\theta\!\left(X_{\rm LR},\log_{10}M_{\rm filter}\right).
\]

KS 只保存为评估 baseline，不再作为模型条件；\(H_{\rm eff}\) 和 \(N_{\rm eff}\) 保存为预留字段但尚未输入模型。当前状态为：

1. `mirror` 默认值和所有正式调用均已改为 false；旧 \(8\times16\) cache 仍是 legacy，不能因为代码已修正就继续使用；
2. `SFRSpatialAugmentor` 默认概率已改为 0；当前不物理的实现尚未重写，正式训练继续保持禁用；
3. 同一 HR snapshot 的所有 Rm/Rh/多分辨率变体已可按 `parent_hr_uid` 强制进入同一 split；
4. 随机 parent-snapshot split 仍未解决相邻时间泄漏，正式生成新 cache 前仍需确定并实现 time-block/run-block 方案；
5. HDF5 已传播 operator、\(M_{\rm filter}\)、kernel、\(H_{\rm eff}\)、\(N_{\rm eff}\)、源身份和逐量守恒诊断；PT version-3 写入/读取代码已实现，但本轮按计划没有生成永久 PT cache；
6. 当前仍是 direct SFR 模型，不是 residual closure；未来物理加法残差可为负，不能直接复用只面向非负 SFR 的 `transform_sfr()`。

### 24.3 下一步执行顺序

1. 在生成正式 PT cache 前，先确定 time-block/run-block 的具体划分尺度；metadata、mirror 和增强默认值已经修复；
2. 在同一 30-snapshot 审计集上增加多个 \(M_{\rm filter}\)，确认 v1 的成功率、KNN 上限与核敏感性；
3. 用相同 HR snapshot 同时生成 \(\mathcal R_h\)、Rm-v0、Rm-v1，匹配输出 shape 和信息预算；
4. 先做 direct CNN 的严格 time-block/run-block 对照；
5. 补齐 MACER-LR SFR 算子 \(\mathcal S_\alpha\)，再构造 \(B_\alpha\)、\(C_\alpha^{\rm true}\) 与 residual closure；
6. 做 leave-one-run-out、leave-one-mass-out、leave-one-operator-out 和未见时间段评估；
7. 获得真实 GIZMO snapshot 后，再验证 particle-to-grid 分布差距与跨代码迁移；
8. 最后才进入在线嵌入和稳定性实验。

本轮数值审计的结论是：**axisymmetric Rm-v1 已经成为比 v0 更可用的候选预处理器，但尚没有任何结果支持“Rm 或 v1 的 SFR 预测一定优于 Rh”。**

---

## 25. 2026-08-18：metadata 契约、UID 与第一阶段训练接口

### 25.1 metadata schema v1

新增 `raw_data_processing/metadata_schema.py`，统一负责规范 JSON、文件 SHA-256、几何指纹、UID 构造和完整性验证。新 processed HDF5 格式版本为 5，并新增 `/metadata/contract_json`。metadata 分为：

- `identity`：`parent_hr_uid` 与 `sample_uid`；
- `source`：simulation/galaxy/run/series、原始 snapshot index、源文件 ID、SHA-256、文件大小、物理时间；
- `geometry`：坐标系、坐标通道顺序、数组空间轴顺序、矢量基、volume source 和几何指纹；
- `fields`：通道的 intensive/extensive/vector 语义；
- `target`：Data-Set-11、reduction、quantity 和单位；
- `derivation`：operator ID/version 和所有会改变结果的 operator parameters。

HR 母身份定义为

\[
\texttt{parent\_hr\_uid}
=H(\texttt{simulation\_code},\texttt{galaxy\_id},\texttt{run\_id},
\texttt{source\_series\_id},\texttt{raw\_snapshot\_index}),
\]

不包含本机绝对路径。具体衍生样本定义为

\[
\texttt{sample\_uid}
=H(\texttt{parent\_hr\_uid},\texttt{operator},\texttt{operator\_version},
\texttt{operator\_parameters},\texttt{geometry\_fingerprint}).
\]

因此同一 HR snapshot 的 Rm/Rh/不同质量尺度共享 `parent_hr_uid`，但 `sample_uid` 必须不同。加载新文件时会重新计算两个 UID，拒绝被篡改或内部不一致的 metadata。

### 25.2 算子与守恒诊断传播

`spatial_block`、Rm-v0、Rm-v1 现在都返回结构化 `operator_parameters`。守恒验证不再只在失败时抛异常，也返回并写入 HDF5：

- 每个广延量的 fine/coarse total；
- absolute error；
- scaled relative error；
- tolerance；
- 跨所有已声明广延量的 `epsilon_conservation`；
- `conservation_passed`。

Rm 原有的平衡误差、迭代数、\(H_{\rm eff}\)、\(N_{\rm eff}\) 和输运诊断继续保存。`GalDataSet` 新增逐样本 UID、operator、\(M_{\rm filter}\)、\(H_{\rm eff}\)、\(N_{\rm eff}\) 字段，并提供显式 `allow_mixed_coarsening`；默认仍拒绝静默混合。按 `parent_hr_uid` 分组的 split 已实现并由测试覆盖。

### 25.3 第一阶段模型输入契约

新 PT cache schema version 3 已完成代码实现，但本轮没有生成永久 cache，也没有训练。第一阶段只允许：

\[
\boldsymbol X_{\rm LR},\qquad \log_{10}M_{\rm filter}
\]

进入模型。`SFRConvNet` 通过 FiLM 使用 `log_mass_filter`，version-3 cache 若未启用 FiLM 会明确报错。KS baseline 被单独保存且 `baseline_is_model_input=false`。完整二维 `h_eff` 和 `n_eff` 一同保存在 tensor payload 中，但列入 `reserved_resolution_features`，不会被 `Dataset.__getitem__()` 返回给模型；是否启用必须由后续消融实验决定。

该接口当前只适用于全部样本都有正有限 \(M_{\rm filter}\) 的 Rm 数据集。Rh 没有自然的 \(M_{\rm filter}\)，不能用 NaN 伪装后直接混入这一版模型；未来比较 Rm/Rh 时需先定义共同的物理分辨率描述。

### 25.4 三份真实 fine snapshot 的小批验证

本机当前 Python 环境缺少 `pyhdf`，因此没有声称重新读取原始 HDF4。本轮使用上一轮已经从 HDF4 重建、含 authoritative Data-Set-15 和坐标边界的 format-v4 fine HDF5，通过 `upgrade_fine_metadata.py` 生成 format-v5 metadata 升级副本。物理数组没有改变。产物位于：

```text
data/metadata_smoke/fine/<run>/00000.h5
data/metadata_smoke/coarse_rh/<run>/00000.h5
data/metadata_smoke/coarse_rm_v0/disk_galaxy_fiducial/00000.h5
```

三个 run 的 UID 与 Rh 守恒结果为：

| run | `parent_hr_uid` | fine `sample_uid` | Rh `sample_uid` | Rh `epsilon_conservation` |
|---|---|---|---|---:|
| disk_galaxy_fiducial | `macerhr:7aa673463572b2a0083b` | `macnetsample:1aeb5af7994380e77e77` | `macnetsample:bd03a70c8b67f7353e72` | \(2.423\times10^{-16}\) |
| disk_galaxy_fiducial_4 | `macerhr:f6f6998bfecd461a916e` | `macnetsample:f34ead0b387cbbfffd17` | `macnetsample:cd86b8d78cb8c5c7c3f2` | \(2.048\times10^{-16}\) |
| disk_galaxy_fiducial_7 | `macerhr:74dffe3879ae7740157c` | `macnetsample:2953ce1779fa05c1fe01` | `macnetsample:b630611f380d7d4e2f3b` | \(2.898\times10^{-16}\) |

每个 Rh 样本与对应 fine 的 `parent_hr_uid` 相同，`sample_uid` 不同。额外对 fiducial 做了一份 Rm-v0：

\[
M_{\rm filter}=1.2877608,
\quad
\epsilon_{\rm conservation}=3.615\times10^{-16},
\quad
\epsilon_{\rm balance}=9.993\times10^{-7},
\quad
N_{\rm iteration}=10280.
\]

其 `sample_uid=macnetsample:2c912e15fdfe87866de9`，并与 fine 共享同一 `parent_hr_uid`。当前本机 `scipy.spatial.cKDTree` 不可用，该 Rm-v0 smoke 使用了慢速 NumPy 全距离回退；这不影响 UID/守恒结论，但不能据此评价正式运行速度。

### 25.5 测试与本轮明确未完成事项

新增的对抗测试覆盖：路径移动不改变 UID、改变 \(M_{\rm filter}\) 改变 `sample_uid`、metadata 篡改被拒绝、HDF5 round-trip、逐量守恒诊断、parent-UID grouped split、version-3 cache 字段隔离，以及 `log_mass_filter` 确实影响 FiLM 输出。当前测试结果为 44 项全部通过。

本轮按要求没有：

- 生成新的永久 PT cache；
- 启动任何训练；
- 把 \(H_{\rm eff}\) 或 \(N_{\rm eff}\) 输入模型；
- 实现 residual closure；
- 决定最终 time-block/run-block 划分尺度；
- 声称三份 smoke 可以代替完整 30-snapshot 或多质量尺度审计。

---

## 26. 2026-08-21：MACER SFR 物理基线、signed closure 与训练接口已实现

### 26.1 两套 SFR 算子

新增纯 NumPy 模块 `raw_data_processing/macer_sfr_operator.py`。两套算子共同采用

\[
\dot\rho_\star=c_\star\rho\,f_{\rm step}(\Delta t,\tau_{\rm SF}),
\qquad
\tau_{\rm SF}=\max[\tau_{\rm cool},\min(\tau_{\rm ff},\tau_{\rm rot})],
\]

\[
\tau_{\rm ff}=\sqrt{\frac{3\pi}{32G\rho}},
\qquad
T=93e/\rho<4\times10^4\ {\rm K},
\qquad
1.02\rho>1.
\]

其中 `f_step` 复现 `star.F::exp_factor_sf`；如果离线数据没有内部流体时间步，
`dt=0` 明确定义为连续极限 (1/\tau_{\rm SF})，不能把 snapshot 间隔冒充流体时间步。

默认数据兼容模式为

\[
G=112.3,\quad c_\star=0.01,\quad
\tau_{\rm rot}=2\pi\sqrt{\frac{r^3}{G M_{\rm code}(<r)}},
\quad M_{\rm code}(<r)\simeq M_{\rm BH}.
\]

论文物理对照模式为

\[
G=112.3,\quad c_\star=0.1,\quad
\tau_{\rm rot}=2\pi\sqrt{\frac{r}{|a_r|}}.
\]

默认模式中的 (M(<r)\simeq M_{\rm BH}) 仍是待获取实际运行配置后复核的近似，metadata
以 `mass_enclosed_approximation` 和 `parameter_provenance` 显式记录，不能将其写成已确认的普适 MACER 公式。

### 26.2 Data-Set 语义和单位

源码核对后的语义为：

- Data-Set-5 是源码变量 `d`，即气体质量密度；`n=1.02d` 只用于恒星形成阈值；
- Data-Set-10 是气体内能密度 (e)，并始终满足 (T=93e/\rho)；
- Data-Set-11 是每格 `cell_volume*rhodot_save`，总 SFR 应直接逐格求和；
- Data-Set-15 是权威网格体积；
- Data-Set-17 是冷却时标；Data-Set-8/9 是球坐标 (r/\theta) 方向加速度。

Data-Set-11 的源码写出值保持代码质量/Gyr。质量单位为 (2.5\times10^7M_\odot)，所以

\[
1\;[M_0/{\rm Gyr}]=0.025\;M_\odot\,{\rm yr}^{-1}.
\]

旧 HDF `long_name` 中的 `M/yr` 不足以证明已执行单位换算。新 target metadata 使用
`MACER_mass_code/Gyr`，并保存 `to_msun_per_yr=0.025`，避免固定 40 倍错误。

原始 HDF 读取还会检查 `long_name`（density、a1、a2、gas energy、SFR、vol、coolt）；
若某个运行因条件输出使 Data-Set 编号偏移，将直接报错而不是静默读取错误字段。

### 26.3 多 snapshot 的 HR 复现结果

从 Fiducial、Fiducial4、Fiducial7 三个盘星系运行中抽取 59 个早/中/晚期和末期低 SFR
快照，结果保存为 `data/macer_sfr_multisnapshot_validation.csv`。除去尚未执行恒星形成更新的
初始 `index=0` 和真值/算子都为零的晚期快照后，共有 51 个正 SFR 快照：

| run | 正 SFR 数 | median (Y/B_{\rm cont}) | min | max |
|---|---:|---:|---:|---:|
| Fiducial | 17 | 1.000029 | 0.999659 | 1.058404 |
| Fiducial4 | 17 | 1.000028 | 0.999936 | 1.000040 |
| Fiducial7 | 17 | 1.000031 | 0.999983 | 1.000041 |

一张独立真实快照回归给出

\[
B_{\rm operator}=70.99483,\qquad Y_{\rm DataSet11}=70.98279,
\qquad Y/B=0.999830.
\]

普通和高 SFR 区间已达到约 (10^{-4}\) 量级的一致性。Fiducial 末期
(Y\simeq1.14\times10^{-3}) 的极低 SFR 快照仍有约 5.84% 相对偏差；这对应很小的绝对误差，
且很可能受到 (M(<r)\simeq M_{\rm BH})、日志上下文窗口平均及浮点输出精度影响。还有一个
不可忽略的时间层差异：`star.F` 先用更新前的 (d,e) 计算并保存 `rhodot_save`，随后在同一步
修改 (d,e)，而 HDF snapshot 写出的是更新后的场和先前保存的 SFR；因此仅凭同一输出
snapshot 的后更新气体场，本来就不保证逐 bit 反演 Data-Set-11。
因此结论是“数据兼容算子已足以作为当前 LR baseline，但不是源码逐 bit 复现”。初始快照
`index=0` 的 Data-Set-11 为零而场公式给出非零，说明该输出发生在恒星形成更新启用前，
应在 HR 回归统计或训练数据质量标记中单独处理。

### 26.4 新字段的保守粗粒化

Rh、Rm-v0、Rm-v1 均已接收冷却和加速度字段。冷却时标不直接平均，而是先构造

\[
C_i=\frac{e_i}{\tau_{{\rm cool},i}},\qquad
L_C=\sum_i f_{Ci}C_iV_i,
\]

再重建

\[
C_C=L_C/V_C,\qquad \tau_{{\rm cool},C}=e_C/C_C.
\]

因此守恒的是总冷却功率 (sum CV)。加速度先转到固定柱坐标子午面基底

\[
a_R=a_r\sin\theta+a_\theta\cos\theta,\qquad
a_z=a_r\cos\theta-a_\theta\sin\theta,
\]

再守恒沉积 (m a_R,m a_z)，最后转回粗网格中心的球坐标基。这样避免 Rm 跨
(\theta) 输运时直接平均可变基分量产生非物理结果。相应冷却功率和 (R/z) 引力总力
已经加入守恒审计。

### 26.5 HDF、metadata 与 UID

processed HDF5 提升为 format version 6，新增独立 `/sfr_supervision`，而不是把监督量写入
`/snapshot`。canonical 组明确保存：

```text
Y_alpha
B_alpha
C_alpha_true
```

并强制验证 (C_\alpha^{\rm true}=Y_\alpha-B_\alpha)。两套算子的逐格 baseline、总 baseline、
上下文、参数和单位均保存。`sample_uid` 仍只标识物理场与粗粒化产品；另增
`physics_product_uid`，它对 sample UID、SFR 算子参数和上下文共同哈希，避免改变
(G,c_\star,\tau_{\rm rot}) 或 MBH/dt 来源后仍得到相同产品身份。

`GalDataSet` 的混合检查改为比较完整的 coarsening `operator_parameters` 和 SFR 参数，
但允许每个 snapshot 的 MBH/dt 数值自然变化。Y/B/C 永远不进入 CNN `input_channels`。

### 26.6 signed closure 训练接口

PT cache schema version 4 保存物理 (Y,B,C) 和 signed-asinh 闭合标签；version 3/legacy
仍可执行 direct-SFR。SFRConvNet 支持：

```text
prediction_mode=direct_sfr
prediction_mode=signed_closure
```

closure 模式中网络输出变换空间的 (widehat C)，经可微反变换后显式组成

\[
\widehat Y=B+\widehat C.
\]

主损失直接作用于归一化后的最终物理 SFR：

\[
\mathcal L_{\rm final}
=\ell\!\left(\frac{B+\widehat C}{s_C},\frac{Y}{s_C}\right).
\]

这里没有再重复添加一个与之物理等价的 (\lambda_C\mathcal L_{\rm closure})。另加可选非负软约束

\[
\mathcal L_{\rm negative}
=\left[\frac{\max(0,-(B+\widehat C))}{s_C}\right]^2,
\]

它只惩罚不可能的负总 SFR，且在负区保留梯度；最终输出不做硬裁剪。KS 经验式继续仅作为
评估 baseline，不能冒充 MACER (B_\alpha)。

### 26.7 验证和仍未执行事项

当前 66 项自动测试全部通过，覆盖两套算子分支、有限时间步、三种限制时标、温度/密度阈值、
冷却功率与固定基引力守恒、Rh/Rm-v0/Rm-v1 字段传播、HDF round-trip、UID 区分、Y=B+C、
标签不泄漏、signed 变换、显式 B+C 和负区梯度。

本轮按要求没有生成完整 PT cache，也没有启动正式训练。旧 processed HDF5 不含 Data-Set-8/9/17
及 SFR supervision，必须从原始 HDF4 重建后才能生成 schema-v4 closure cache。
