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
已经加入守恒审计。.8

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

截至 2026-08-21，当前 76 项自动测试全部通过，覆盖两套算子分支、有限时间步、三种限制时标、温度/密度阈值、
冷却功率与固定基引力守恒、Rh/Rm-v0/Rm-v1 字段传播、HDF round-trip、UID 区分、Y=B+C、
标签不泄漏、signed 变换、显式 B+C 和负区梯度。

本轮按要求没有生成完整 PT cache，也没有启动正式训练。旧 processed HDF5 不含 Data-Set-8/9/17
及 SFR supervision，必须从原始 HDF4 重建后才能生成 schema-v4 closure cache。

## 27. 2026-08-21：固定 Rh 训练条件与稀疏日志空窗口修复

固定单一分辨率 Rh 不再伪造 `mass_filter`。新 Rh CachePT 使用 schema version 5，保存
inactive 的全零 `conditioning_scalar`，并声明 `conditioning.name=none`；此时禁止分辨率
FiLM。Rm 继续原样使用 schema v3/v4 和 `log10(M_filter)`，旧 cache 可继续读取。初始化
snapshot 可按 `raw_snapshot_index=0` 精确排除，不需要删除晚期物理零 SFR。

`fiducial_4` 日志本身没有 NaN，但在 15 个 snapshot 的
\(t\pm0.000625\,\mathrm{Gyr}\) 窗口中没有日志行；旧流程对空表求均值后把 NaN 静默写入
`log_data`，新 SFR baseline 的有限性检查才将其暴露。现在盘星系日志选择采用：

1. 窗口非空：保持原窗口平均；
2. 窗口为空且有双侧日志：按 snapshot 时间对连续数值列做线性插值；
3. snapshot 落在日志时间范围外：使用最近端点。

重复日志时间会先聚合再插值。选择代码、窗口记录数、左右时间、bracket 宽度和插值权重
均进入 SFR supervision context 并参与 `physics_product_uid`。真实 `fiducial_4` 的 15 个空
窗口均得到有限正 `mbh`；首个失败点 snapshot 2 得到
\(M_{\rm BH}=8.0365833177\)，其 bracket 为
\([0.00159787288,0.00349827950]\,\mathrm{Gyr}\)。该 snapshot 的 Rh 8x8 产品满足
\(Y=B+C\)，HDF5 round-trip 保留 `linear_interpolation_empty_window` provenance。

---

## 28. 2026-09-07：预处理入口拆分为 raw→fine 与 fine→coarse

### 28.1 拆分目标与兼容策略

原 `raw_data_processing/galaxy_data.py` 同一个主程序会在每次粗粒化实验中重复执行：

```text
HDFRA 读取 → 日志匹配 → fine 写出 → coarse 写出
```

这使改变 \(R_h\) 输出形状、\(R_m\) 质量尺度或核函数时仍要重新读取较慢的原始 HDF4。
本轮保留 `galaxy_data.py` 的 `GalData/GalDataSet`、统一算子调度和旧一体化 CLI 兼容入口，
新增两个推荐入口：

```text
raw_data_processing/build_fine_from_raw.py
    raw HDFRA + Parquet log → format-v6 fine HDF5

raw_data_processing/build_coarse_from_fine.py
    format-v6 fine HDF5 → Rh / Rm-v0 / Rm-v1 coarse HDF5
```

两个脚本没有复制物理公式：raw 入口复用 `read_hdfra/set_coord/save_h5`，coarse 入口复用
`load_h5/coarsen/save_h5`。因此 Data-Set 语义、Data-Set-15 体积、冷却率密度、加速度
固定基沉积、SFR baseline、守恒审计和 metadata/UID 仍只有 `GalData` 与各粗粒化模块一套
权威实现。

`build_coarse_from_fine.py` 保持原有参数名和默认值：`--coarsener`、`--coarse-shape`、
`--mass-filter`、`--kernel`、`--coarse-dirname`、`--initial-neighbors`、
`--balance-*` 与 `--v1-*`。两个入口新增 `--snapshot-index` 和 `--max-snapshots`，用于不改
正式默认行为的抽样 smoke test；另有 `--config-dir` 和 `--fine-dirname` 以支持隔离验证。

### 28.2 新增的输入防线

fine→coarse 在运行算子前会检查：

1. metadata 必须为 complete，且 UID 可由 source/geometry/derivation 重新计算；
2. `derivation.operator_id` 必须为 `identity_fine`，防止误做 coarse→coarse；
3. 必须存在构造当前 LR MACER baseline 的密度、内能密度、冷却和径向加速度字段；
4. canonical `sfr_target/sfr_macer_baseline/sfr_closure_true` 可读取；
5. \(R_m\) 必须提供正 `mass_filter`，且核函数与 v0/v1 契约一致。

另外修复了旧 HDF4 入口的 Windows 跨盘及中文路径问题。原实现无条件调用
`os.path.relpath`，workspace 在 D:、RawData 在 E: 时会先抛 `ValueError`；中间版本改为
跨盘时传绝对路径，又会因为 pyhdf 的 HDF4 C 库不能处理中文绝对路径而报
`HDF4Error: Bad file name on open`。当前实现由 Python 临时进入源文件目录，只把 ASCII
basename `hdfra.<index>` 交给 `SD()`，文件句柄打开后立即恢复原工作目录。集成测试同时
覆盖了中文源目录和跨盘布局。

### 28.3 文档

新增：

- `raw_data_processing/build_fine_from_raw.md`；
- `raw_data_processing/build_coarse_from_fine.md`。

两份文档分别记录职责边界、完整参数表、默认值、命令、输出结构、失败语义和训练衔接。
`galaxy_data.md` 也已注明新两阶段入口；旧一体化命令仍存在，但不再推荐用于多参数粗粒化
实验。

### 28.4 小样本可靠性与训练接口验证

自动集成测试用 2 个带正确 `long_name`、Data-Set-15、Data-Set-11、坐标和日志的最小
HDF4 snapshot 实际执行 raw→fine。两个输出均能 HDF5 round-trip，形状为 \(4\times6\)，
metadata operator 为 `identity_fine`，Data-Set-11 总 SFR 为 3，并满足 \(Y=B+C\)。

另一组测试把 2 个完整 synthetic fine snapshot 经新 fine→coarse 入口降为 \(2\times3\)，
随后依次通过：

```text
GalData.load_h5
GalDataSet.load_data(require_complete_metadata=True)
CachePt.to_pt(objective_mode="direct_sfr")
CachePt.to_pt(objective_mode="signed_closure")
GalPTDataset(direct_sfr / signed_closure)
SFRConvNet + train.predict_batch
```

因此 metadata、inactive Rh conditioning、Y/B/C payload 和 `train.py` batch 接口相容。

还对真实 `data/disk_galaxy_fiducial/fine/00000.h5` 与 `00001.h5` 运行新 CLI，生成 Rh
\(8\times8\) 临时输出。与对同一 fine 对象直接调用原有 `GalData.coarsen()` 相比：

- 所有 snapshot 字段最大绝对差为 0；
- `sample_uid` 完全相同；
- metadata 为 complete，operator 为 `spatial_block`；
- `epsilon_conservation` 分别为 \(3.064\times10^{-16}\) 与
  \(2.969\times10^{-16}\)；
- 两个文件的 \(Y-(B+C)\) 均为 0。

真实临时 coarse 数据继续生成 direct/closure 两种 PT 并由 `GalPTDataset` 加载，张量形状
为 `(2, 13, 8, 8)`，Rh conditioning 均为 `none`。float32 cache 中
\(Y-B-C\) 最大绝对舍入误差为 \(3.815\times10^{-6}\)，通过既有容差。验证结束后临时 HDF5
和 PT 均已删除，没有生成正式 cache 或启动训练。

本轮完整自动测试为 89 项全部通过。该结果验证的是入口拆分和接口等价性，不代替后续对
全 snapshot、多输出分辨率和 \(R_m\) 参数组合的科学审计。

### 28.5 后续推荐工作流

1. 用 `build_fine_from_raw.py` 一次性重建并审计 format-v6 fine 数据；
2. 固定一份 fine 数据版本和 source UID 后，用 `build_coarse_from_fine.py` 分别生成
   Rh 8×8、16×16 及其他候选分辨率；
3. 每种算子/网格/质量尺度使用独立 `coarse_dirname`，不在同目录静默混合；
4. 先抽样检查守恒、\(Y=B+C\)、坐标一致性和极低 SFR，再批量生成；
5. 对同一时间块划分分别构建 direct-SFR 与 signed-closure cache，确保比较只改变学习目标，
   不改变样本集合和标准化统计量；
6. \(R_m\) 继续单独使用 `log_mass_filter`/FiLM 契约，固定单分辨率 Rh 不启用 FiLM。

## 29. 2026-09-08：Rh 8×8 训练性能优化与逐项审计

本轮在既有 `Exp1_Rh8x8_Closure_auto_*.pt` 上进行性能基准，不修改粗粒化、MACER SFR 算子、
metadata 物理契约、closure scale、\(Y=B+C\)、CNN 架构或正式 cache。未启动正式 Optuna study。

### 29.1 保留与撤回

保留验证端设备内分星系统计、CPU 整批 tensor 索引、HPO 同进程 train/val dataset 复用，
以及不重复加载测试集/追踪模型图。新增 `Experiment/training_performance.py`；
Optuna 默认使用 `tensor_cpu`，单次 train 保持 `standard` 默认，可用 `--batch_backend tensor_cpu` 启用。
tensor 路径要求 worker=0 且无在线增强，不会静默更改增强或样本数量。

逐 batch loss 的设备端累积、GPU 常驻正式入口、位置编码缓存因缺乏可靠额外收益而撤回；
CPU 线程数和 worker 增加也未被强制采用。修复 `num_workers` 隐式除二与预取配置矛盾的问题，
并保证日志 writer 在异常、中断及 OOM 路径释放。

### 29.2 最终测量与数值检查

本地 RTX 4060 Laptop、Windows、PyTorch 2.5.1，固定小训练交替重复，修正后的微基准显示：
batch64/128/512 的 train+val 单轮耗时分别减少约 8.1%/14.5%/24.0%。
完整 14615/4871 样本各训练 2 轮的补充检查，实际训练验证循环耗时减少约 12%；
cache 命中后短任务总耗时约从 5.66 秒变为 3.74 秒，但不能把短任务启动收益外推为 90 轮 trial 的加速比。

最终比较的权重最大绝对差为 0，微基准逐轮验证指标差异约 \(10^{-7}\)，来自归约精度差异。
完整回归测试 109 项全部通过；direct-SFR、base_ch=64、dropout=0.3 的 GPU 补充对照也通过。
验证仍按“各星系内部平均 → 有样本星系等权平均”聚合，而非按各星系样本数重加权。
正式训练仍测试最佳权重；HPO 不读测试集做计算。不会为了速度改变训练目标或绕过物理基线。

审计发现并修正了旧模块独立导入时 WeightedLoss 类身份不同的 benchmark 问题；
旧的部分微基准不作最终证据，已重跑并增加验证指标一致性断言。
详细步骤、有效/无效 JSON 标记、保留/撤回原因、复现命令和服务器同步文件见
`Experiment/TRAINING_PERFORMANCE_AUDIT_20260908.md`。

### 29.3 下一步

同步同一版本的 `train.py`、`run_optuna.py` 和 `training_performance.py` 到服务器。
不需要重新生成 PT。恢复已有 study 时保持原 cache 前缀、objective、loss 等固定契约不变；
性能参数不改变搜索空间，逐 trial 另写 `performance_settings` 便于追溯。
A100/Linux 的收益必须在目标实例上重新测量，不能把本地加速比当成服务器结论。

## 30. 2026-09-10：Rh 8×8 Direct/Closure 损失函数、实测比较与适用边界

当前 Direct 与 signed-closure 实验不是只把 MSE 替换为 L1，而是同时改变了预测对象、
目标变换空间和损失函数。因此，已有结果可以比较两条完整路线，但不能单独归因为
“L1 优于 MSE”。

### 30.1 Direct：`log10_1p` 目标空间的 MSE

Direct 模型学习：

\[
\boldsymbol X_{R_h}\longrightarrow \widehat y,
\qquad
y=\log_{10}(1+\mathrm{SFR}),
\]

损失为：

\[
\mathcal L_{\rm direct}
=
\frac{1}{N}\sum_i(\widehat y_i-y_i)^2.
\]

当前正式运行 `Exp1_Rh8x8_Direct_Trial27_seed42_20260909_200536` 的契约为：

```text
objective_mode = direct_sfr
loss = mse
weight_mode = none
target_transform = log10_1p
```

这一设定适合 SFR 跨越多个数量级、同时关注高低 SFR 区域，且更关心倍数/相对误差的场景。
对数变换压缩动态范围，MSE 会对 target space 中的大残差施加二次惩罚，优化通常较平滑。
但它不直接最小化物理 SFR 误差；高 SFR 爆发期的大绝对误差可能被对数压缩，
并且 target space 无偏不等于反变换后物理空间无偏。Direct 的反变换还会默认将负物理
SFR 裁剪为零，所以 `NegativePredFrac=0` 不能独立证明网络原始输出天然满足非负性。

### 30.2 Signed closure：最终物理 SFR 空间的归一化 L1

Closure 模型先预测 signed-asinh 空间的闭合项，反变换后显式组合：

\[
\widehat C_i=s_C\sinh(\widehat z_{C,i}),
\qquad
\widehat Y_i=B_i+\widehat C_i,
\]

其中当前自动确定的：

\[
s_C=226.31541789318408.
\]

主损失为归一化后最终物理 SFR 的 L1，而不是变换后 closure 本身的 L1：

\[
\mathcal L_{\rm primary}
=
\frac{1}{N}\sum_i
\left|
\frac{\widehat Y_i-Y_i}{s_C}
\right|.
\]

同时使用负 SFR 平方 hinge 惩罚：

\[
\mathcal L_{\rm negative}
=
\frac{1}{N}\sum_i
\left[
\frac{\max(-\widehat Y_i,0)}{s_C}
\right]^2,
\]

\[
\mathcal L_{\rm closure}
=
\mathcal L_{\rm primary}
+\lambda_{\rm negative}\mathcal L_{\rm negative},
\qquad
\lambda_{\rm negative}=1.
\]

当前正式运行 `Exp1_Rh8x8_Closure_BestTrial61_20260908_082411` 使用：

```text
objective_mode = signed_closure
loss = l1
weight_mode = none
closure_transform = signed_asinh
closure_scale = 226.31541789318408
negative_sfr_weight = 1.0
```

这一设定直接对齐最终物理 MAE，对少量极端 snapshot 更稳健，并保留了
\(\mathcal S_{\rm LR}+\mathcal C_\theta\) 的可解释闭合结构，适合未来作为亚网格模型嵌入模拟。
其局限是 L1 对巨大误差只线性增长，可能不足以约束灾难性失败，且条件中位数倾向可能弱化
SFR 峰值。若 baseline 很差，网络还需同时抵消 \(B\) 并重建大部分 \(Y\)。

### 30.3 当前同一 Cached Test Set 的端到端结果

两个结果使用 4873 个测试样本，MACER-LR 与 KS baseline 数值完全一致，因此可以比较
两条完整预测路线：

| Cached Test Set 物理指标 | Direct + log-MSE | Closure + physical-L1 |
|---|---:|---:|
| Physical RMSE | 26.4438 | **17.7567** |
| Physical MAE | 11.8621 | **7.7072** |
| Physical \(R^2\) | 0.996767 | **0.998542** |
| 中位绝对相对误差 | 2.6759% | **1.7662%** |
| 负最终 SFR 比例 | **0** | 0.6567% |

相对于当前 Direct，Closure 的 Physical RMSE 降低约 32.8%，Physical MAE 降低约 35.0%，
中位相对误差降低约 34.0%。因此，若项目的首要判据是最终物理 SFR 接近 MACER 真值，
当前 **Closure 完整方案明显优于 Direct 完整方案**。

Direct 在自己的 target space 中表现更好：

| Cached Test Set target-space 指标 | Direct | Closure |
|---|---:|---:|
| RMSE | **0.03222** | 0.04933 |
| MAE | **0.01744** | 0.01947 |
| \(R^2\) | **0.99790** | 0.99479 |

这与两者的优化目标一致：Direct 直接优化对数 target-space MSE，Closure 直接优化最终
物理 SFR 的归一化 L1。Closure 在 target space 中只有 4841 个有效样本，因为 32 个负预测
无法进入 `log10_1p` 变换；物理空间指标正确包含全部 4873 个样本。

### 30.4 当前可以与不可以宣称的结论

当前可以说：

> 在现有 Rh 8×8 cache、相同 Cached Test Set 与各自最优训练契约下，
> `signed-closure + physical-L1` 的物理 SFR 预测优于 `direct-SFR + log-MSE`。

当前不能说：

> L1 损失函数本身必然优于 MSE。

原因是实验同时改变了 Direct/Closure 预测对象、log/physical 损失空间、MSE/L1、
模型最优超参数和随机训练轨迹。目前两个正式结果也各只是一次运行，尚不能量化随机种子方差。
另外，当前 Exp1 的测试主要表征当前 split 下的分布内能力，不代表 time-block 外推、
leave-one-galaxy-out 或 GIZMO 输入上的迁移能力。

### 30.5 下一步损失函数与路线消融

要分离“路线”和“损失函数”的影响，至少需要四组控制实验：

| 实验 | 预测路线 | 损失/评价空间 |
|---|---|---|
| A | Direct | log-space MSE（当前对照） |
| B | Direct | physical-space L1 或 Huber |
| C | Closure | physical-space L1（当前主模型） |
| D | Closure | physical-space MSE 或 Huber |

四组必须保持相同数据 split、样本集、模型容量、HPO 预算、early-stopping 规则和预先确定的
随机种子集。不能根据测试集选择随机种子。最终应统一报告物理空间 RMSE/MAE/\(R^2\)、
相对误差分位数、负 SFR 率、每个星系的误差、SFR 峰值幅度/时刻、time-block 外推和累积新生恒星质量。

现阶段将 Closure + physical-L1 作为主模型，Direct + log-MSE 作为强数据驱动对照。
同时必须单独分析 Closure 的 32 个负 SFR 预测。若后续发现 L1 对爆发峰值的约束不足，
可测试归一化物理空间 Huber：

\[
\mathcal L
=
\operatorname{Huber}
\left(
\frac{B+\widehat C-Y}{s_C}
\right)
+\lambda_{\rm negative}\mathcal L_{\rm negative}.
\]

Huber 在小残差区近似 MSE，在大残差区近似 L1，可作为长尾 SFR 的折中候选；
其优势必须由上述控制变量实验验证，不应预先假定。

---

## 31. Exp3：Zou 五组训练/验证与 torus 域外测试（2026-09-19）

### 31.1 最终实验定义

本轮按以下固定口径实现，不再沿用旧 Exp3：

| 项目 | 新 Exp3 | 迁移后的 Exp8 |
|---|---|---|
| 数据域 | Zou 五组为开发域；`torus_100yr` 为唯一 test-only 域 | 原 Exp3 的非 low 盘星系集合 |
| Zou 运行 | `fiducial`、`fiducial_4`、`fiducial_7`、`model_0_75`、`model_1_1` | 沿用原逻辑 |
| 划分 | 每个 Zou 运行内随机 75% train / 25% validation；无 Zou test | 随机 60/20/20 |
| torus 用途 | 只写入 `test.pt`，不得进入训练、验证、归一化或 HPO | 不使用 |
| CachePT 运行平衡 | 不复制、不欠采样 | 沿用原 Exp3 行为 |
| 训练抽样 | 每个运行总权重相同；运行内按物理时间间隔加权 | 沿用原行为 |
| validation/test 抽样 | 顺序读取，不重采样 | 沿用原行为 |

旧 `.config/ExperimentSetting.json` 中四个 `Exp3_*` 配置已改名为对应 `Exp8_*`，
`data_exp` 同步由 `Exp3_` 改为 `Exp8_`。`prepare_data.py` 和 `run_train.py` 均已增加
`exp8` 入口；新 `exp3` 简写则指向 Zou→torus 实验。

用户明确要求本轮保留随机帧划分。因此，75/25 是按运行分层的随机 snapshot split，
不是 time-block split。它可用于模型选择，但相邻时间帧可能分居 train/validation，不能把
validation 结果解释为严格时间外推。真正的域外检验来自从未参与开发的 torus test。

### 31.2 数据角色与防泄漏约束

`.config/RawDataConfig.json` 为 `torus_100yr` 新增：

```json
{
  "folder_name": "DiskGalaxy/torus_100yr/torus_100yr",
  "log_file": "../torus_100yr.log",
  "mode": "filename_index",
  "snapshot_dt": 1e-7,
  "range": [0, 1000],
  "offset": 0,
  "role": "test_only"
}
```

`role=test_only` 使默认盘星系路径只返回训练角色数据，避免 torus 意外进入 Exp1、Exp2、
Exp4 或 Exp8。Exp3 再使用严格 allowlist：五组 Zou 缺一不可，torus 也必须存在；程序不会
静默用 `low`、`fiducial_10` 或其他运行补位。

Cache 构建和训练加载均检查：

1. train/validation 实际 group 恰好等于五组 Zou；test 恰好只含 torus；
2. Zou 与 torus 的输入通道顺序和 target metadata 完全一致；
3. train、validation、test 的 `sample_uid` 两两不交叉；
4. 三者的 `parent_hr_uid` 两两不交叉；
5. 归一化均值/标准差、通道 `log1p` 掩码和 closure scale 只由 train split 确定。

其中第 5 点同时修正了旧预处理的一处域外风险：旧代码会分别根据每个 split 的最小值判断
某通道是否做 `log1p`。现在由训练集确定布尔掩码并写入 PT、NPZ 和 metadata，validation/
torus 必须复用；若训练定义为非负的通道在 torus 出现负值，程序停止并要求检查坏帧或字段
语义，而不是悄悄更换变换。

### 31.3 torus raw → fine 排除规则

`build_fine_from_raw.py` 新增可重复参数 `--exclude-snapshot INDEX`。二维 HDFRA 默认仍按
排序位置跳过前 46 帧；一旦用户显式提供该参数，显式清单完整取代默认规则。新 Exp3 的
torus 主产品只排除已确认损坏的 11 帧：

```text
747, 754, 755, 763, 764, 765, 766, 767, 768, 769, 770
```

参考命令：

```powershell
python raw_data_processing/build_fine_from_raw.py `
  --galaxy disk_galaxy_torus_100yr `
  --exclude-snapshot 747 --exclude-snapshot 754 --exclude-snapshot 755 `
  --exclude-snapshot 763 --exclude-snapshot 764 --exclude-snapshot 765 `
  --exclude-snapshot 766 --exclude-snapshot 767 --exclude-snapshot 768 `
  --exclude-snapshot 769 --exclude-snapshot 770
```

排除在打开 HDF4 前发生，`--snapshot-index` 与 `--exclude-snapshot` 有交集会报错。本轮不
自动生成另一套含/不含初始帧的敏感性数据；纳入哪些 snapshot 完全由用户给出的排除清单
决定。`--exclude-raw-snapshot-index` 是后续 CachePT 的全组过滤器，不应用它承载 torus
专属的 11 帧清单，否则会把同一 index 同时从五组 Zou 中删除。

### 31.4 运行等权、物理时间加权

训练集保留每个运行的原始样本，不在 cache 中把短运行复制到最长运行。训练 loader 使用
`WeightedRandomSampler`，每个 epoch 抽取 `len(train)` 个样本并允许有放回抽样。对单个
运行按物理时间排序，使用梯形积分支持宽度：

\[
\tilde w_0=\frac{t_1-t_0}{2},\qquad
\tilde w_i=\frac{t_{i+1}-t_{i-1}}{2},\qquad
\tilde w_{n-1}=\frac{t_{n-1}-t_{n-2}}{2}.
\]

每个运行内再归一化为 \(\sum_i w_i=1\)。因此五个运行理论上各占训练抽样概率的 20%，
而运行内部的时间缺口和不同 cadence 由权重表示。单点运行权重为 1；同一运行内时间重复、
倒序或非有限时立即停止。validation 与 torus test 使用顺序 sampler，不做过采样。本轮不
加入“把高频运行降采样到 12.5 Myr”的敏感性实验。

### 31.5 Fourier 位置编码的已知限制

本轮按要求保留现有 `FourierPositionalEncoding2D`。它对每个样本分别计算：

\[
r_{\rm norm}=\frac{\log r-\min(\log r)}{\max(\log r)-\min(\log r)}.
\]

这会把 Zou 的约 100 pc–500 kpc 和 torus 的 1 pc–1 kpc 都映射为 `[0,1]`。网络可以学习
各自网格内的相对径向结构，但位置通道本身没有统一保留绝对 pc 尺度；绝对尺度只能通过
气体场、cell volume 等输入间接体现。因此 Exp3 若失败，可能同时包含物理状态域移、积分
孔径差异以及位置表示域移，不能只归因于“更靠近黑洞”。后续可比较固定物理参考尺度的
`log10(r/pc)` 编码，但不属于本轮修改。

### 31.6 执行顺序与当前状态

1. 运行 `parse_log_file.py --galaxy disk_galaxy_torus_100yr`，只生成 torus Parquet；
2. 用上述 11 个 `--exclude-snapshot` 生成 torus fine；
3. 用与 Zou 相同的算子和 `coarse_rh_12x20_sfr_v6` 生成 torus coarse；
4. 运行 `prepare_data.py --experiments exp3 --resolution-dirname coarse_rh_12x20_sfr_v6`；
5. 检查 metadata 中 75/25 样本数、五组 Zou/torus group、UID、单位、通道和训练统计来源；
6. 运行 `run_train.py --experiments exp3`，其简写会启用 `run_time_balanced` 并对固定
   12×20 R_h 自动关闭 FiLM；
7. HPO 只读取 train/validation，确定模型后才在 torus test 上做一次正式评价。

截至本节记录时，配置、数据角色、划分、采样器、训练期防泄漏检查、测试和说明文件已经
修改；尚未执行全量 torus raw→fine/coarse、Exp3 cache 生成或模型训练。旧 cache 不会被
自动删除或改名。

### 31.7 Exp3 正式测试结果与分析产物（2026-09-19）

Exp3 已完成训练，并从最佳 validation checkpoint 对 torus test-only 数据执行正式评价。
分析统一在反变换后的物理 SFR 空间进行；训练日志中的 `Test metric loss=2.714167` 是
归一化 closure 空间的训练目标，不能与下表的物理量空间指标直接比较。

| 指标 | torus test |
|---|---:|
| RMSE | 432.226165 |
| MAE | 428.434459 |
| R² | -1734.032517 |
| 总时间段 SFR 低估率 | -1225.975316% |

总时间段 SFR 低估率定义为
`100 × (∫SFR_true dt - ∫SFR_pred dt) / ∫SFR_true dt`；因此负值表示高估。该结果即模型在
整个 torus 模拟时间段内积分 SFR 高估 1225.975316%。积分前按物理时间排序、合并重复时间
点，并保留真实时间间隔。

分析入口为 `VisualScripts/ResultAnalysis/analyze_exp3.py`，Notebook 为
`VisualScripts/ResultAnalysis/Results_stat_Exp3.ipynb`。产物包括总体指标 CSV、逐帧预测
CSV、分析 metadata，以及与 Exp2 同功能的散点图、误差分布图、瞬时 SFR 时间演化图和
累计恒星质量图。本轮未加入其他统计量。

训练末尾的 `torch.load(..., weights_only=False)` FutureWarning 不影响本次已完成结果；
它提示传统 pickle 反序列化不应加载不可信 checkpoint。实际 checkpoint 已验证只含
state dict、tensor 和基础容器，因此加载改为显式 `weights_only=True`，不改变模型权重或
数值结果，并消除后续运行的该警告。

### 31.8 Exp3 分析兼容 Direct 与 Closure（2026-09-20）

`VisualScripts/ResultAnalysis/analyze_exp3.py` 与 `Results_stat_Exp3.ipynb` 已改为同时支持
`direct_sfr` 和 `signed_closure`。模式由结果目录中的 `hyperparameters.json` 与 checkpoint
自动确定并交叉校验，不根据目录名猜测。Direct 模式直接对网络输出执行目标反变换；Closure
模式继续按 `macer_sfr_baseline + closure_hat_physical` 组成最终 SFR。两种模式使用相同的
torus test-only 契约、物理 SFR 指标、全时间段积分口径和四类图表。现有 Direct 与 Closure
正式结果目录均已完成端到端运行验证。

### 31.9 Fiducial flat 绝对时间与日志匹配修复（2026-09-20）

E 盘恢复后现场核验发现，`fiducial`、`fiducial_4`、`fiducial_7` 的 flat 文件不能使用
`global_index*snapshot_dt` 作为日志绝对时间。index 0 是时间 0 的特殊初始帧；index 1 起
包含 0.5625 Gyr 的既有松弛时间。每个原始重启段推进 1000 个输出步，而扁平命名按 999
个正编号拼接，因此对 `i>0`：

\[
t_{\rm abs}=0.5625+
\left[i+\left\lfloor\frac{i-1}{999}\right\rfloor\right]0.00125\ {\rm Gyr}.
\]

修复内容：

1. 三个运行的 `RawDataConfig` 新增结构化 `time_mapping`，显式保存 index 0、正编号偏移、
   segment cycle size 和每段额外时间步；
2. `_build_snapshot_schedule` 实现 `segmented_restart_absolute`，其余运行仍沿用线性映射；
3. `GalData.read_hdfra` 读取原始文件后，按 Data-Set-11 `long_name` 的实际打印精度核验
   `AT TIME=`；不一致时停止，拒绝以错误时间匹配日志中的 `mbh/dt`；
4. 新 fine metadata 保存完整 time-mapping 契约；物理 snapshot 的 `parent_hr_uid` 保持不变，
   但由 `mbh/dt` 上下文决定的 baseline `physics_product_uid` 会变化；
5. 增加跨 999 边界的单元测试，并以 E 盘原始 HDF 对三个 flat 运行做全量时间标签核验。

现有 E 盘 `snapshot_manifest.csv` 的 `time_gyr=global_index*snapshot_dt` 是旧扁平化工具写出的
线性辅助列，不含上述松弛偏移与重启段额外步；本次不改写 RawData。新的 raw→fine 不读取
该列，而以配置的 `time_mapping` 生成绝对时间并由原始 HDF 标签校验。manifest 中的
segment/local/global index 仍可用于追溯文件来源。

该修复不原地改写已经生成的 `fine_sfr_v6`、`coarse_rh_12x20_sfr_v6`、CachePT 或既有
checkpoint；它们保留为旧实验可复现产物。后续有效 Exp3 必须使用新的版本目录重建
三个 flat 运行的 fine/coarse，重新生成 cache 并重新训练。Model0.75、Model1.1 与 torus
的现有时间映射已与原始 HDF 相符，不因本项修复重建。

建议使用 `fine_sfr_v7_timefix` 和 `coarse_rh_12x20_sfr_v7_timefix`，不要覆盖 v6。三个 flat
运行的 fine 需要修复时间及日志上下文；可以从 raw 重建，或使用第 33 节经核验的完整 v6
fine 迁移入口。Model0.75、Model1.1 和 torus 的 fine 可继续读取既有产品，但为使
Exp3 维持统一 `--resolution-dirname`，六个运行均应向新的 coarse 目录写产品。随后运行：

```powershell
python Experiment/prepare_data.py --experiments exp3 `
  --resolution-dirname coarse_rh_12x20_sfr_v7_timefix
```

重建三个 flat 时不传 `--exclude-snapshot`，会按用户指定的当前默认规则跳过排序后的前 46
个 snapshot。旧 `fine_sfr_v6` 曾把三个 flat 的 0–45 也纳入，因此 v7 的样本数变化不仅来自
metadata/日志时间修正，还包含既定稳定性排除规则真正落地；比较新旧训练时必须明确这一点。

## 32. Exp3.1 / Exp3.2 显式物理尺度输入消融（2026-10-07）

### 32.1 定义与实现范围

保留原相对 Fourier 和 snapshot 标量输出，只在 processed→CachePT 阶段追加几何通道：

| 实验 | 模式 | 通道 |
|---|---|---|
| Exp3.1 | `absolute_radius` | `log10_radius_kpc=log10(r/(1 kpc))` |
| Exp3.2 | `absolute_radius_widths` | 上述通道，加 `log10_radial_width_kpc=log10(Δr/(1 kpc))`、`log10_polar_arc_width_kpc=log10(rΔtheta/(1 kpc))` |

数据按 `(N_theta,N_r)`；使用保存的中心与 cell edges，固定 1 kpc 参考单位。
不从 coarse 中心重新猜边界，不改变 Data-Set-15 体积，不重算或替换 Data-Set-11 真值。
保存边界来自现有上游流程，其中 fine edges 由坐标中心推断；不是独立读取的原始 faces。
几何字段已 log10，只用训练集 mean/std 标准化，强制 log1p=False，不做逐帧 min-max。
原物理通道、坐标、Y/B/C 和输入排除清单保持不变；不额外排除初始 torus 帧。

`GalDataSet.load_data(geometry_input_mode=...)` 默认为 `none`，split/add 保留契约。
显式几何仅支持二维 polar、正有限半径和宽度；遇非法 shape/边界/单位契约立即报错，
不静默 clamp。几何 cache 使用独立 schema 6，原 schema 3/4/5 不变；processed HDF5
format-version 6 不变。原 12 通道配方分别增加至 13/15，模型自动获取 c_in。
第一层输入参数随通道增加，其余 backbone/池化/目标不变，故不是严格等参数量消融。

### 32.2 训练、分析与泄漏边界

PT、metadata JSON、norm NPZ、hyperparameters 和 checkpoint 保存几何公式、来源、固定单位、
通道和统计来源。加载核验 payload/sidecar、各 split 契约、追加通道与预期模式一致；
已经 log10 的字段若再次启用 log1p 会拒绝。新几何 cache 不覆盖旧文件。
共享分析支持 schema 6；HDF5 重评估从 reference cache 继承几何模式与训练 log 掩码，
必须提供训练 mean/std，不能在 torus 重估统计。Direct 和 signed_closure 均保留。

Zou 五组随机 75/25、torus test-only、UID 互斥和 run_time_balanced 不变。
宽度通道是一种局部、方向性的几何分辨率信息，但不是 R_m 的 log_mass_filter FiLM；
固定 R_h 仍 conditioning=none，FiLM=False。它也不等于 MFM 核尺度或求解器完整有效分辨率。
宇宙学模拟的局部分辨率随状态和配置变化，不能断言所有单元必然比 MACER 更宽。

`run_train.py` 新增 tracked 独立配方：`exp3.1`、`exp3.2` 默认 signed_closure Trial23；
`exp3.1-direct`、`exp3.2-direct` 使用 Direct Trial61。不继承 Exp1，不读取本机 JSON/HPO
来改变消融参数；原 exp3 简写行为不变。两路线原本容量/loss 不同，消融主要做各路线内部比较。

### 32.3 数据前提、命令与当前状态

现场检查 v7_timefix 尚未生成。正式实验先按 31.9 在新目录重建三个 flat fine 和六组统一
coarse，再生成几何 cache；原 Exp3 对照也应在同源修复产品上重训，不能把与旧 v6 的差异
全部归因于几何通道。本次不启动全量重建、不覆盖生产产品、不启动正式训练或 HPO。

完整本地 PowerShell（反引号续行）和服务器 Bash（反斜杠续行）命令、配方、cache 前缀、
前置重建及限制见 `Experiment/exp3_geometry_ablations.md`。本机 torus 可复用 fine 目录名是
`fine`，Model0.75/1.1 为 `fine_sfr_v6`；服务器必须核对 `.config/BaseConfig.json` 实际路径。

修改实现：`geometry_features.py`、`galaxy_data.py`、`geometry_contract.py`、`prepare_data.py`、
`run_train.py`、`train.py`、共享 `analysis_utils.py`，并同步对应脚本文档和 README。
新增测试：`test_geometry_input_features.py`、`test_geometry_ablation_aliases.py`、
`test_geometry_ablation_pipeline.py`。

验证覆盖：保存边界公式与方向、固定单位缩放、非法几何拒绝、默认输入不变、禁止重复 log、
训练归一化不泄漏、原字段/坐标/Y/B/C/time/UID 不变、75/25 与 test-only、旧 cache 兼容、
模式/sidecar 防错，以及两消融×两 objective 的一轮 CPU 训练→checkpoint→共享分析加载。
CPU smoke 使用合成小数据、临时目录和空 TensorBoard writer，不是正式训练，不登记 Train_history。
原生 Fiducial/torus 各一个 coarse snapshot 的几何只读 smoke 通过；未执行全量数据构建、
全量训练、GPU/AMP 或 Linux 实机验证，不能承诺该改动改善 13 倍高估。

torus 已用于失败诊断，本消融属于探索性域外研究；后续选参数只使用 Zou 内部验证，不能把
针对已知 torus 失败反复优化后的结果称为完全未接触的最终独立测试。

本轮实际验证命令：

```powershell
python -m unittest tests.test_geometry_input_features tests.test_geometry_ablation_aliases tests.test_geometry_ablation_pipeline tests.test_rh_training_conditioning tests.test_exp3_torus_contract tests.test_signed_closure_training tests.test_signed_closure_result_analysis tests.test_exp3_analysis tests.test_training_performance tests.test_metadata_contract tests.test_sfr_supervision_products tests.test_spatial_coarsening_module -q
python -m py_compile raw_data_processing/geometry_features.py raw_data_processing/galaxy_data.py Experiment/geometry_contract.py Experiment/prepare_data.py Experiment/train.py Experiment/run_train.py VisualScripts/ResultAnalysis/analysis_utils.py
```

12 个测试模块合计 79 项通过；语法检查通过。已有测试产生的 tensor→scalar 提示以及
缺少 scipy 时的 R_m 慢速 NumPy 邻居搜索提示不影响通过结果；本轮未改动这些既有行为。
为减少控制台输出，实际整批 unittest 调用经标准库 runner 汇总，测试内容与上面命令一致。

## 33. 完整二维 fine 的时间/日志迁移（2026-10-07）

源码与真实数据确认：时间错配不影响已保存的原始气体数组、坐标/权威体积和 Data-Set-11
真值 Y。因此修复必须发生在 fine 层，但不必全量重新读取 HDF4。新增
`raw_data_processing/migrate_fine_time_context.py`，只支持完整二维 format-version 6、
`identity_fine`、filename_index 盘星系数据；目录版本 v7 不代表 HDF5 format 升为 7。

迁移复用新版 raw→fine 的文件清单、数值排序、时间映射、默认前 46 帧与显式排除规则。
需要 E 盘以枚举来源和检查大小，默认复用旧 fine 的 raw SHA256，不重新通读 raw 内容。
日志每运行只读取一次，索引窗口后调用共享 `ll_dg_from_frame`；原文件入口也委托同一函数，
窗口平均、空窗口插值、重复时间平均及端点行为不变。保留 Y/单位/物理数组，重新生成全部
日志诊断、两套逐格及总 baseline/closure、context 和 physics_product_uid；snapshot UID 不变。

新增可选 metadata `time_context_migration`，内部版本 1，记录旧 fine/日志哈希、旧时间、
排除清单及是否重验原始时间标签；HDF5/schema 主版本不变。默认不读取 raw 时间标签，
显式 `--verify-raw-time` 才通过 pyhdf 重验。复用旧哈希+大小不能发现同大小 raw 篡改。
不同配置、日志、字段或筛选契约禁止混写；不提供覆盖选项，同签名才可续跑。每帧通过
内容保留和 HDF5 round-trip 检查后独占发布，同目录临时文件由本次清理。

本次只在独立 smoke 子目录处理三个运行各 `[46,999,1000,9531]`，共 12 帧。修正后的
时间为 `[0.62,1.81125,1.81375,12.4875]` Gyr。与新版 raw→fine 独立重建对照，物理数组、
坐标/体积、Y/单位/UID 完全相同，日志在 rtol=1e-12/atol=1e-14 内一致，两套逐格 baseline
与物理产品 UID 一致。原始 HDF 时间标签全通过；同参数续跑跳过 12 帧，新写出 0。
两套 fine 分别生成 12×20 coarse，共 24 个 smoke coarse；对应数组、两套 B/C 与 UID
一致，三个迁移目录完整 metadata dataset 加载通过。生产 v6、已有 v7、cache/checkpoint
及训练配置均未改写；未执行全量迁移、CachePT 生成、训练或全量性能测量。

修改文件：迁移脚本及独立说明、`parse_log_file.py` 的共享内存日志入口、新增迁移测试，
以及 README、raw→fine 说明和本研究计划。测试环境是已有的
`D:/conda_envs/physics_ai/python.exe`，未安装依赖或修改 base。

```powershell
& 'D:/conda_envs/physics_ai/python.exe' -B -X utf8 -m unittest tests.test_fine_time_context_migration tests.test_log_context_selection tests.test_split_snapshot_builders tests.test_metadata_contract tests.test_sfr_supervision_products tests.test_spatial_coarsening_module -q
```

具体全量/抽样命令、安全限制、测试目录和迁移契约见
`raw_data_processing/migrate_fine_time_context.md`。正式流程后续必须重新构建 coarse、CachePT
与训练统计并重训。默认排除前 46 帧仍是独立的样本变化，比较新旧实验时必须说明。

上述 6 个模块合计 39 项通过，包括零 Y 与负 closure、拒绝混写和失败临时文件清理；
迁移脚本、日志脚本和新测试的 3 个源文件内存编译检查通过，未写入 pyc。现场只读检查
发现 Fiducial 的 `fine_sfr_v7` 已有 raw 重建产物（核验时 261 帧），没有修改或清理。
全量迁移命令因此使用尚未生成的独立 `fine_sfr_v7_timefix`，不要向旧 v7 混写。

## 2026-11-02：方案 C 的二维逐格 baseline 与监督数据层

本次用户范围仅为逐格 baseline/监督及前置适配，不修改原非逐格 CNN。
实现依据是当前源码、Data-Set-11 既有单位契约、合成 HDF4 集成测试及已有真实 processed 产品；
没有改变 MACER 物理参数，也没有重新以 HR 算子替代标签。

### 定义与实现

对每个 R_h coarse cell：`Y_i=sum(DataSet11_j)`，`B_i=S_i[R_h(U_HR)]`，
`C_i=Y_i-B_i`。数组形状为 `(N_theta,N_r)`；每格已经积分，单位 code SFR，不能再乘体积。
默认数据兼容 baseline 和论文物理 baseline 都保留，分别生成 signed closure。
验证三张图求和等于原标量 Y/B/C；零标签和负 closure 均保留。

- `cell_sfr_supervision.py`：纯数组物理检查、摘要和独立逐格 product UID。
- `GalData`：raw→fine 保留局部真值，R_h 使用与气体相同的索引分块求和，HDF5 保存/回读校验。
- format-version 6 及原标量契约保持不变；新增可选 `metadata.cell_supervision/version=1`。
  canonical 新增 `Y_alpha_cell/B_alpha_cell/C_alpha_true_cell`；每种物理模式新增局部 closure。
  监督字段不进入 snapshot 输入或 GalDataSet 通道。
- `build_cell_sfr_supervision.py`：已有 fine/R_h 可直接从原 HDF4 补齐并另存。核验 raw SHA256、
  shape、权威体积映射和标签总量；保留 gas、日志、时间、sample UID、原总量和已存 baseline。
- `build_coarse_from_fine.py --require-cell-supervision`：可选严格模式，缺失标签即报错，拒绝覆盖。
  默认关闭，不破坏旧 scalar fine 产品的使用。
- `Experiment/build_cell_supervision_cache.py`：按原标量 cache UID/顺序生成独立监督 sidecar，
  保留过采样重复项，并核验 parent UID、坐标、shape、时间和总量。源 cache SHA256 绑定附加文件。
  不重新 split/标准化、不改原 cache，不把格子单独随机划分。

目前局部标签只支持二维 identity_fine/R_h。R_m 仍支持原标量流程，但缺乏真实输运权重时不
传播逐格标签。三维、新逐格 CNN、局部 loss/scale、HPO 和空间结果分析不在本次实现范围。
原 `prepare_data.py/models.py/train.py/run_train.py/run_optuna.py` 均未修改。
局部监督 sidecar 不能直接作为旧 `run_train.py` 的完整数据集；后续需独立空间 dataset/model。
局部 closure scale 应在后续仅用训练集确定，不能直接套原总量 scale。

### 验证记录与限制

使用现有 `D:/conda_envs/physics_ai/python.exe`，未安装新依赖；该环境没有 pytest，使用 unittest。

```powershell
& 'D:/conda_envs/physics_ai/python.exe' -m unittest tests.test_cell_sfr_supervision tests.test_split_snapshot_builders tests.test_metadata_contract tests.test_fine_time_context_migration tests.test_spatial_coarsening_module tests.test_thermodynamic_semantics tests.test_sfr_supervision_products tests.test_signed_closure_training tests.test_rh_training_conditioning -q
```

以上 62 项通过：包括非均匀体积下避免重复积分、两套 baseline、HDF5 round-trip、
零 SFR/负 closure、局部错位但总量不变的拒绝、缺失/非法数组、旧 scalar/R_m 兼容、
过采样 UID 顺序、源 cache 不变，以及两帧合成 HDF4 的 raw→fine→R_h 和旧 fine/coarse 补齐。
旧 direct/closure CachePT 与原 SFRConvNet 前向接口的回归通过；没有进行正式训练。

只读抽查五组真实 `coarse_rh_12x20_sfr_v7` 各 3 帧，共 15 个旧产品：

| 运行 | 原始 snapshot index |
|---|---|
| fiducial | 46、5206、9531 |
| fiducial_4 | 46、4640、9531 |
| fiducial_7 | 46、3558、9531 |
| model_0_75 | 46、523、999 |
| model_1_1 | 46、523、999 |

均可加载，两套 baseline 地图求和与原总量一致，原 Y=B+C 成立。
本次 E 盘未挂载，无法读取真实 raw Data-Set-11，因此**没有完成真实数据的逐格补齐或全量 cache**，
不能将合成测试称为真实 HR 标签验证。挂载后先少量补齐并核验，再批量执行。
未覆盖用户已有产品/cache/checkpoint，不向 Train_history 登记任何训练结果。

完整命令及参数见 `raw_data_processing/build_cell_sfr_supervision.md` 与
`Experiment/build_cell_supervision_cache.md`。

## 2026-11-02：独立逐格 SFRUNet、物理损失与数据适配器

本节更新上一条“逐格 CNN、局部 loss 尚未实现”的状态。原标量模型/训练入口保持不变。

### 已实现的模型与目标

- 新增 `Experiment/SFRUNet.py`：两次下采样、三尺度 32/64/128 通道的残差 U-Net。
  默认 E1/E2/瓶颈/D2/D1 各两块；对应 12×20→6×10→3×5→6×10→12×20。
  双线性隐藏特征上采样、1×1 通道投影、Concat 跳跃连接，线性逐格头。
  复用现有残差块、SE、BN、Fourier 坐标编码；固定 R_h 关闭 FiLM。
  默认 4 bands、max_freq=4、p_drop=0；12 物理输入通道时参数量 1,156,097。
  瓶颈可减至一块（852,321 参数）；未因参数量更少宣称效果更好。
- 新增 `cell_sfr_objective.py`：输出 z 经 `C_hat=s_C*sinh(z)` 后计算 `Y_hat=B+C_hat`。
  最终总 SFR 为地图求和，不重复乘体积。baseline 不作为网络输入。
- 损失为归一化物理空间的逐格 MSE、总 SFR MSE、逐格负 SFR 平方惩罚之和。
  后两项权重默认均为 1，可消融；零标签纳入监督；不增加重复 closure MSE，不裁剪预测。
  这不是变换域 z 的 MSE，signed-asinh 仅定义输出参数化。
- closure scale 为训练集非零 |Y-B| 中位数；cell/total loss 尺度分别为训练 Y 地图和总量 RMS。
  训练过采样重复项保留，全零尺度显式回退到 1 code SFR；验证/test/out 不能重新估计。
  新增独立 scales/loss/model 契约，原 HDF5/scalar metadata schema 不变。
- 新增 `cell_sfr_dataset.py`：原 scalar cache + 逐格 sidecar→字典 batch。
  校验源 SHA256、样本/母帧 UID 顺序、地图摘要、Y=B+C、原标量总和及输入通道。
  当前限定二维 R_h、data_compatible baseline、conditioning=none；跨 split 检查 UID/parent 隔离。
  支持已有 Direct 或 Closure scalar cache，只要完整物理监督及契约存在。

### 验证及证据

```powershell
python Experiment/SFRUNet.py --smoke-test
python -m unittest tests.test_sfr_unet tests.test_cell_sfr_supervision tests.test_split_snapshot_builders tests.test_metadata_contract tests.test_fine_time_context_migration tests.test_spatial_coarsening_module tests.test_thermodynamic_semantics tests.test_sfr_supervision_products tests.test_signed_closure_training tests.test_rh_training_conditioning -q
```

使用本机 physics_ai 环境，CPU 上合计 **71 项测试通过**（新测试 9 项）。
包含奇数尺寸、完整参数反向传播、state_dict 恢复、负 closure/零真值、手算损失、
总量相同而局部分布错误的区分、物理单位缩放不变性、拒绝非有限损失/隐式广播、
训练尺度隔离、sidecar 篡改与跨 split UID 重叠检测、旧标量 Direct/Closure 回归。

本次 E 盘已挂载，额外在临时目录完成五组真实盘星系各 3 帧，共 15 帧的
raw Data-Set-11→旧 R_h 产品逐格补齐→HDF5 回读→scalar smoke cache→sidecar→Dataset→网络/loss。
抽样 index 与上一节的 15 帧相同；逐帧核验原始文件 SHA256、体积映射，
以及新旧 snapshot 所有气体字段逐元素完全相等。
真实 Y 地图求和相对于原 scalar Y 的最大误差为 0；抽样总 Y 范围为
5.744094974–1630.801167303 code SFR，**没有覆盖真实全零帧**，全零行为由合成测试覆盖。
CPU、seed42、Adam lr=1e-4，全 15 帧 batch 仅做 3 步梯度 smoke：
loss 为 0.7217402、0.7215835、0.7214270，所有可训练参数梯度均存在且有限。
此数值只表明接口/反向传播跑通，不能说明拟合、泛化或 burst 预测能力。
临时数据自动清理，未修改原产品/cache/checkpoint，未生成全量 cache，未登记 Train_history。

### 尚未完成与下一步

网络/损失/数据 API 已可组合使用；正式逐格训练、断点恢复、HPO、地图评估入口尚未接通。
旧 `run_train.py/run_optuna.py` 仍仅适用于原标量模型，不能直接替换模型名运行。
后续建立独立空间训练入口，保存模型/损失/尺度/输入归一化/split 完整契约，
先在固定 split 验证，再比较逐格、总量与 burst 分段误差。
未验证 CUDA/AMP 性能、全量数据适配、R_m 局部标签、3D 或真实 GIZMO 泛化。
详细接口、参数和使用示例见 `Experiment/SFRUNet.md`。
