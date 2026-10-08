# MACNet CNN 训练参数、损失函数与关键过程说明

## 1. 文档范围与代码依据

本文只解释当前代码中的二维 `SFRConvNet` 及其训练入口，不把一般 CNN 教材中的常见做法误写成项目已经实现的行为。主要依据如下：

- 网络结构：[`Experiment/models.py`](Experiment/models.py)，重点是 `FourierPositionalEncoding2D`、`SEBlock`、`ConvFiLMBlock` 和 `SFRConvNet`。
- 训练入口：[`Experiment/train.py`](Experiment/train.py)，重点是参数解析、损失构造、`WarmupCosineScheduler`、`train_one_epoch` 和模型实例化。
- 数据批处理：[`Experiment/training_performance.py`](Experiment/training_performance.py)，重点是 `DataLoader` 与 `TensorBatchLoader` 的行为。
- HPO 搜索空间：[`Experiment/run_optuna.py`](Experiment/run_optuna.py)。
- SFR/closure 数值变换：[`Experiment/target_transform.py`](Experiment/target_transform.py)。

文中“默认值”若无额外说明，指 `Experiment/train.py::parse_args()` 的命令行默认值。`SFRConvNet` 类自身也有构造默认值，但正式训练时会由 `train.py` 显式传参覆盖。例如，`pos_max_freq` 的模型类默认值是 `64.0`，训练入口默认值却是 `101`；实际由训练入口启动时采用后者。

## 2. 当前 CNN 的整体数据流

设一个批次的物理场输入为

\[
x\in\mathbb{R}^{B\times C_{in}\times H\times W},
\quad r,\theta\in\mathbb{R}^{B\times1\times H\times W}.
\]

当前 `SFRConvNet` 的前向过程为：

1. 对 `r`、`theta` 做 Fourier 位置编码，得到 `pos`；
2. 在通道维拼接 `x` 与 `pos`；
3. stem 卷积把拼接结果映射到 `base_ch` 个通道；
4. 依次通过多个残差 stage；除第 0 个 stage 外，每个 stage 的第一个 block 用 `stride=2` 下采样，并将通道数翻倍；
5. 每个 block 内使用 BatchNorm、卷积、可选 FiLM、Dropout2d、SE 通道注意力和残差连接；
6. 回归头做全局平均池化、Flatten、LayerNorm、全连接、GELU、Dropout 和最终全连接，输出每个样本一个无界标量。

对应代码为 `models.py:615-755`。输出标量的物理语义由 `objective_mode` 决定：`direct_sfr` 表示变换后的 SFR，`signed_closure` 表示变换后的 closure；最终物理量的组合在训练代码中完成，而不在网络头内完成。

## 3. 模型结构参数

### 3.1 `base_ch`

**严格定义：** `base_ch` 是 stem 的输出通道数，也是第 0 个 stage 的通道数。第 `i` 个 stage 的通道数为

\[
C_i=\texttt{base\_ch}\times 2^i,
\qquad i=0,1,\ldots,L-1,
\]

其中 `L = len(stage_depths)`。

代码对应：

- stem：`Conv2d(stem_in, base_ch, ...)`，见 `models.py:676-682`；
- stage 通道：`out_ch = base_ch * (2 ** i)`，见 `models.py:684-704`；
- 默认值：`64`，见 `train.py:797`。

当默认 `base_ch=64`、`stage_depths=(2,2,2,2)` 时，各 stage 通道数为 `64, 128, 256, 512`，回归头首先接收 512 维向量。增大 `base_ch` 会同时增宽 stem、所有 stage、SE MLP、可选 FiLM 生成器的输出以及回归头，因此参数量和计算量的增长不是只发生在第一层。对于相邻的普通 3×3 卷积，参数量的主项与通道数乘积成正比；整体宽度同比放大时，许多层的参数量近似按宽度平方增长。

`run_optuna.py:204` 对它搜索 `[32, 64, 128]`。

### 3.2 `stage_depth` 与代码中的 `stage_depths`

用户所列的 `stage_depth` 在当前代码中没有同名单数参数；实际参数是复数形式 `stage_depths`。

**严格定义：** `stage_depths=(d_0,d_1,\ldots,d_{L-1})` 是一个整数元组，`d_i` 表示第 `i` 个 stage 中 `ConvFiLMBlock` 的数量。block 总数为

\[
D=\sum_{i=0}^{L-1}d_i.
\]

默认命令行字符串是 `"2,2,2,2"`，经 `_parse_stage_depths()` 转成 `(2,2,2,2)`，即 4 个 stage、每个 2 个 block、共 8 个 block。解析与使用分别见 `train.py:799-803`、`train.py:841-857`、`train.py:1621-1629` 和 `models.py:688-704`。

stage 边界同时决定宽度和空间尺度：

- stage 0 的所有 block 都是 `stride=1`；
- 对 `i>0`，只有该 stage 的第一个 block，即 `j=0`，使用 `stride=2`；
- 同一 stage 的其余 block 使用 `stride=1`。

因此，若忽略奇数尺寸的取整，第 `i` 个 stage 的空间尺寸约为 `(H/2^i, W/2^i)`。

严格限制方面，解析器目前只负责把文本转成整数元组，没有显式检查元组非空或每项为正数。因此“能被解析”不等于“是有效架构”：空元组会在 `stage_channels[-1]` 处失败；包含 0 或负深度也可能产生空 stage 或通道衔接错误。实际配置应使用非空正整数元组。

当前 `run_optuna.py` 没有搜索 `stage_depths`，所以 HPO trial 沿用训练默认值 `(2,2,2,2)`。

### 3.3 `stride`

**严格定义：** stride 是卷积核在输入空间上相邻两次采样位置的步长。对单个空间维度，卷积输出尺寸为

\[
L_{out}=\left\lfloor
\frac{L_{in}+2p-d(k-1)-1}{s}+1
\right\rfloor,
\]

其中 `p` 是 padding，`d` 是 dilation，`k` 是 kernel size，`s` 是 stride。

`ConvFiLMBlock` 的主分支第一层是 `kernel_size=3, padding=1, dilation=1`。因此：

- `stride=1` 时输出尺寸不变；
- `stride=2` 时输出尺寸为 `floor((L_in+1)/2)`，也就是 `ceil(L_in/2)`。

当 `stride=2` 或输入输出通道不同，skip 分支同时使用 `1×1`、相同 stride 的卷积，使残差相加前两条分支形状一致。见 `models.py:542-552`。当前 SFRConvNet 的 stride 不是命令行超参数，而是由 stage 位置自动决定，见 `models.py:692-700`。

一个重要的项目事实是：当前 CNN 的空间下采样由 stride-2 卷积完成，不是由池化层完成。

### 3.4 Batch Normalization / `BatchNorm2d`

`batch norm` 不是当前命令行参数，而是网络中固定使用的模块。stem 使用一次 `BatchNorm2d(base_ch)`；每个 `ConvFiLMBlock` 使用两个 `BatchNorm2d`，见 `models.py:679-681` 和 `models.py:542-545`。

对输入 `u∈R^(B×C×H×W)`，训练态下每个通道 `c` 在 batch 与空间维 `(B,H,W)` 上计算统计量：

\[
\mu_c=\frac{1}{BHW}\sum_{b,h,w}u_{bchw},
\qquad
\sigma_c^2=\frac{1}{BHW}\sum_{b,h,w}(u_{bchw}-\mu_c)^2,
\]

\[
y_{bchw}=\gamma_c
\frac{u_{bchw}-\mu_c}{\sqrt{\sigma_c^2+\varepsilon}}
+\beta_c.
\]

这里 `gamma_c`、`beta_c` 是可学习仿射参数。代码没有覆写 PyTorch 构造参数，因此采用 `BatchNorm2d` 默认配置：`eps=1e-5`、`momentum=0.1`、`affine=True`、`track_running_stats=True`。PyTorch 在训练前向归一化时使用有偏方差估计，但更新 running variance 时使用无偏估计；评估态使用累计的 running mean/variance，而不是当前验证 batch 的统计量。

本网络的具体排列也应区分：

- stem 是 `Conv → BN → ReLU`；
- 残差 block 是 pre-activation 风格：`BN → ReLU → Conv → BN → [FiLM] → ReLU → Dropout2d → Conv → SE → 残差相加`。

因此，`batch_size` 会影响训练态 BatchNorm 的统计噪声。因为 `drop_last=False`，最后一个 batch 可以小于配置值；若最后一批恰好只有 1 个样本，并且某个 BatchNorm 的特征图已经变成 `1×1`，每通道只有一个统计值，PyTorch 训练态 BatchNorm 会报错。当前数据加载代码没有通过丢弃尾批来规避此情况。

### 3.5 `pos_num_bands`

**严格定义：** `pos_num_bands=N_f` 是每个坐标 `r`、`theta` 所使用的 Fourier 频带数量。频率缓冲区由

```python
torch.logspace(0.0, log10(pos_max_freq), steps=pos_num_bands)
```

构造，见 `models.py:218-221`。当 `N_f>1` 时，第 `k` 个频率为

\[
f_k=10^{\frac{k}{N_f-1}\log_{10}(f_{max})},
\qquad k=0,\ldots,N_f-1.
\]

即从 1 到 `pos_max_freq` 对数均匀取点，而非线性均匀取点。每个频带对每个坐标各产生一个正弦和一个余弦通道。因为 SFRConvNet 设置 `include_input=True`，还保留一个归一化半径通道和一个原始角度通道，所以位置编码输出通道数严格为

\[
D_{pos}=2+4N_f.
\]

默认 `N_f=32` 时，`D_pos=130`；stem 的实际输入通道数为 `C_in+130`，不是固定的 145。只有当 `C_in=15` 时才是 145。相关代码见 `models.py:232-236` 和 `models.py:667-680`。

训练入口默认值为 `32`，当前 HPO v2 的可选搜索集合是 `[4,8,16]`。默认完整 HPO 会搜索它；若通过 `--search-params` 排除该参数，则固定使用训练入口默认值。有效配置应为正整数；当前训练参数解析器没有提前验证该条件。

#### 增大 `pos_num_bands` 的收益、代价与当前项目判断

在保持 `pos_max_freq` 不变时，增加 `pos_num_bands` 的本质是：在频率区间 `[1,pos_max_freq]` 内放入更多对数间隔的采样点。它提高的是**频率采样密度**，不会提高输入网格分辨率，也不会继续提高最高频率。

例如 `pos_max_freq=16` 时：

```text
pos_num_bands=2: [1, 16]
pos_num_bands=3: [1, 4, 16]
pos_num_bands=5: [1, 2, 4, 8, 16]
```

还要注意，这些频带通常不是嵌套集合。增加 bands 后，除首尾频率外，中间频率会重新排布；因此比较两个 `pos_num_bands` 时，不只是“在原编码末尾增加通道”，而是更换了一组中间基函数。

**可能的优点：**

- **覆盖更多空间尺度。** 当真正有用的位置依赖尺度未知时，更密集的频带提高了某些频率接近合适尺度的概率。
- **降低后续网络合成位置函数的难度。** stem 的线性卷积可以直接组合更多正余弦基，不必完全依靠深层 ReLU/GELU 合成复杂的位置关系。
- **改善多尺度目标的拟合能力。** 如果目标同时具有缓慢的径向趋势和较细的局部位置变化，低、中、高频同时存在可能比少量频带更容易优化。
- **在 bands 太少时减少频率覆盖空洞。** 例如只有 `[1,101]` 两个频率时，中间尺度完全缺失；适量增加 bands 能补上这些尺度。

这些只是表达能力上的可能收益。网络能否利用新增通道，仍取决于网格采样、目标规律、stem 宽度和训练数据量。

**可能的缺点：**

- **通道数线性增加。** `D_pos=2+4N_f`，每增加一个 band，就增加 `r_sin/r_cos/theta_sin/theta_cos` 四个通道。
- **stem 参数和计算增加。** 当前 3×3 stem 每增加一个 band，就增加

  \[
  3\times3\times4\times base\_ch=36\times base\_ch
  \]

  个卷积权重。默认 `base_ch=64` 时是每 band 2,304 个权重。
- **激活与显存增加。** 仅位置 tensor 就会为每个新增 band 增加 `4×B×H×W` 个元素，随后拼接 tensor 和 stem 卷积也要处理这些通道。
- **有限网格上的线性冗余。** 网格只有有限位置时，许多频率可能产生线性相关或近似相关的离散模式；通道更多不等于新信息更多。
- **更容易学习网格点身份而非平滑物理规律。** 当前最高频率很高，密集加入中高频后，网络可能更容易记住固定网格位置，导致同网格训练损失降低但跨星系或跨网格泛化不改善。
- **输入相关性可能使优化条件变差。** 高度相关的正余弦通道会让 stem 面对多种近似等价的权重组合，可能增加优化方差，而非稳定改善训练。
- **可能挤占模型容量。** 当 `base_ch` 固定时，stem 最终仍压缩到相同的 `base_ch` 个输出通道。过多位置通道会与真正的物理场通道共同竞争这有限的输出表示。

对当前 `12×20` cache，在固定 `pos_max_freq=101` 的条件下，直接用当前编码器计算 `240×D_pos` 位置矩阵的数值秩，结果为：

| `pos_num_bands` | `D_pos` | 数值秩 |
|---:|---:|---:|
| 2 | 10 | 10 |
| 4 | 18 | 18 |
| 8 | 34 | 29 |
| 16 | 66 | 31 |
| 32 | 130 | 31 |
| 64 | 258 | 31 |

这表明在当前固定网格上，bands 从 16 增加到 32 或 64 时，位置通道数继续翻倍，但没有增加线性独立的离散空间模式。它不能证明 16 的验证精度一定优于 32，因为不同频带仍可能改变优化难度；但它说明继续增加 bands 的边际信息收益很可能已经饱和，而参数、计算和过拟合风险仍在上升。

因此，对当前项目更合理的判断是：

- 从非常小的值增加到 `4` 或 `8` 可能明显补足尺度覆盖；
- `8→16` 仍可能有收益，应由训练验证；
- `16→32→64` 不应预设会更好，当前网格证据反而提示强冗余；
- 不应只调整 bands 而固定 `pos_max_freq=101`，因为“频带数量”和“频率范围”需要联合决定。

建议优先联合测试：

```text
pos_num_bands ∈ {4, 8, 16}
pos_max_freq ∈ {2, 4, 8, 16, 32, 101}
```

选择标准应以多随机种子的验证指标、各星系分组损失、物理 SFR MAE 和 train/validation gap 为主，而不是以训练 loss 最低或通道数最多为准。

### 3.6 `pos_max_freq`

**严格定义：** `pos_max_freq=f_max` 是上述对数频带序列的上端点，不是采样率，也不是直接以物理长度为单位的“最大频率”。它在径向和角向分支中的实际含义由当前公式共同决定。

训练入口默认值是 `101`，见 `train.py:795`；`SFRConvNet` 类签名自身默认是 `64.0`，见 `models.py:642`，但通过 `train.py` 启动时会显式传入 `101`。当前 HPO v2 的可选搜索集合为 `[2,4,8,16,32,101]`；默认完整 HPO 会搜索它，子集搜索排除它时则固定为 `101`。

`f_max` 必须大于 0 才能计算 `log10(f_max)`，而通常还应满足 `f_max≥1`，以保持代码注释所说的“从 1 增长到最大频率”的语义。解析器目前未显式验证。若 `pos_num_bands=1`，`torch.logspace(..., steps=1)` 只产生起点 1，此时 `pos_max_freq` 实际不会成为频带值。

### 3.7 `p_drop`

**严格定义：** `p_drop=p` 是训练态随机失活概率。当前同一个参数同时控制两种不同的 dropout：

- 每个 `ConvFiLMBlock` 中的 `nn.Dropout2d(p)`：对四维特征图按样本随机将整个通道平面置零；见 `models.py:546-547`、`models.py:571-573`。
- 回归头中的 `nn.Dropout(p)`：对池化后的特征向量逐元素随机置零；见 `models.py:715-723`。

训练态下，被保留的元素/通道按 `1/(1-p)` 放大，使输出期望值不变；`model.eval()` 时 dropout 变成恒等映射。`p=0` 等价于关闭 dropout。PyTorch 要求 `0≤p≤1`，但实际训练通常不应取 `p=1`。

训练入口默认值是 `0.3`，而模型类自身默认是 `0.1`；正式训练入口显式传入前者。HPO 从 `0.0` 到 `0.5`、步长 `0.1` 搜索，见 `run_optuna.py:210`。

### 3.8 `se_reduction`

**严格定义：** `se_reduction=r_se` 控制 SE 通道注意力 MLP 的中间宽度。对一个具有 `C` 个通道的 block，中间维度不是始终严格等于 `C/r_se`，而是代码中的

\[
C_{mid}=\max\left(\left\lfloor\frac{C}{r_{se}}\right\rfloor,8\right).
\]

见 `models.py:503-515`。因此增大 `se_reduction` 通常会缩小 SE MLP，但当 `C//r_se≤8` 时会受到下限 8 的约束，继续增大 reduction 不再缩小中间层。

训练入口默认值为 `4`；HPO 搜索 `[4,8,16]`，见 `train.py:804` 和 `run_optuna.py:205`。它应为正整数，否则整数除法会失败或失去预期语义。

## 4. batch、batch size 与数据加载参数

### 4.1 `batch`

当前代码中没有名为 `--batch` 的训练参数。`batch` 是 `for batch in loader` 每次迭代得到的一组字段，随后由 `_move_batch()` 解包，见 `train.py:1117-1138` 和 `train.py:1158-1161`。

两种目标模式的 batch 结构不同：

| 模式 | batch 字段顺序 |
|---|---|
| `direct_sfr` | `(x, coord, conditioning_scalar, y, groups, n_groups)` |
| `signed_closure` | `(x, coord, conditioning_scalar, y, macer_sfr_baseline, target_sfr_physical, groups, n_groups)` |

这些字段来自 `GalPTDataset.__getitem__()`，见 `train.py:611-627`。其中：

- `x` 是成批物理场，形状为 `(B,C_in,H,W)`；
- `coord` 的前两个通道分别拆成 `r` 与 `theta`；
- `conditioning_scalar` 是 cache 契约指定的条件标量；固定 `R_h` 的 version-5 cache 使用 inactive 全零占位，活动 `R_m` cache 才表示 `log10(M_filter)`；
- `y` 在 direct 模式是缓存的 SFR 目标，在 closure 模式被切换为变换后的真实 closure；
- `groups` 是星系/域编号；
- `n_groups` 是组数元数据。

`batch` 是一次梯度更新所消费的数据容器；`batch_size` 才是其目标样本数配置。

### 4.2 `batch_size` / “batch size”

**严格定义：** `batch_size=B_cfg` 是 DataLoader 计划在一次迭代中组合的最大样本数。命令行参数名严格为 `--batch_size`，默认 `64`，见 `train.py:699`。

标准和 tensor 两条数据路径都设置 `drop_last=False`，因此对含 `N` 个样本的数据集：

\[
\text{batches per epoch}=\left\lceil\frac{N}{B_{cfg}}\right\rceil,
\]

最后一个 batch 的实际大小为 `N mod B_cfg`；若能整除则为 `B_cfg`。见 `training_performance.py:67-69`、`training_performance.py:95-106`。

训练时每个 batch 只进行一次反向传播和一次 `optimizer.step()`，所以增大 batch size 会减少每个 epoch 的优化步数，也会改变：

- 梯度估计的样本集合；
- BatchNorm 的 batch 统计量；
- 单步内存/显存占用；
- scheduler 的 `steps_per_epoch` 与 `total_steps`。

`train_one_epoch()` 用实际 `bs=y.size(0)` 汇总各 batch 的均值损失，因此尾批按其真实样本数计入 epoch 训练均值，见 `train.py:1212-1215`。

HPO 搜索的 batch size 为 `[64,128,256,512,1024]`，见 `run_optuna.py:213-215`。能否使用较大值取决于输入分辨率、通道宽度、AMP 和设备内存，代码本身不保证这些候选都不会 OOM。

### 4.3 `num_workers`

**严格定义：** `num_workers=W_loader` 是标准 PyTorch `DataLoader` 用于并行取得/整理样本的 worker 子进程数；它不是 CPU 线程数，也不改变模型并行度。

- `num_workers=0`：数据读取在训练主进程内完成；
- `num_workers>0`：使用指定数量的 worker 进程。

默认值与操作系统有关：Windows (`os.name == "nt"`) 默认 `0`；其他系统默认 `min(16, floor(cpu_count/2))`，见 `train.py:722-727`。负数会在参数解析后被拒绝，见 `train.py:833-837`。

它只对 `batch_backend=standard` 有效。`batch_backend=tensor_cpu` 明确要求 `num_workers=0`，否则抛出异常，因为该路径直接对内存 tensor 做整批索引，见 `training_performance.py:51-60`。

在标准路径中，若 `num_workers>0`：

- Windows 会强制 `persistent_workers=True`；
- 其他系统只有显式传入 `--persistent_workers` 才保持 worker 跨 epoch 存活。

见 `training_performance.py:63-69`。

### 4.4 `prefetch_factor`

**严格定义：** `prefetch_factor=F_prefetch` 是标准 DataLoader 在多 worker 情况下为**每个 worker**预取的 batch 数，而不是样本数。代码只在 `num_workers>0` 时把它传给 DataLoader，并使用

```python
prefetch_factor=max(1, args.prefetch_factor)
```

见 `training_performance.py:63-66`。因此命令行默认值是 `3`，但小于 1 的输入会被钳制为 1；当 `num_workers=0` 或使用 `tensor_cpu` backend 时，此参数不生效。

稳态下可理解为最多约有 `num_workers × prefetch_factor` 个 batch 任务处于预取队列/处理中，但这不是对同时驻留内存数据量的严格上界，因为还涉及正在消费的 batch、进程间队列和 pin-memory 复制。增大它只可能缓解数据供应等待，不会改变模型、损失或梯度定义，同时会增加 CPU 内存与预取开销。

## 5. 优化参数

### 5.1 `lr`

**严格定义：** `lr=eta_base` 是传给 AdamW 的基础/峰值学习率，见 `train.py:1682`。命令行默认值为 `1e-3`，见 `train.py:701`；HPO 在 `[1e-5,1e-2]` 上做对数采样，见 `run_optuna.py:211`。

它不是在所有更新步中保持不变的常数。当前代码随后创建 `WarmupCosineScheduler`，并逐优化 step 更新 optimizer 的学习率。因此更准确地说，`lr` 是 scheduler 的 `base_lrs`，也就是 warmup 结束时的峰值。

尤其要注意默认配置：`lr_warmup_steps=0` 但 `lr_warmup_ratio=0.2`，代码会自动令 warmup 步数为 `int(total_steps×0.2)`，并且默认 `lr_warmup_start_factor=0.0`。只要由此算出的 warmup 步数大于 0，scheduler 就会在构造时立即把第 0 步学习率设为 0，所以第一次 `optimizer.step()` 使用 0 学习率；该次更新之后才把学习率推进到 warmup 第 1 步。若训练总步数太少导致取整后的 warmup 步数为 0，则第 0 步直接使用 `lr`。相关顺序见 `train.py:1082-1108`、`train.py:1181-1188` 与 `train.py:1683-1699`。

### 5.2 `weight_decay`

**严格定义：** `weight_decay=lambda` 是 AdamW 的解耦权重衰减系数。概念化地，在学习率为 `eta_t` 的一步中，参数先包含如下衰减：

\[
\theta\leftarrow(1-\eta_t\lambda)\theta,
\]

再叠加 Adam 的自适应梯度更新。AdamW 的 weight decay 与“把 `lambda/2·||theta||²` 加进损失后再交给 Adam”并不严格等价，所以源码 help 中的 “L2 regularization” 只能作为直观说明，严格名称应是 decoupled weight decay。

代码直接把 `model.parameters()` 放进单一参数组：

```python
optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
```

因此当前实现会对 optimizer 中所有可训练参数统一设置该衰减，包括卷积/线性权重、bias、BatchNorm/LayerNorm 的仿射参数；没有常见的“Norm 与 bias 免衰减”分组。默认值为 `0`，即关闭；HPO 在 `[1e-6,1e-3]` 上对数采样，见 `train.py:702`、`run_optuna.py:212`。

### 5.3 `lr_min`

**严格定义：** `lr_min=eta_min_arg` 是余弦衰减希望达到的最低学习率。scheduler 对它做两次约束：

\[
\eta_{min}=\min\bigl(\max(0,\eta_{min\_arg}),\eta_{base}\bigr).
\]

因此负值会变成 0，高于 `lr` 的值会变成 `lr`。默认值为 `0.0`，见 `train.py:703` 和 `train.py:1080-1081`。源码该行中文注释中的“最大学习率”是笔误；实际 help、变量名和 scheduler 运算都表明它是最低学习率。

### 5.4 `grad_clip`

**严格定义：** `grad_clip=G_max` 是所有模型参数梯度合并后的全局 2-范数上限。因为调用是

```python
nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

且没有指定 `norm_type`，所以采用 PyTorch 默认的 `2.0`。设全部参数梯度拼接后的范数为

\[
G=\left(\sum_j\lVert g_j\rVert_2^2\right)^{1/2},
\]

当 `G>G_max` 时，所有梯度按同一比例缩小，使总范数不超过 `G_max`；它不是把每个梯度元素分别截断到 `[-G_max,G_max]`。

代码只在 `grad_clip>0` 时启用，默认 `0.3`；非正值关闭。普通精度路径在 `loss.backward()` 后、`optimizer.step()` 前裁剪；AMP 路径会先执行 `scaler.unscale_(optimizer)`，再裁剪真实尺度的梯度，见 `train.py:1181-1208`。这一步不改变 forward loss，只改变过大梯度对应的参数更新。

## 6. 损失函数：L1、MSE 与 Huber

### 6.1 代码中的共同归约规则

`make_criterion()` 先创建 `reduction="none"` 的基础损失，再交给 `WeightedLoss`：

\[
L=\frac{1}{B}\sum_{i=1}^{B}w_i\,\ell(e_i),
\qquad e_i=\hat y_i-y_i.
\]

见 `train.py:944-962` 和 `train.py:993-1056`。当前默认 `weight_mode="none"`，所以 `w_i=1`。若启用其他权重模式，代码计算的是 `mean(w_i·loss_i)`，并不除以 `sum(w_i)`；因此权重的整体尺度也会改变 loss 与梯度尺度。

网络对每个样本输出一个标量，故此处通常是一项/样本。如果未来输出具有额外维度，基础损失会先保留那些维度；训练 wrapper 最终仍对全部元素取 `.mean()`。

### 6.2 L1 loss

命令行选择：`--loss l1`；实现：`nn.L1Loss(reduction="none")`。

\[
\ell_{L1}(e)=|e|.
\]

除 `e=0` 的不可微点由框架选择次梯度外，其梯度幅值基本不随误差大小增长，因此相较 MSE 不会让极大误差获得平方级影响；代价是零附近的梯度不如平方损失平滑。

### 6.3 MSE loss

命令行选择：`--loss mse`；实现：`nn.MSELoss(reduction="none")`；也是默认选择。

\[
\ell_{MSE}(e)=e^2,
\qquad
\frac{\partial\ell}{\partial\hat y}=2e.
\]

大误差会被平方放大，对离群样本更敏感，但零附近连续可微。

### 6.4 Huber loss

命令行选择：`--loss huber`；实现：`nn.HuberLoss(delta=huber_delta, reduction="none")`。默认 `huber_delta=1`，见 `train.py:753-754` 与 `train.py:1006-1007`。

PyTorch Huber 的严格定义为

\[
\ell_\delta(e)=
\begin{cases}
\frac{1}{2}e^2, & |e|\le\delta,\\
\delta\left(|e|-\frac{1}{2}\delta\right), & |e|>\delta.
\end{cases}
\]

对应梯度为

\[
\frac{\partial\ell_\delta}{\partial\hat y}=
\begin{cases}
e, & |e|\le\delta,\\
\delta\,\operatorname{sign}(e), & |e|>\delta.
\end{cases}
\]

所以它在小残差区是带 `1/2` 系数的二次损失，在大残差区转为线性增长，并把梯度幅值限制在 `delta`。`delta` 必须为正；当前 CLI 没有提前验证，非法值会在损失模块构造或使用时失败。

### 6.5 三种损失在本项目两种目标模式中的真实作用域

只写上述公式仍不足以描述本项目，因为误差 `e` 的定义由 `objective_mode` 改变。

#### `direct_sfr`

代码直接计算

\[
e_i=\hat y_i-y_i,
\]

其中 `y` 是 cache 中的 `target_sfr`。它通常是经过 cache metadata 声明的 SFR 变换后的值，例如 `target_transform.py` 支持 `log10_1p` 与 `log10_floor`。因此 direct 模式的 L1/MSE/Huber 一般位于**变换后的目标空间**，不能自动解释为物理 SFR 单位下的误差。实际变换应以当前 cache 的 `target_metadata` 为准，训练时会打印该元数据，见 `train.py:1548-1551`。

#### `signed_closure`

设网络输出为变换后的 closure `c_hat_t`，物理 baseline 为 `B`，closure 尺度为 `s`。若 cache 契约使用 `signed_asinh`，则

\[
\hat C=s\sinh(\hat c_t),
\qquad
\hat Y=B+\hat C.
\]

若使用 `identity`，则 `C_hat=c_hat_t`。但主损失不是 `ell(c_hat_t-c_t)`，而是

\[
L_{primary}
=\frac1B\sum_i w_i\,
\ell\left(\frac{\hat Y_i-Y_i}{s}\right).
\]

随后再加最终物理 SFR 为负的平方 hinge 惩罚：

\[
L_{neg}=\frac1B\sum_i
\left[\max\left(0,-\frac{\hat Y_i}{s}\right)\right]^2,
\]

\[
L_{total}=L_{primary}
+\texttt{negative\_sfr\_weight}\,L_{neg}.
\]

这段实现见 `train.py:86-175`。所以在 closure 模式下，L1/MSE/Huber 的残差是**由 baseline 与预测 closure 组成的最终物理 SFR，在除以 closure scale 后的残差**。代码保留负预测并用可微软惩罚约束，没有对输出硬裁剪。

## 7. 关键训练/网络过程

### 7.1 Fourier 位置编码

当前实现不是对输入物理场做 FFT，也不是 Fourier Neural Operator；它是把连续坐标映射成多频正弦/余弦特征，然后与物理场按通道拼接。

#### 第一步：半径归一化

SFRConvNet 创建编码器时保留默认 `r_log_scale=True`。对每个样本分别在其空间网格上计算：

\[
r_{safe}=\max(r,10^{-6}),
\]

\[
r_n=
\frac{\log r_{safe}-\min_{h,w}(\log r_{safe})}
{\max_{h,w}(\log r_{safe})-\min_{h,w}(\log r_{safe})+10^{-6}}.
\]

见 `models.py:245-255`。这是逐样本 min-max 归一化，不是用全训练集统一统计量。

#### 第二步：角度处理

`FourierPositionalEncoding2D` 类默认支持 `theta_pi_periodic=True`，但 `SFRConvNet` 实例化时明确传入 `theta_pi_periodic=False`，见 `models.py:667-673`。所以当前 CNN 实际执行

\[
\theta\leftarrow \theta\bmod 2\pi,
\qquad f_k^{(theta)}=f_k.
\]

见 `models.py:264-268`。不能用该类默认的 `theta mod pi` 分支解释当前 SFRConvNet。

#### 第三步：投影与正余弦展开

当前源码的精确公式为

\[
\phi_r(r_n)=
[\sin(2\pi f_k r_n),\cos(2\pi f_k r_n)]_{k=0}^{N_f-1},
\]

\[
\phi_\theta(\theta)=
[\sin(2 f_k\theta),\cos(2 f_k\theta)]_{k=0}^{N_f-1}.
\]

注意角向公式是 `2*f_k*theta`，不是 `2*pi*f_k*theta`；这是 `models.py:285-292` 的真实实现。由于 `theta` 自身以弧度表达，径向频带与角向频带不能简单当作相同物理单位的频率。

最终输出为

\[
pos=[r_n,\theta,\phi_r,\phi_\theta]
\in\mathbb{R}^{B\times(2+4N_f)\times H\times W},
\]

并与 `x` 做 `torch.cat([x,pos], dim=1)`，见 `models.py:294-300` 与 `models.py:733-738`。

#### 作用

卷积本身对平移共享权重，单靠局部物理场不直接知道网格点的绝对 `r/theta`。位置特征让后续卷积可以学习依赖绝对位置和不同空间尺度的映射。`pos_num_bands` 控制表示维度/频率采样密度，`pos_max_freq` 控制频带上端；两者增大都会增强高频表示机会，但前者还线性增加 stem 输入通道和计算量。

#### 这类编码在概念上究竟做了什么

普通卷积核在所有空间位置共享同一组参数。若两个局部窗口中的物理场数值完全相同，卷积本身倾向于给出相同的局部响应；它不会自动知道其中一个窗口位于星系中心附近，另一个位于外缘。把坐标作为额外输入后，同样的物理场模式可以因为位置不同而产生不同响应。这类做法可以视为 CoordConv 的多尺度版本。

只输入一个标量坐标也可以提供绝对位置，但后续网络若要表达随位置快速变化的函数，必须用多层非线性运算逐步合成。Fourier 映射预先提供

\[
\sin(\omega_k q),\quad \cos(\omega_k q)
\]

等一组不同尺度的基函数，使紧随其后的线性卷积可以直接组合出较复杂的位置依赖。低频分量缓慢变化，倾向于描述大尺度趋势；高频分量在相邻位置间变化更快，能够区分更细的位置结构。

它并不意味着网络在频域中执行卷积，也不会把物理场变换到 Fourier 空间。当前流程始终是：

\[
(r,\theta)
\xrightarrow{\text{确定性正余弦映射}}
pos
\xrightarrow{\text{通道拼接}}
[x;pos]
\xrightarrow{3\times3\ \text{stem 卷积}}
feature.
\]

`freq_bands` 通过 `register_buffer()` 注册，见 `models.py:218-221`。这表示它会随模型迁移设备并写入 `state_dict`，但不是 `nn.Parameter`，不会被 optimizer 更新。当前 Fourier 频率是固定超参数，不是训练得到的频率。

#### 完整实现过程与张量变化

以 `pos_num_bands=N_f` 为例，一个 batch 的实现可拆成以下步骤：

1. 输入 `r,theta` 的形状均为 `(B,1,H,W)`。
2. 对 `r` 做 `clamp → log → 每样本空间 min-max`，得到 `r_n`。
3. 对 `theta` 做 `remainder(theta,2*pi)`；当前 SFRConvNet 不走类默认的 `pi` 周期分支。
4. 把空间维临时展平为 `(B,1,H*W)`。
5. 把频带 reshape 为 `(1,N_f,1)`，利用广播得到 `(B,N_f,H*W)` 的相位。
6. 分别计算 `r_sin,r_cos,theta_sin,theta_cos`。
7. 拼接原始两通道与四组频率通道，恢复为 `(B,2+4N_f,H,W)`。
8. 与内容输入 `x` 在通道维拼接，得到 `(B,C_in+2+4N_f,H,W)`。
9. stem 的 3×3 卷积将其映射为 `(B,base_ch,H,W)`，后接 BatchNorm 和 ReLU。

位置编码自身没有可训练参数。位置特征如何参与预测，完全由 stem 及后续网络学到的卷积权重决定。网络既可以利用某些频带，也可以把无用通道对应的卷积权重学到接近零。

#### `sin` 与 `cos` 为什么必须成对出现

只使用 `sin(omega*q)` 会把某些不同坐标映射到相同值，并且相位平移不易表达。成对使用时，单个频率对应二维单位圆上的点：

\[
[\sin(\omega q),\cos(\omega q)].
\]

其内积满足

\[
\sin(\omega q_1)\sin(\omega q_2)
+\cos(\omega q_1)\cos(\omega q_2)
=\cos(\omega(q_1-q_2)),
\]

所以该频率下两个位置的相似度只依赖坐标差。多频率拼接后，模型同时得到多种尺度的位置差异信息。`include_input=True` 又保留了非周期的原始坐标，使网络不必仅靠周期函数辨认绝对位置。

不过，当前实现对半径保留的是 `r_n`，对角度保留的是取模后的弧度 `theta`，两者的数值尺度并不一致：`r_n` 约在 `[0,1]`，`theta` 在当前二维球极坐标数据中约位于 `[0,pi]`。拼接前没有独立的位置编码归一化层；统一的 BatchNorm 位于 stem 卷积之后。

#### 一个可以逐项手算的具体例子

下面构造一个极小的例子，并直接调用当前 `FourierPositionalEncoding2D` 核对数值。设：

```text
B = 1
H = W = 2
pos_num_bands = 2
pos_max_freq = 4
include_input = True
theta_pi_periodic = False   # 与当前 SFRConvNet 一致
```

输入坐标为

\[
r=
\begin{bmatrix}
1&10\\
100&1000
\end{bmatrix},
\qquad
\theta=
\begin{bmatrix}
0&\pi/4\\
\pi/2&3\pi/4
\end{bmatrix}.
\]

在代码中，它们的完整 tensor 形状都是 `(1,1,2,2)`；上式省略了 batch 和 channel 两个长度为 1 的维度。

##### 第 1 步：生成频带

代码执行

```python
torch.logspace(0, log10(4), steps=2)
```

因此

\[
f=[f_0,f_1]=[1,4].
\]

实际运行时第二项显示为约 `4.00000048`，来自默认 float32 频带缓冲区的浮点舍入，不改变下面的概念推导。

##### 第 2 步：半径取对数并逐样本归一化

先取自然对数：

\[
\log r=
\begin{bmatrix}
0&\log 10\\
2\log 10&3\log 10
\end{bmatrix}.
\]

该样本空间范围的最小值为 0，最大值为 `3 log(10)`，所以忽略分母中仅用于数值稳定的 `eps=1e-6` 时：

\[
r_n\approx
\begin{bmatrix}
0&1/3\\
2/3&1
\end{bmatrix}.
\]

严格按代码运行得到

\[
r_n=
\begin{bmatrix}
0&0.33333329\\
0.66666657&0.99999986
\end{bmatrix}.
\]

这里可以看到，对数归一化保留的是数量级位置：`1→10→100→1000` 在 `r_n` 上近似等间隔。

##### 第 3 步：处理角度

当前 SFRConvNet 采用 `theta_pi_periodic=False`，执行

\[
\theta\leftarrow\theta\bmod2\pi.
\]

本例所有角度已经位于 `[0,2pi)`，所以数值不变。随后临时展平：

\[
r_{flat}\approx[0,1/3,2/3,1],
\]

\[
\theta_{flat}=[0,\pi/4,\pi/2,3\pi/4].
\]

其形状从 `(1,1,2,2)` 变为 `(1,1,4)`；这只是为了让频带广播，不会改变像素顺序。

##### 第 4 步：计算径向 Fourier 特征

当前径向公式为

\[
r_{sin,k}=\sin(2\pi f_k r_n),
\qquad
r_{cos,k}=\cos(2\pi f_k r_n).
\]

以右上角像素 `r=10` 为例，`r_n≈1/3`：

\[
r_{proj}=[1\times1/3,4\times1/3]=[1/3,4/3].
\]

因此

\[
r_{sin}\approx
[\sin(2\pi/3),\sin(8\pi/3)]
=[0.8660,0.8660],
\]

\[
r_{cos}\approx
[\cos(2\pi/3),\cos(8\pi/3)]
=[-0.5,-0.5].
\]

##### 第 5 步：计算角向 Fourier 特征

当前角向代码不是 `sin(2*pi*f*theta)`，而是

\[
\theta_{sin,k}=\sin(2f_k\theta),
\qquad
\theta_{cos,k}=\cos(2f_k\theta).
\]

仍看右上角像素 `theta=pi/4`：

\[
\theta_{proj}=[1\times\pi/4,4\times\pi/4]
=[\pi/4,\pi].
\]

应用代码中的外层乘 2 后：

\[
\theta_{sin}
=[\sin(\pi/2),\sin(2\pi)]
=[1,0],
\]

\[
\theta_{cos}
=[\cos(\pi/2),\cos(2\pi)]
=[0,1].
\]

##### 第 6 步：按源码顺序拼成该像素的位置向量

`models.py:294-299` 的拼接顺序是：

```text
[r_flat,
 theta_flat,
 所有 r_sin 频带,
 所有 r_cos 频带,
 所有 theta_sin 频带,
 所有 theta_cos 频带]
```

所以右上角像素 `(r=10,theta=pi/4)` 的 10 维位置向量约为

\[
\begin{aligned}
pos_{右上}=[&0.33333329,\ 0.78539816,\\
            &0.86602556,\ 0.86602551,\\
            &-0.49999974,\ -0.49999981,\\
            &1.0,\ 0.00000075,\\
            &0.0,\ 1.0].
\end{aligned}
\]

其中极小的 `7.5e-7` 理论值为 0，偏差来自浮点表示。因为

\[
D_{pos}=2+4N_f=2+4\times2=10,
\]

整个位置编码 tensor 的形状是

\[
(B,D_{pos},H,W)=(1,10,2,2).
\]

这不是把四个像素变成四个 token；空间维仍然保持 `2×2`，只是每个像素从两个坐标数扩展成了 10 个位置特征通道。

##### 第 7 步：与物理场拼接并进入 CNN

假设本例物理输入有 3 个通道：

\[
x\in\mathbb{R}^{1\times3\times2\times2}.
\]

执行

```python
feat = torch.cat([x, pos], dim=1)
```

后得到

\[
feat\in\mathbb{R}^{1\times(3+10)\times2\times2}
=\mathbb{R}^{1\times13\times2\times2}.
\]

若 `base_ch=64`，stem 再执行 3×3 卷积：

\[
(1,13,2,2)\longrightarrow(1,64,2,2).
\]

卷积核此后可以联合使用物理通道和位置通道。例如，它可以学习“同样的高密度结构在小半径与大半径应产生不同贡献”，而不是把位置编码单独输出为预测。

在仓库当前的实际 `12×20` cache 中，`C_in=12`、默认 `N_f=32`，对应过程是

\[
x:(B,12,12,20),
\]

\[
pos:(B,130,12,20),
\]

\[
[x;pos]:(B,142,12,20),
\]

\[
stem([x;pos]):(B,64,12,20).
\]

##### 这个小例子如何直观展示混叠

本例归一化半径近似为 `{0,1/3,2/3,1}`。在这些有限采样点上，`f=1` 与 `f=4` 的径向正余弦恰好几乎相同，因为频率相差 3，而每个位置都是 `1/3` 的倍数。例如在 `r_n=1/3`：

\[
\sin(2\pi\times1\times1/3)
=\sin(2\pi\times4\times1/3).
\]

角向 `f=4` 也几乎退化：本例角度都是 `pi/4` 的倍数，而其相位是 `2×4×theta=8theta`，所以 `theta_sin` 在四个位置都约为 0，`theta_cos` 都约为 1。

这说明：连续函数中的两个不同频率，采样到有限网格后可能产生相同或近似相同的离散通道。增加高频并不保证增加新信息；它是否有用取决于频率与实际 `r/theta` 网格的共同关系。这正是前文建议联合调整 `pos_num_bands`、`pos_max_freq` 并检查位置特征秩的原因。

#### 对当前 cache 的有限网格分析

对仓库现有 cache 做只读检查得到：

| cache | `x` 形状 | 坐标形状 | 唯一 `r` 数 | 唯一 `theta` 数 | 坐标范围 |
|---|---:|---:|---:|---:|---|
| `Exp1_Rh8x8_Closure_auto_test.pt` | `(4873,12,8,8)` | `(4873,2,8,8)` | 8 | 8 | `r≈0.1819…298.0880`，`theta≈0.2269…2.9147` |
| `Exp1_Rh12x20_Closure_test.pt` | `(4873,12,12,20)` | `(4873,2,12,20)` | 20 | 12 | `r≈0.1286…406.5845`，`theta≈0.1629…2.9787` |

两个被检查的 cache 中，坐标网格在所有样本间完全相同。这与数据构造代码相符：`GalData.set_coord()` 从物理轴生成 `np.meshgrid(r,theta)`，`GalaxyDataset` 还会拒绝把不同物理坐标网格静默拼接，见 `galaxy_data.py:1002-1025` 与 `galaxy_data.py:1922-1930`。

这个事实对频带数有直接影响。对 `12×20` 网格：

- 所有径向编码通道只随 20 个 `r` 位置变化，因此径向通道矩阵的秩最多为 20；
- 所有角向编码通道只随 12 个 `theta` 位置变化，因此角向通道矩阵的秩最多为 12；
- 两组拼起来的空间模式秩最多为 32，实际还可能更低。

默认 `N_f=32` 会产生 `2+4×32=130` 个位置通道，所以在这个有限固定网格上，这 130 个通道不可能全部线性独立。对 `8×8` 网格，相应上界最多为 16。冗余通道仍可能改变优化路径或提供数值上更容易使用的基，但它们不会无限增加可辨认的位置数量。

进一步直接调用当前 `FourierPositionalEncoding2D`，把 `12×20` 网格上的输出整理成 `240×D_pos` 的 float64 矩阵并使用 `torch.linalg.matrix_rank` 检查，得到：

| `pos_num_bands` | `pos_max_freq` | `D_pos` | 当前网格数值秩 |
|---:|---:|---:|---:|
| 4 | 8 | 18 | 18 |
| 8 | 16 | 34 | 29 |
| 16 | 32 | 66 | 31 |
| 32 | 101 | 130 | 31 |

这组诊断不是模型精度实验，但它说明默认设置从 66 增长到 130 个位置通道时，在当前网格上没有增加线性独立的空间模式。是否仍有优化收益必须由完整训练验证，不能由秩单独决定。

频率上端也应结合网格密度理解。半径归一化后横跨约 `[0,1]`，`f=101` 的径向特征理论上在该区间振荡约 101 个周期，但径向仅采样 20 个位置；角向最高频特征的相位为 `2×101×theta`，在当前约 `0.16…2.98` 的角区间内同样经历远多于 12 个采样点能够平滑解析的振荡。严格的均匀采样 Nyquist 结论不能直接套到当前非均匀径向网格，但这仍清楚表明最高频通道会表现为欠采样/强混叠的离散位置特征，而不是平滑可解析的空间波。

这可能有两种相反效果：

- 正面：高频通道像固定随机特征一样，能让网络容易区分有限的离散网格点；
- 负面：相邻物理位置的编码可能剧烈跳变，削弱“物理上相邻的位置应有相似表示”的归纳偏置，并增加输入维度和过拟合机会。

因此不能只凭“频率越高、表达力越强”判断效果，必须用消融实验决定。

#### 当前角向公式需要先确认其物理意图

当前 SFRConvNet 设置 `theta_pi_periodic=False`，因此先把 `theta` 映射到 `[0,2pi)`；但后面的公共公式始终是

```python
t_sin = torch.sin(2 * t_proj)
t_cos = torch.cos(2 * t_proj)
```

于是当前最低角向基函数是 `sin(2theta),cos(2theta)`，而不是标准 2π 周期编码常用的 `sin(theta),cos(theta)`。如果启用 `theta_pi_periodic=True`，代码又会先令 `fb_theta=2*freq_bands`，最终相位成为 `4*f_k*theta`。这与源码注释“even harmonics: 2k*theta”的字面描述存在额外的二倍因子，见 `models.py:257-268` 和 `models.py:284-292`。

此外，频率是对数间隔的非整数。对一般非整数 `f_k`，`sin(2*f_k*theta)` 并不严格满足以 `2pi` 为周期：

\[
\sin(2f_k(\theta+2\pi))
\ne\sin(2f_k\theta)
\]

除非 `2f_k` 恰为整数。当前输入 `theta` 是球极坐标的 polar angle，而不是方位角 `phi`；`galaxy_data.py:1009-1013` 明确了这一语义，实际数据也位于 `[0,pi]` 内。因此在修改前应先回答物理问题：模型究竟需要保留南北方向差异、关于赤道反射对称，还是只需要平滑表示极角？不能仅依据“角度通常是周期变量”直接选择周期。

#### 7.1.1 为改善训练效果可调整的方向

以下方向按建议优先级排列，并明确区分“现有参数即可完成”和“需要修改模型代码”。这里的候选值是基于当前 `8×8`、`12×20` 有限网格提出的实验起点，不是未经验证的最优结论。

##### 方向一：联合搜索 `pos_num_bands` 与 `pos_max_freq`（最高优先级，现有模型参数即可）

当前 HPO v2 已将两者设置为独立 categorical 参数。默认完整 HPO 会连同其余六个参数一起采样；若要隔离检验 Fourier 参数，可使用 `--search-params pos_num_bands pos_max_freq`，让其余参数固定为 `train.py` 默认值。搜索空间版本与搜索子集都会进入 study/database 名称，避免把不兼容的 trial 混合。

建议首先做二维搜索，例如：

```text
pos_num_bands ∈ {4, 8, 16}
pos_max_freq ∈ {2, 4, 8, 16, 32, 101}
```

保留 101 是为了给现状提供对照，而不是预设它合理。若计算预算有限，可先固定 `pos_num_bands=8` 或 16，扫描 `pos_max_freq`；确定大致频率范围后再细化 band 数。

预期判断方式：

- 若较低 `pos_max_freq` 提高验证指标并缩小 train/val gap，说明原高频更可能造成混叠或过拟合；
- 若高 `pos_max_freq` 始终更好，说明模型确实在利用离散位置识别能力；
- 若减少 bands 几乎不影响结果，则默认位置通道存在不必要的冗余；
- 若增加 bands 只降低训练损失、不改善验证，则不应把它解释为有效表达力提升。

##### 方向二：先做“位置编码是否真正有用”的分层消融（需要很小的代码改动）

建议比较四组，而不是只比较不同 `pos_num_bands`：

1. 不输入坐标；
2. 只输入归一化后的原始 `r,theta`；
3. 原始坐标 + 低频 Fourier；
4. 当前默认的 32 bands、`max_freq=101`。

当前类没有显式 `encoding_mode`，`pos_num_bands=0` 也不是可靠的“关闭 Fourier”方式，因此最好增加明确开关。这个消融能回答三件不同的事：绝对坐标是否有益、Fourier 展开是否优于原始坐标、高频是否有额外贡献。

还可以在训练完成后进行推理期诊断：保持模型权重不变，分别把 `pos` 置零、打乱位置或只保留低频，观察验证指标下降量。它能衡量模型实际依赖哪些位置特征，但这种干预分布外诊断不能替代重新训练的正式消融。

##### 方向三：修正或重构角向编码（需要代码改动，优先于盲目扩大 HPO）

至少应把“角度预处理”和“相位倍率”拆成明确选项，例如：

- **保留方向的非周期 polar-angle 表示：** 输入归一化 `theta/pi`，并附加 `sin(theta),cos(theta)`；
- **假设 pi 周期：** 使用整数谐波 `sin(2k theta),cos(2k theta)`，避免当前非整数频率破坏严格周期性；
- **利用球坐标几何：** 使用 `mu=cos(theta)`，必要时再对 `mu∈[-1,1]` 做 Fourier/RBF 展开；
- **利用柱坐标几何：** 加入 `R=r sin(theta)`、`z=r cos(theta)` 或其规范化形式。

项目的体积和坐标变换代码本身大量使用 `sin(theta)`、`cos(theta)`，例如 `galaxy_data.py:430-454` 和 `coarsening_common.py:90-114`，所以 `cos(theta)` 或 `(R,z)` 不是凭空引入的新物理量。但究竟采用哪种表示，仍取决于目标是否应满足赤道反射对称以及输入场各分量的奇偶性。

修正时应为编码器增加单元测试，至少验证：输出维数、指定周期、`theta=0/pi/2/pi` 的期望值、不同 batch 上固定网格的一致性，以及 checkpoint 能否按原配置重建。

##### 方向四：减少位置通道在 stem 中的成本（需要代码改动）

当前 stem 的卷积权重数为

\[
N_{stem}=3\times3\times(C_{in}+2+4N_f)\times base\_ch.
\]

每增加 1 个 band 就增加 4 个输入通道，即增加 `36×base_ch` 个 stem 权重。以实际 cache 的 `C_in=12`、默认 `base_ch=64` 为例：

- `N_f=8`：位置通道 34，stem 权重 `9×46×64=26,496`；
- `N_f=16`：位置通道 66，stem 权重 `9×78×64=44,928`；
- `N_f=32`：位置通道 130，stem 权重 `9×142×64=81,792`；
- `N_f=64`：位置通道 258，stem 权重 `9×270×64=155,520`。

可增加独立的 `1×1` 位置投影，把 `D_pos` 先压到较小的 `pos_ch`，再与内容特征融合；或分别投影 content/position 后相加。仓库中的 Transformer 路径已经采用 `content_proj` 与 `pos_proj` 分支，见 `models.py:427-429`，可作为结构参考，但 CNN 中是否更好仍需独立验证。

另一种方式是为位置分支增加可学习标量或逐通道 gate：

\[
feature=Conv([x;\alpha\odot pos]).
\]

这样模型可以主动压低无用高频，而不是完全依赖 stem 卷积隐式做到这一点。

##### 方向五：统一坐标尺度与归一化契约（需要代码改动）

当前 `r_n` 位于约 `[0,1]`，保留的 `theta` 却约在 `[0,pi]`；径向正弦使用 `2pi*f*r_n`，角向正弦使用 `2*f*theta`。建议把“坐标归一化”“频率单位”“物理周期”写成同一个明确契约，例如都先映射到无量纲 `[0,1]`，再统一使用 `2pi*f*q`，或对 polar angle 使用单独的物理基。

对当前固定网格，逐样本半径 min-max 会得到完全相同的 `r_n`。但如果未来要跨不同径向范围或不同粗粒化网格泛化，逐样本归一化会抹掉绝对半径范围：两个物理范围不同的网格可能映射到相同的 `[0,1]`。此时应考虑：

- 使用训练集/cache metadata 固定的 `log(r)` 上下界；
- 同时保留规范化位置和绝对 `log(r)`；
- 把网格范围作为额外全局条件。

这项改变会影响模型输入契约，必须把归一化参数写入 checkpoint/hyperparameter metadata，避免训练与分析重建不一致。

##### 方向六：在多 stage 重新注入位置（较大结构改动）

当前位置编码只在 stem 前注入一次。经过多个残差块和 stride-2 下采样后，深层只能依靠早期特征携带位置信息。若实验发现深层对位置不敏感，可以在每个 stage 开始时：

- 对原位置特征做与当前分辨率一致的下采样后拼接；或
- 在各尺度重新计算低频位置编码；或
- 用 FiLM/gating 让位置调制 stage 特征。

这可能改善多尺度位置依赖，但会增加参数和混杂变量。应在完成“低频/高频/无编码”基础消融后再尝试。

##### 方向七：将频率改为可学习参数（较高风险的后续方向）

可以把 `freq_bands` 从 buffer 改为受约束的 `nn.Parameter`，或学习固定频率的乘性修正。优点是频率可以适应数据；风险是当前坐标网格固定且很小，频率容易退化成记忆网格点的编码，降低跨星系或跨网格泛化，并可能因高频梯度造成训练不稳定。

若尝试，应至少：

- 保证频率为正且有上界，例如用 softplus 后再 clamp；
- 从合理的低频 logspace 初始化，而不是从 101 上端开始；
- 对频率幅值或相邻频率间隔加正则；
- 分别报告同网格验证与跨网格测试结果。

它不应作为第一轮优化，因为现有固定编码的周期语义与频率范围尚未完成基本消融。

##### 方向八：位置编码缓存只属于速度优化，不应与精度优化混为一谈

由于当前 cache 的坐标网格在样本间固定，可以理论上只计算一次 `pos` 并按 batch 扩展。但仓库已经做过该候选实验：`benchmark_training.py:44-56` 将其标记为“仅用于重现已回退的固定位置编码候选”，`TRAINING_PERFORMANCE_AUDIT_20260908.md:38` 记录其稳定收益不足且存在坐标陈旧风险，因此没有进入正式训练路径。

缓存不会改变位置特征的数学内容，正常情况下也不会改善泛化，只可能改变运行时间和显存/内存行为。所以它不能替代频率、归一化与结构消融；若以后重新评估，必须保留严格的坐标一致性检查。

#### 7.1.2 建议的实验顺序与判据

为了避免同时改变太多因素，建议按以下顺序：

1. **复现实验基线：** `N_f=32,f_max=101`，记录多个随机种子的结果。
2. **频率上限扫描：** 固定 `N_f=8` 或 16，扫描 `f_max∈{2,4,8,16,32,101}`。
3. **频带数扫描：** 在上一步较好的 `f_max` 附近扫描 `N_f∈{4,8,16}`；`N_f=32` 可作为旧默认基线单独复现，不进入当前 v2 HPO 搜索空间。
4. **编码消融：** 无坐标、原始坐标、低频 Fourier、当前 Fourier。
5. **角向语义实验：** 当前 `2f theta`、标准化 polar angle、`cos(theta)`/几何坐标；先确认物理对称性。
6. **结构实验：** 位置投影/gate；只有前面显示深层位置表达不足时再做多 stage 注入。
7. **最后才考虑可学习频率。**

比较时至少保持数据划分、`base_ch`、`stage_depths`、loss、batch size、学习率日程和训练 epoch 不变，并使用多个随机种子。当前验证主指标是各 galaxy group 损失的算术平均，而不是旧注释所说的 worst-domain loss，见 `train.py:1371-1405`。在 `signed_closure` 模式下还应同时观察：

- `val/final_sfr_physical_mae`；
- `val/final_sfr_negative_fraction`；
- train/validation gap；
- 各 `val/loss_domain_i`，避免总体均值改善但某一星系显著退化。

除最终指标外，可以增加两类诊断：

- **相邻位置连续性：** 计算相邻网格点位置向量的距离，检查高频设置是否让物理邻点出现异常大的编码跳变；
- **位置特征有效秩/相关性：** 对固定网格上的 `pos` 展平后做 SVD 或通道相关矩阵，量化增加 bands 后究竟增加了多少独立模式。

最终选择不应只看单次 trial 的最低 loss。更稳妥的准则是：多种子验证均值改善、方差可接受、各星系没有系统性退化，并且参数/显存成本与收益相称。

### 7.2 SE 通道注意力机制

当前通道注意力是 Squeeze-and-Excitation，不是空间注意力，也不是 Transformer self-attention。对 block 输出

\[
X\in\mathbb{R}^{B\times C\times H\times W},
\]

先做全局平均池化（Squeeze）：

\[
z_{bc}=\frac{1}{HW}\sum_{h,w}X_{bchw}.
\]

然后通过两层 MLP 生成通道权重（Excitation）：

\[
a_b=\sigma\left(W_2\operatorname{SiLU}(W_1z_b+b_1)+b_2\right),
\qquad a_b\in(0,1)^C.
\]

最后在空间维广播并重标定：

\[
Y_{bchw}=X_{bchw}a_{bc}.
\]

见 `models.py:503-521`。它为每个样本、每个通道产生一个标量权重，同一通道内所有空间位置共享该权重。SE 位于第二个卷积之后、残差相加之前，见 `models.py:573-576`。

### 7.3 余弦退火（含 warmup）

当前不是按 epoch 更新的 `CosineAnnealingLR`，而是自定义的、按**每个 optimizer step** 更新的 warmup + cosine scheduler。

令：

- `T = len(train_loader) × epochs` 为总优化步数；
- `W` 为 warmup 步数；若显式 `lr_warmup_steps=0` 且 `lr_warmup_ratio>0`，则 `W=int(T×ratio)`；
- `eta_base=lr`；
- `eta_min` 为约束后的 `lr_min`；
- `s` 为 `lr_warmup_start_factor`，被限制到 `[0,1]`；
- `t` 为 scheduler 内部步编号，范围 `[0,T]`。

warmup 段为线性增长：

\[
\eta(t)=\eta_{base}
\left[s+(1-s)\frac{t}{W}\right],
\qquad 0\le t\le W.
\]

cosine 段为

\[
p(t)=\operatorname{clip}\left(\frac{t-W}{T-W},0,1\right),
\]

\[
\eta(t)=\eta_{min}
+(\eta_{base}-\eta_{min})
\frac{1+\cos(\pi p(t))}{2},
\qquad W<t\le T.
\]

见 `train.py:1062-1108`。scheduler 在 optimizer 更新之后调用 `scheduler.step()`，因此某一步更新使用的是调用前已经设置好的学习率。训练中途 early stopping 时，退火也会随之提前结束，并不保证实际走到 `lr_min`。

### 7.4 池化

**严格定义：** 池化把一个空间邻域或整个特征图归约成更少的值。当前 SFRConvNet 只使用自适应全局平均池化 `AdaptiveAvgPool2d(1)`，没有 `MaxPool2d`。

它出现于两个位置：

1. 每个 SEBlock 内，把 `(B,C,H,W)` 变成 `(B,C,1,1)`，用于估计通道权重；
2. 回归头开头，把最终特征图 `(B,C_last,H_last,W_last)` 变成 `(B,C_last,1,1)`。

当输出大小为 `1×1` 时，其结果就是每通道对所有空间位置求平均：

\[
y_{bc11}=\frac1{H_{last}W_{last}}\sum_{h,w}x_{bchw}.
\]

见 `models.py:509-520` 与 `models.py:715-718`。池化本身没有可学习参数，并使回归头能够接受不同空间尺寸的特征图。再次强调：stage 间的分辨率减半来自 stride-2 卷积；这里的全局池化只在 SE 汇总和最终回归汇总中使用。

### 7.5 Flatten

`nn.Flatten()` 按默认参数从维度 1 开始把后续维度合并，保留 batch 维。在回归头中：

\[
(B,C_{last},1,1)\longrightarrow(B,C_{last}).
\]

见 `models.py:717-720`。它只改变 tensor 形状/视图，不执行统计汇总、没有参数，也不会混合不同样本；真正消除空间信息的是它前面的全局平均池化。

项目中还有一种应与 `nn.Flatten()` 区分的临时展平：Fourier 编码用 `.view(B,1,H*W)` 把空间网格暂时排成序列以便广播频带，编码完成后又恢复成 `(B,D_pos,H,W)`，见 `models.py:278-299`。这一步没有永久丢弃二维空间对应关系。

## 8. 参数之间的直接耦合关系

| 参数/过程 | 在当前代码中的直接耦合 |
|---|---|
| `base_ch` | 决定所有 stage 通道、SE/FiLM 输出宽度和回归头宽度 |
| `stage_depths` | 决定 block 总数、stage 数、下采样次数和最后通道数 |
| `stride` | stage 边界取 2，直接缩小空间尺寸；skip 分支同步投影 |
| `pos_num_bands` | 决定 `D_pos=2+4N_f`，从而改变 stem 输入通道数 |
| `pos_max_freq` | 改变对数频带上端，不改变位置编码通道数 |
| `p_drop` | 同时控制 block 的通道 dropout 与 head 的元素 dropout |
| `se_reduction` | 通过 `max(C//r,8)` 改变每个 SE MLP 的中间宽度 |
| `batch_size` | 改变每步样本数、BatchNorm 统计、每 epoch 步数及 scheduler 总步数 |
| `num_workers`、`prefetch_factor` | 只改变标准数据路径的供数并行/预取，不改变数学目标 |
| `lr`、`lr_min` | 分别给出 warmup 峰值和 cosine 末端；实际逐步学习率由 scheduler 产生 |
| `weight_decay` | 每一步按当前实际学习率做 AdamW 解耦参数衰减 |
| `grad_clip` | backward 后、optimizer 更新前约束全局梯度 2-范数 |
| L1/MSE/Huber | 先逐样本计算，再乘可选样本权重并取算术平均；误差空间由 objective mode 决定 |

## 9. 默认值与 HPO 状态速查

| 名称 | `train.py` 默认值/状态 | `run_optuna.py` 当前搜索状态 |
|---|---:|---|
| `base_ch` | `64` | `[32,64,128]` |
| `batch_size` | `64` | `[64,128,256,512,1024]` |
| BatchNorm | 固定启用 | 不搜索 |
| stride | stage 0 为 1；后续 stage 首 block 为 2 | 不搜索 |
| `weight_decay` | `0` | `1e-6` 至 `1e-3`，对数采样 |
| `stage_depths` | `(2,2,2,2)` | 不搜索 |
| `pos_num_bands` | `32` | `[4,8,16]` |
| `pos_max_freq` | `101` | `[2,4,8,16,32,101]` |
| `p_drop` | `0.3` | `0.0` 至 `0.5`，步长 `0.1` |
| `se_reduction` | `4` | `[4,8,16]` |
| `num_workers` | Windows `0`；其他系统至多 `16` | 由 HPO 启动参数固定，不由 trial 搜索 |
| `prefetch_factor` | `3` | 不搜索 |
| `lr` | `1e-3` | `1e-5` 至 `1e-2`，对数采样 |
| `lr_min` | `0.0` | 不搜索 |
| `grad_clip` | `0.3` | 不搜索 |
| `loss` | `mse` | 由 HPO 启动参数固定，不由 trial 搜索 |
| `huber_delta` | `1` | 不搜索 |

这张表描述的是当前源码默认/搜索空间，不代表某次已经运行实验的最终参数；单次命令行、JSON 配方或 Optuna trial 都可能覆盖部分默认值。

表中列为“搜索”的八个参数均可通过 `run_optuna.py --search-params ...` 人为选择。
省略该选项或传入 `all` 时搜索全部八项；被排除的参数固定使用表中的 `train.py`
默认值，而不是沿用某个历史 trial 的最佳值。也可通过
`--fixed-params 参数名=值` 覆盖任意未搜索参数；同一参数不能同时搜索和固定。

对于 `pos_num_bands×pos_max_freq` 的完整离散实验，可使用 `--sampler grid` 保证18种
组合各运行一次。用 `--seeds 42 43 44` 时，seed作为重复实验维度将网格扩展为54个
格点；不应把验证结果最好的seed解释为可优化模型参数。默认 `tpe` 仍允许重复采样，
不会因18种理论组合存在而自动切换成穷举。
