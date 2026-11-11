# train_sfr_unet.py：逐格训练核心（Exp1/Exp2）

本脚本训练独立 `SFRUNet`，**固定为 signed-closure、物理空间逐格 MSE**。
原 `train.py/models.py` 的标量 Direct/Closure 流程不变。当前不接受 Exp3 或其他实验策略。
先阅读 [SFRUNet.md](SFRUNet.md) 和研究计划 2026-11-03 记录。

## 数据要求

每个单实验/单折需要原 cache：

```text
<data_exp>metadata.json
<data_exp>norm.npz
<data_exp>train.pt
<data_exp>val.pt
<data_exp>test.pt
<data_exp>out.pt       # 仅 Exp2
```

以及 [build_cell_supervision_cache.py](build_cell_supervision_cache.md) 生成的附加文件：

```text
<cell_exp>train_cells.pt
<cell_exp>val_cells.pt
<cell_exp>test_cells.pt
<cell_exp>out_cells.pt # 仅 Exp2
```

`cell_exp` 默认等于 `data_exp`，附加文件目录默认等于 `cache_dir`。文件名中的前缀应含最后的下划线。
**旧 scalar cache 单独不够，需要逐格 sidecar；不必重新 split 或标准化。**
原 scalar cache 可以是 Direct 或 Closure，但必须包含完整物理 Y/B/C。
新模型的 objective 不随原 cache 的 prepared_objective 改变，始终为逐格 signed_closure。

入口检查实际运行名、metadata 声明、UID/parent 跨 split 隔离、地图摘要与标量总和、
通道/形状/输入变换，以及 norm.npz 的 mean/std。norm.npz 随 checkpoint 保存；不重新估计输入统计。
仅接受 `exp1/random_snapshot_within_group` 或 `exp2/leave_one_galaxy_out`。
Exp2 的 train/val/test 运行集合必须等于 train_groups，out 必须是唯一 held_out_group。
训练按已有 cache 随机打乱，保留 cache 内过采样重复项；不叠加 run_time_balanced，不启用 mirror。

## 损失和固定选模指标

每格 Y/B 的单位为 code SFR，已经乘过体积。`C_hat=s_C*sinh(z)`，`Y_hat=B+C_hat`；
总 SFR 直接对格子求和，不裁剪负值。训练损失为：

\[
L=L_{cell}+\lambda_{total}L_{total}+\lambda_-L_-.
\]

三项定义见 SFRUNet.md。逐格与总量损失用训练 Y 的对应 RMS 缩放；s_C 用训练非零 |Y-B| 中位数。
三个尺度只估计一次，包含训练 cache 的重复样本，验证/测试复用。

早停、best checkpoint 与 HPO 使用同一固定指标：先在各运行内部对 snapshot 平均
`Lcell+Ltotal+Lnegative`（系数固定为1），再对运行等权平均。训练时改变 lambda 不改变此评价尺子。
指标不平均 batch 均值，每帧贡献正确归约。物理 MAE/RMSE 和 baseline 指标同时按运行/整体记录。
HPO 不读取 test/out；正式训练在开始前检查它们的隔离，完成后仅用 best checkpoint 评估。

## 命令

```powershell
python Experiment/train_sfr_unet.py --data-exp Exp1_Rh12x20_FiveRuns_Closure_ --cell-exp Exp1_Rh12x20_FiveRuns_Cell_ --epochs 100 --device cuda
```

示例前缀需替换为实际已生成的文件名；脚本不会自动生成 sidecar。
完整参数：`python Experiment/train_sfr_unet.py --help`。

## 可变参数与默认值

| 参数 | 默认 | 含义/选项 |
|---|---|---|
| `--config` | 无 | 独立 JSON，字段名使用下划线；CLI 覆盖 JSON，JSON 覆盖默认值 |
| `--data-exp` | 必填 | 原 cache 单实验/单折前缀 |
| `--cache-dir` | `.cache` | 原 cache、metadata、norm 所在目录 |
| `--cell-exp` | 同 data_exp | 逐格 sidecar 前缀 |
| `--cell-cache-dir` | 同 cache_dir | 逐格文件目录 |
| `--output-root` | `Results` | 新建独立结果目录；同名实验也不会覆盖旧结果 |
| `--run-name` | 自动 | 输出名称前缀，自动追加北京时间、UUID |
| `--seed` | 42 | 固定训练随机种子，不作为 HPO 参数 |
| `--device` | `auto` | `auto/cpu/cuda/cuda:0` 等 PyTorch 设备名 |
| `--cpu-threads` | 4 | CPU tensor 线程数，适用于8核实例的起始配置 |
| `--epochs` | 100 | 预定总轮数，同时决定学习率日程 |
| `--patience` | 20 | 验证指标无改善轮数，0关闭早停 |
| `--batch-size` | 64 | batch大小；训练尾批仅一帧时合入前批，不丢样本 |
| `--lr` | 0.0003 | AdamW 峰值学习率，正数 |
| `--weight-decay` | 0.00001 | AdamW 权重衰减，非负 |
| `--warmup-ratio` | 0.1 | 总轮数中的 warmup 比例，向上取整；其后余弦衰减 |
| `--min-lr-ratio` | 0.01 | 最小学习率/峰值学习率 |
| `--grad-clip` | 1.0 | 梯度范数上限；非有限梯度报错，不跳过隐藏 |
| `--base-ch` | 32 | 三尺度通道 c、2c、4c |
| `--encoder-depths` | `2 2` | 两个编码阶段的残差块数 |
| `--decoder-depths` | `2 2` | 从粗到细两个解码阶段的残差块数 |
| `--bottleneck-depth` | 2 | 瓶颈残差块数，不是下采样次数 |
| `--pos-num-bands` | 4 | Fourier bands 正整数 |
| `--pos-max-freq` | 4 | 最高频率，≥1 |
| `--p-drop` | 0 | Dropout2D，范围[0,1) |
| `--se-reduction` | 4 | SE 压缩比，正整数 |
| `--total-weight` | 1 | 训练总量损失系数，非负 |
| `--negative-weight` | 1 | 训练负 SFR 惩罚系数，非负 |
| `--amp` | `none` | `none/bfloat16`；BF16 仅适用于支持它的 CUDA 设备 |
| `--tensorboard / --no-tensorboard` | 开启 | 写标量，不写模型图；HPO 强制关闭逐 trial TensorBoard |
| `--resume` | 无 | 本入口生成的 `last.pt`，不是旧标量 checkpoint |
| `--stop-after-epochs` | 0 | 本次最多执行几轮后暂停；0不额外限制，不改变预定总轮数 |

网络 AMP 只覆盖网络前向，sinh/物理组合/损失在外部执行并保留 float64 标签精度。
CPU 和 CUDA 的 FP32、BF16 smoke 已通过；尚未在服务器 A100 测吞吐量，不保证 AMP 必然更快。
默认整批 CPU tensor 索引，无 DataLoader worker，减少小网格的逐样本 Python 开销。

## 输出和恢复

结果位于 `<output-root>/<run-name>_<时间>_<UUID>/`：

- `config.json`：完整实际配置，可再用于启动器。
- `contract.json`：模型、损失、尺度、输入统计、cache SHA256、源实现摘要、实验 metadata。
- `split_manifest.json`：各 split 的 UID/parent UID。
- `history.json`：每轮三项训练损失、固定 val 指标、各运行物理误差、学习率、耗时。
- `best.pt`：按固定验证指标选择的最佳 checkpoint。
- `last.pt`：最后一个完整 epoch；含模型、优化器、scheduler、随机数状态、早停状态、历史。
- `summary.json`：完成或暂停状态、最佳轮次及正式 test/out 指标。
- `test_predictions.pt`、Exp2 的 `out_predictions.pt`：最佳模型的物理地图、总量、Y/B/C_hat、UID、时间和运行名。
- `tensorboard/`：可选训练/验证曲线。

预测文件地图为 N1HW；保存 code SFR，乘 0.025 转 Msun/yr。不应直接交给原标量分析 notebook；
后续空间分析可读取其地图和总量。总量可用于瞬时历史曲线，累计量需按真实时间排序积分。

```powershell
# 最初计划100轮，本次先完成5轮后暂停。
python Experiment/train_sfr_unet.py --data-exp PREFIX_ --epochs 100 --stop-after-epochs 5
# 不改变预定epochs，恢复其余训练；替换实际目录。
python Experiment/train_sfr_unet.py --config Results/RUN/config.json --resume Results/RUN/last.pt --stop-after-epochs 0
```

恢复必须携带同目录 best.pt，模型/损失/日程/数据/实现契约必须一致；允许改路径、设备及暂停长度。
Ctrl+C 后可从已写好的 last.pt 恢复，未完成 epoch 会重新执行。CPU 小测试已验证逐参数一致；
不保证跨硬件 CUDA 逐位一致。更改 epochs 会改变余弦日程，因此不允许通过 resume 延长原实验。
输入文件只读；写文件使用同目录临时文件后原子替换。


## 2026-11-06：训练内非负输出与候选辅助监督

原模式默认保留：`--output-constraint none --candidate-weight 0`。
新增 `--output-constraint relu --candidate-weight 1`（1 是初始对照值，尚未正式优化）：

```text
Delta = closure_scale * sinh(z)
q = B + Delta
Y_hat = ReLU(q)
C_hat_final = Y_hat - B
L = mean(((Y_hat-Y)/cell_scale)^2)
    + total_weight * mean(((sum(Y_hat)-sum(Y))/total_scale)^2)
    + candidate_weight * mean(((q-Y)/cell_scale)^2)
```

全部格子（含零标签）参与辅助监督。它绕过 ReLU 的负区间，给 q<0、Y>0 的错误预测提供梯度；
不是修改 Y/B/C_true 标签，也不是对旧 checkpoint 后处理裁剪。最终 closure 仍可为负，且不小于 -B。
ReLU 模式要求 candidate_weight 为有限正数；none 要求它为0。候选及所有损失必须有限，
不允许 ReLU 将负无穷隐藏为零。辅助项使用已有训练集 cell_scale，不再估计测试集尺度。

`negative_weight` 仍保存在配置中供旧模式使用，但 relu 下有效权重固定为0；启动日志和 loss 契约
明确报告有效值。逐格与总量主损失都使用最终 Y_hat，候选辅助项直接使用 q。
选模指标仍为各运行等权的 `cell + total + negative`（系数固定1）；candidate loss 不参与该指标，
避免搜索辅助权重时改变评分口径。非负模式下 negative 项自然为0。

无需重建已有 scalar cache/逐格 sidecar；必须重新训练，使用独立结果目录。
模型卷积结构和参数数量不变；本次只扩展物理组成与损失。原标量CNN/Direct入口不变。

### 参数与启动方式

- `output_constraint` / `--output-constraint`：默认 `none`；可选 `none, relu`。
- `candidate_weight` / `--candidate-weight`：默认0；relu时显式设置正数，例如1、0.3、0.1。
- 不使用 Softplus，不需要 epsilon，没有正值背景。

```powershell
# 用上次最优配置固定其他条件，重新训练一个非负对照；将路径替换为实际 best_config.json
python Experiment/run_train_sfr_unet.py --config .cache/STUDY/best_config.json --output-constraint relu --candidate-weight 1 --run-name SFRUNet_ReLU_Candidate1 --device cuda

# 或同一个 Exp2 cache 新开 HPO（默认仍搜索原六个参数，辅助权重固定1）
python Experiment/run_optuna_sfr_unet.py --data-exp Exp2_Rh12x20_FiveRuns_Closure_v7_LOO_disk_galaxy_fiducial_ --output-constraint relu --candidate-weight 1 --device cuda --n-trials 100
```

若显式搜索辅助权重，在 `--search-params` 的完整列表中加入 `candidate_weight`，范围为 `{0.1,0.3,1}`；
`--search-params` 会替换默认列表。该搜索仅允许 relu，relu 禁止搜索已经停用的 negative_weight。
输出模式与权重进入配置/恢复/HPO身份；新实验不会续接旧study。旧JSON缺少新字段时默认旧模式。
同版本暂停恢复可以继续；**跨本次代码修改的旧 last.pt 仍会因严格代码/配置契约检查而拒绝直接续训**。
旧模型权重结构和旧预测分析仍兼容，不允许靠更改旧checkpoint的模式字段将其冒充重训结果。

### 导出与分析

新 loss contract 为 v2，新 prediction schema 为 `macnet_sfr_unet_predictions_v2`，包含：
- `prediction_map`：最终 Y_hat；`closure_map`：最终 Y_hat−B。
- `candidate_map`：投影前 q；`candidate_closure_map`：投影前 Delta，不能冒充最终 closure。
- `output_constraint` 和完整 `loss_contract`。

history.train_loss 保留前4项（总损失、cell、global、negative），追加第5项原始 candidate loss；
TensorBoard `train/candidate` 为未乘权重的辅助损失，`train/total` 已包含其加权贡献。
分析器同时支持旧v1与新v2；v2额外检查 q=B+Delta、Y_hat=ReLU(q)/q、非负性以及与训练loss契约一致。
现有 Results_stat_Exp2_SFRUNet.ipynb 无需改结构，切换 RESULT_DIR 即可分析新结果。


## 2026-11-12：真实零格子L1

新增 `--zero-l1-weight`（默认0，有限非负）；真实零格子最终SFR的归一化L1，保留现有功能。
训练、日志、loss contract v3与HPO身份均保存该值；默认搜索列表不变，可选搜索范围为0/0.01/0.03/0.1/0.3/1。
已有cache可复用，需从头进行对照训练；完整公式、Fiducial多种子命令与分析见[对照指南](zero_l1_comparison.md)。
