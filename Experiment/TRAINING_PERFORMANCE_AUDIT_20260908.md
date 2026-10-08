# MACNet Rh 8×8 训练性能优化审计（2026-09-08）

## 1. 目标、边界与结论

在已有 `Exp1_Rh8x8_Closure_auto_*.pt` 上，按“固定基准 → 单项候选 → 小训练 → 核对数值与计时 → 保留/撤回 → 最终组合”实施。
不是重新调参，不改变科学任务，不重新生成数据，不运行正式 HPO，也不修改已有 Optuna 数据库。

最终保留：验证统计向量化、CPU 整批索引、同进程 CPU dataset 复用、HPO 跳过测试集加载与重复模型图。
撤回或不启用：逐 batch loss 的设备端累计、生产入口的 GPU 常驻、强制 CPU 线程数、增加 worker、位置编码缓存。
模型架构和参数没有改变，signed closure 的物理组合与优化目标没有改变。

## 2. 固定基准

- 本地：Windows，RTX 4060 Laptop GPU 8 GB；PyTorch 2.5.1 + CUDA 12.1。
- 基础模型：SFRConvNet，12 通道，8×8，base_ch=32，stage_depths=2,2,2,2，32 个位置频带，dropout=0。
- signed closure，signed_asinh，cache 中的 scale=226.31541789318408；L1，negative_sfr_weight=1。
- AdamW，lr=5e-4，weight_decay=1e-5，FP16 AMP，grad_clip=0.3。
- 微基准：固定 1539 个训练样本、769 个验证样本；固定权重种子与 batch 索引；尾批保留。
- 最终微基准：每次 8 个 epoch，4 次交替顺序重复，排除每次首轮 warm-up。
- 计时在 epoch 边界同步 CUDA；同时保存验证指标、权重最大差和峰值 CUDA 分配量。
- 微基准不含 TensorBoard 文件写入；完整 `train.main` 基准包含读取、校验、模型建立、优化器、scheduler、TensorBoard 和日志关闭。
- 微基准大部分限制 CPU 算子线程数为 8，便于固定环境；这不是更改正式代码默认线程数，也不等价于复刻服务器 8 vCPU。

训练集文件约 181 MB，验证/测试集各约 60 MB（磁盘文件大小，并非驻留显存）。
不能直接把本地加速比外推到 Linux/A100，尤其 Windows worker spawn 开销差异很大。

## 3. 八项逐项决策

| 顺序 | 项目 | 实验和判断 | 最终处理 |
|---|---|---|---|
| 1 | 固定性能基准 | 小子集、完整 split、交替重复、CUDA 边界同步；发现并排除明显运行状态漂移 | 保留两个 benchmark 脚本与原始 JSON |
| 2 | 减少训练 loss `.item()` 同步 | 改成设备端 float64 累积；组合内复测总耗时收益约 1%，波动可大于收益 | 撤回，恢复原训练累计方式 |
| 3 | 验证统计向量化 | 在设备端 scatter_add 分星系统计，轮末统一读回；不再每个 batch 对每个星系同步 | 保留；保持星系等权均值而非样本加权总均值 |
| 4 | 整批 tensor 索引 | 一组 batch 索引同时提取全部字段，减少逐样本 `__getitem__` 与 collate | 保留；Optuna 默认 tensor_cpu，单次 train 默认 standard |
| 5 | 数据 GPU 常驻 | 先在早期噪声阶段表现更快，稳定阶段不再可靠优于 CPU 整批索引 | 不加入正式入口；仅 benchmark 可重现此候选 |
| 6 | CPU threads / worker | 测 1/2/4/8 线程及普通路径 2/8/24 线程；测 0/1/2/4 worker | 不强制线程数，不增加默认 worker；另修复 worker 数被除以二的 bug |
| 7 | trial 启动和重复工作 | 跳过模型图、实际不加载 test、复用已校验 CPU train/val；用完整 main 分开测量 | 保留；模型图开关保持旧 seed/取样消耗，标量仍记录 |
| 8 | 固定网格位置编码缓存 | 固定坐标验证、缓存编码、遇不同坐标回退；守卫也有开销，稳定收益不足 | 从模型与正式训练接口撤回；不引入坐标陈旧风险 |

第 6 项：Windows 下 1/2/4 worker 的首轮 train 和 val 各额外花约 3–4 / 7 / 13–14 秒来启动进程。
稳态没有可靠优于 0 worker，不能因为 worker 更多就认为更快。正式代码中的 `num_workers=1` 现在确实使用 1 个 worker。

## 4. 最终有效计时

以下只使用修正过验证对照的 `13_corrected_final_b*.json`。
“单轮”是 train+val 总耗时：先对每次运行的非首轮 epoch 取中位数，再对 4 次运行取中位数。

| Batch size | 原版单轮（秒） | 最终组合单轮（秒） | 耗时减少 |
|---|---:|---:|---:|
| 64 | 0.4285 | 0.3938 | 8.1% |
| 128 | 0.2482 | 0.2123 | 14.5% |
| 512 | 0.1287 | 0.0978 | 24.0% |

这些是在相同 batch size 内的对比，不意味着应该通过改 batch size 获得不公平的“同一训练”加速。
所有最终微基准对照权重最大绝对差为 0；逐轮验证指标最大差约 1.02×10^-7，属于归约精度/顺序差异。
这不等于承诺任意硬件、所有超参数及任意长训练都逐 bit 一致。

### 完整 train/val split 补充检查

使用完整 14615/4871 个样本，每次训练 2 轮，batch=128；3 次交替对照。
来源：`11_final_full_split.json`。不加载正式测试集做性能评估。

- 排除首次 CUDA 冷启动及第一次 dataset cache 填充后，原入口总耗时约 5.66 秒，新入口约 3.74 秒，减少约 34%。
- 其中两轮实际 train+val 原版约 4.20 秒，新版约 3.70 秒，稳态计算耗时减少约 12%。
- 其余主要是启动/加载/日志等开销：原版约 1.47 秒，新版缓存命中后约 0.032 秒。
- 完整 split 权重最大差仍为 0；最佳验证指标相差约 4.6×10^-10。

**不能把两轮任务的 34% 总耗时改善直接外推到 90 轮 HPO。** 长 trial 会摊薄固定启动收益；大的 base_ch、不同 batch、服务器 IO/CPU/GPU 都会改变比例。

启动候选拆分实验 `07_startup.json` 表明：不写模型图使启动/日志开销约从 1.26 秒降到 0.82 秒；
复用命中后约 0.03–0.04 秒。首次复用仍需要读盘，不能宣称每个 trial 从第一次开始都免加载。

## 5. 对照脚本审计与无效数据标记

最初微基准独立导入旧版 train 模块，却用新版模块建立 WeightedLoss。
Python 中两者的类身份不同，旧版验证不能识别它为逐样本加权损失，closure 分支随后通过广播掩盖了这个问题；direct 分支则报错。

已修正为“从验证函数所属模块建立 criterion”，并增加逐 epoch metric 一致性断言。
`01/02/03/05/10` 中跨旧版验证模块的微基准，不作为最终速度或验证精度证据；其权重比较也不能独自证明验证正确。
`04/06/08/09` 使用共同的新版验证函数的对照不受此类身份问题影响。
`07/11` 通过各自完整 main 建立自己的 criterion，也不受影响。
最终结论引用 `13` 修正后的微基准；`14` 为修正后的单项候选重测。
所有旧 JSON 留存，以保留审计轨迹，未覆盖成看似一次就成功的实验记录。

## 6. 可靠性与物理边界

- 批次字段始终使用同一索引，direct/closure 接口、baseline、target、分组不允许错位。
- 对常规 DataLoader 检查连续多轮 RNG 状态、样本顺序和尾批完全一致。
- 不改 `Y=B+C`，不把 baseline 当成额外网络输入；不更改 closure scale、transform、负 SFR 惩罚、星系权重和样本划分。
- 检查 direct 的 L1/MSE/Huber 逐样本验证、closure 的不均衡星系与缺失组、空验证集。
- dataset 复用是 CPU 只读共享；对象浅拷贝隔离增强器，文件属性变化后失效，缓存容量有上限。
- 不支持在线增强的 tensor 路径会报错，不能偷偷绕过增强。
- HPO 不反序列化 test；正式训练仍加载和测试最佳权重，Exp2 正式路径仍完整验证 out 分组。
- 日志 writer 在正常结束、cache 异常、OOM、中断时都会关闭，避免逐 trial 资源泄漏。
- 不改变 BatchNorm 的单样本尾批限制。最早用 1537 个样本、batch128 时遇到尾批1，基准改为1539来保留尾批3；没有通过 drop_last 丢弃正式数据。

最终执行 `python -X utf8 -m unittest discover -s tests`：109 项测试全部通过。
补充的 `15_corrected_direct_dropout.json` 使用 direct-SFR、base_ch=64、dropout=0.3，
两次交替运行、每次 4 轮；同样通过逐轮验证指标容差检查，权重最大绝对差为 0。
本轮结束时模型源文件与开始快照一致，正式 cache metadata 与开始快照一致。

## 7. 文件与服务器使用

服务器至少同步：`Experiment/train.py`、`Experiment/run_optuna.py`、新增的 `Experiment/training_performance.py`。
模型文件相对本轮开始未改变；建议同步同一版本的 Experiment 代码，不混用旧入口和新依赖。
不需要重新生成 PT，也不需要重建已有 Optuna study。

以本次已存在的 auto cache 为例，在服务器项目根目录运行：

```bash
python Experiment/run_optuna.py \
  --data-exp Exp1_Rh8x8_Closure_auto_ \
  --objective-mode signed_closure \
  --negative-sfr-weight 1.0 \
  --loss l1 \
  --batch-backend tensor_cpu \
  --num-workers 0 \
  --n-trials 100 \
  --timeout 28800
```

若续跑原数据库，必须保持其实际 cache 前缀与 loss 等固定契约；不要机械地把旧的非 auto 前缀改成 auto。
此处 `n-trials=100` 指本次调用尝试的数量，并非“将历史总 trial 数补到 100”。

若想在服务器再测 worker，必须明确 `--batch-backend standard --num-workers N`。
普通数据路径对照可添加 `--write-model-graph --no-reuse-datasets`，但验证向量化仍然启用。
训练集复用只对同一个 Python HPO 进程有效，不能跨重启保留。
原脚本保存快照位于 `Results/performance_audit_20260908/*_before.py`；不自动回退用户其他代码。

## 8. 重现本地/服务器性能对照

需要同时保留 `Results/performance_audit_20260908/train_before.py`，因为它是本轮开始时的原版参考。
默认脚本必须有 CUDA；它不是 CPU-only 测试工具。

```bash
python Experiment/benchmark_training.py --cases baseline batched --epochs 8 --repeats 4 --batch-size 128 --output Results/performance_audit_server/b128.json
python Experiment/benchmark_training_startup.py --cases reference batched --epochs 2 --repeats 3 --train-samples 14615 --val-samples 4871 --output Results/performance_audit_server/full_split.json
python -m unittest tests.test_training_performance tests.test_optuna_objective_contract
```

运行 benchmark 时不要并行运行其他 GPU 任务，固定供电/散热状态，比较同一硬件的交替配对结果。
`pe`/`accum` 候选重现还需要本地归档的 `models_position_candidate.py` / `train_accum_candidate.py`，不用于正式训练。
benchmark 输出 JSON、测试日志和 TensorBoard 写入新的审计目录，不覆盖正式 cache、checkpoint 或 HPO 数据库。
