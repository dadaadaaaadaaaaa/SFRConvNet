# training_performance.py：训练性能辅助接口

本模块不独立运行，不生成 cache，不改变物理监督量。由 `train.py`、`run_optuna.py` 导入。

## 1. DatasetCache

`DatasetCache(max_entries=4)`：有界的已解析 CPU dataset 缓存；HPO 显式使用 2，存放 train/val。

`load(dataset_type, path, objective_mode=..., augmentor=None)`：

- 第一次调用通过原 `GalPTDataset` 完整加载、校验；后续返回对象浅拷贝，不重读 PT。
- key 包含 dataset 类型、绝对路径、objective 和 PT/metadata 的大小、mtime_ns、ctime_ns。
- 文件属性变化后重载，最多保留 `max_entries` 项。
- 原 tensor 共享且约定只读，增强器对象在返回对象中单独设置；不缓存模型、优化器或 CUDA 权重。
- 不应在 HPO 过程中覆盖 cache；若修改源码增加 tensor 原地写入，需要重新审查此只读约定。

## 2. make_training_loader

`make_training_loader(dataset, args, device, shuffle=...)`：

- `args.batch_backend=standard`：普通 DataLoader，实际 worker 数等于 `args.num_workers`。
- `tensor_cpu`：一次用同一索引批量提取输入、坐标、条件、目标、baseline、物理 SFR、组标签，避免逐样本 Python 调用及 collate。
- tensor 路径要求 `num_workers=0` 且 `dataset.augmentor is None`，否则明确报错。
- 不改变 `drop_last=False`，不通过丢弃尾批获取虚假的加速。

当 `args.sampling_mode=run_time_balanced` 且 `shuffle=True` 时，训练 loader 使用
`WeightedRandomSampler(replacement=True, num_samples=len(train))`。设某运行按时间排序后的
时间为 \(t_i\)，端点权重取相邻间隔的一半，内部点权重取左右间隔和的一半：

\[
w_0=\frac{t_1-t_0}{2},\qquad
w_i=\frac{t_{i+1}-t_{i-1}}{2},\qquad
w_{n-1}=\frac{t_{n-1}-t_{n-2}}{2}.
\]

随后每个运行内部归一化到总权重 1，所以不同运行拥有相同的理论总抽样概率；间隔较大的
snapshot 代表更长物理时间。重复或倒序时间会直接报错。`shuffle=False` 的 validation/test
不使用该 sampler，因此不会因评估集重采样而改变指标定义。

## 3. TensorBatchLoader

`TensorBatchLoader(dataset, batch_size, shuffle=..., device="cpu", batches=None)`。

使用与 DataLoader 相同的 CPU sampler 和迭代器 seed 消耗，返回既有 direct 六元组或 closure 八元组。
`n_groups` 是单个整数，而非重复 batch_size 次的 tensor，训练解析器同时支持二者。
`batches` 用于 benchmark 指定完全一致的索引；正式训练不传它。
内部 `device` 参数仅保留用于重现 GPU 常驻对照实验，正式训练入口只启用 CPU 整批路径。

若训练集最后一批只有 1 个样本且模型含 BatchNorm 的 1×1 特征图，仍会触发原模型的限制。
此优化不悄悄丢弃、重复或合并该样本；需要另行明确选择批次策略。

## 4. 验证

```bash
python -m unittest tests.test_training_performance tests.test_optuna_objective_contract
```

测试覆盖 batch 对齐、RNG、尾批、direct/closure、组权重、worker=1、cache 失效、测试集隔离和日志资源释放。
