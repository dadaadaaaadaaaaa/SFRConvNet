# `run_optuna.py` 使用说明

## 1. 功能概述

`run_optuna.py` 使用 Optuna 搜索 SFRConvNet 超参数。每个 study 会：

1. 检查 train/val/test cache 和 `metadata.json` 是否完整；
2. 验证可用训练目标和条件变量契约；
3. 默认从 `prepared_objective` 自动解析训练目标，也允许显式指定；
4. 从搜索空间为每个 trial 采样配置；
5. 只用训练集训练、用验证集选择超参数，不在 HPO 阶段评估测试集；
6. 将 study 持久化到包含训练目标名称的 SQLite 数据库。

## 2. 当前可选搜索空间

| 参数 | 采样方式 | 可选值/范围 |
|---|---|---|
| `model_type` | 固定 | `convnet` |
| `base_ch` | categorical | `32`、`64`、`128` |
| `se_reduction` | categorical | `4`、`8`、`16` |
| `pos_num_bands` | categorical | `4`、`8`、`16` |
| `pos_max_freq` | categorical | `2`、`4`、`8`、`16`、`32`、`101` |
| `p_drop` | 等间隔浮点 | `0.0` 至 `0.5`，步长 `0.1` |
| `lr` | 对数均匀 | `1e-5` 至 `1e-2` |
| `weight_decay` | 对数均匀 | `1e-6` 至 `1e-3` |
| `batch_size` | categorical | `64`、`128`、`256`、`512`、`1024` |

主损失函数、训练目标、负 SFR 惩罚系数和是否启用 FiLM 是固定实验条件，
不是同一 study 内的待搜索变量。

不传 `--search-params` 时，上表八个可搜索参数全部参与采样。也可以用
`--search-params` 指定其中任意子集；未选中的参数不调用 Optuna 采样，而是使用
`--fixed-params 参数名=值` 给出的值；没有显式固定的未搜索参数使用
`train.py::parse_args()` 的训练入口默认值。`model_type=convnet` 始终固定，不能列入
这两个选项。

当前搜索空间版本为 `v2`，完整搜索的 study 名与数据库文件名均带 `_v2` 后缀。
这样不会把旧版 `pos_num_bands=[8,16,32,64]` 的 trial 与新的联合 Fourier
搜索混入同一个 Optuna study；旧数据库会原样保留。

子集搜索会再增加 `_only_<参数列表>` 后缀。例如，只搜索两个 Fourier 参数时为
`_v2_only_pos_num_bands-pos_max_freq`。参数列表按代码中的规范顺序排列，因此命令行
书写顺序不同不会重复创建等价 study；不同搜索子集则不会静默混用历史 trial。若
固定值不同于训练默认值，还会增加 `_fixed_<10位配置哈希>` 后缀，让不同固定配置
进入不同数据库；完整固定值仍记录在 study 属性中。

采样器可选：

- `tpe`：默认，自适应且允许重复采样，不保证覆盖全部离散组合；
- `grid`：逐一覆盖有限候选集合，单进程下每个格点只运行一次，穷尽后自动停止。

当前 `lr`、`weight_decay` 是连续对数搜索空间，不能用于 `GridSampler`。其余六个
可搜索参数均有有限网格。只搜索两个 Fourier 参数时，grid 大小严格为 `3×6=18`。

## 3. 每个 trial 的固定参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `epochs` | `90` | HPO 最大训练轮数 |
| `patience` | `15` | early stopping |
| `num_workers` | `0` | 可通过命令行修改 |
| `log_dir` | `runs/HPO/<objective>` | TensorBoard 目录 |
| `no_save` | `True` | 不保存正式最佳模型 |
| `skip_test` | `True` | 不在每个 trial 反复评估测试集 |

固定 (R_h) cache 会自动关闭 FiLM；带活动 `log_mass_filter` 条件的 (R_m)
cache 会自动启用 FiLM。二者均由 metadata 决定。

## 4. 命令行参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--cache-dir` | `.cache` | CachePT 和 metadata 所在目录 |
| `--data-exp` | `Exp1_InstantSFR_NoInitialStellarDensity_` | Cache 文件前缀 |
| `--objective-mode` | `auto` | `auto`、`direct_sfr` 或 `signed_closure` |
| `--sampling-mode` | `auto` | 从 cache metadata 的 `training_sampling` 解析；也可显式用 `uniform` 或 `run_time_balanced` |
| `--negative-sfr-weight` | `1.0` | closure 模式下最终负 SFR 的惩罚系数 |
| `--loss` | `mse` | 所有 trial 固定使用 `mse`、`l1` 或 `huber` |
| `--num-workers` | `0` | DataLoader worker 数量 |
| `--batch-backend` | `tensor_cpu` | CPU 整批索引；`standard` 使用普通 DataLoader。前者要求 worker 为 0 |
| `--write-model-graph` | 关闭 | 开启后逐 trial 追踪 TensorBoard 模型图；默认只写标量 |
| `--no-reuse-datasets` | 关闭 | 开启后逐 trial 重新读取 PT；默认在同一个 HPO 进程复用已校验的 CPU train/val |
| `--search-params` | 全部八项 | 指定本次由 Optuna 搜索的参数；支持空格或逗号分隔，也可显式传 `all` |
| `--fixed-params` | 训练入口默认值 | 按 `参数名=值` 覆盖未搜索参数；支持空格或逗号分隔 |
| `--sampler` | `tpe` | `tpe` 自适应采样，或 `grid` 穷举有限组合 |
| `--sampler-seed` | 随机 | 只控制 sampler 的组合顺序，不控制模型训练 |
| `--seeds` | `42` | 模型训练 seed；多个 seed 只用于 grid，并扩展为重复实验维度 |
| `--n-trials` | `600` | 最大 trial 数 |
| `--timeout` | `28800` 秒 | study 总时限，默认 8 小时 |

trial 数或总时限任一先达到，不再启动后续 trial；`timeout` 通常不会强制中断正在运行的 trial。
sampling mode 会写入 study 契约；非 `uniform` 模式也会进入 study/database 名称，避免把
不同训练分布的 trial 混入同一个 study。Exp3 metadata 声明 `run_time_balanced`，因此默认
无需手工指定，且 HPO 仍只加载 train/validation，不接触 torus test。

## 5. 执行命令

为 signed-closure cache 调参：

```powershell
python Experiment/run_optuna.py `
  --data-exp Exp1_Rh8x8_Closure_auto_ `
  --objective-mode signed_closure `
  --negative-sfr-weight 1.0 `
  --loss l1 `
  --n-trials 100 `
  --timeout 28800
```

也可以省略 `--objective-mode`。此时程序从 cache metadata 自动读取；启动时应明确看到：

```text
Resolved learning objective: signed_closure
```

只搜索 `pos_num_bands` 与 `pos_max_freq`，其余六项固定为训练入口默认值：

```powershell
python Experiment/run_optuna.py `
  --data-exp Exp2_Rh8x8_Closure_auto_ `
  --search-params pos_num_bands pos_max_freq `
  --fixed-params base_ch=64 se_reduction=4 p_drop=0.3 lr=1e-3 weight_decay=0 batch_size=64 `
  --n-trials 100
```

上例显式写出了全部六个固定参数。可以只覆盖关心的固定参数，其余自动采用默认值：

```powershell
python Experiment/run_optuna.py `
  --search-params pos_num_bands pos_max_freq `
  --fixed-params base_ch=128 lr=2e-4
```

以下逗号写法等价：

```powershell
python Experiment/run_optuna.py --search-params pos_num_bands,pos_max_freq
```

可选名称严格限定为：`base_ch`、`se_reduction`、`pos_num_bands`、
`pos_max_freq`、`p_drop`、`lr`、`weight_decay`、`batch_size`。未知名称、重复名称，
或者把 `all` 与具体名称混用都会在启动阶段报错。

`--fixed-params` 使用同一组可选名称。整数参数为 `base_ch`、`se_reduction`、
`pos_num_bands` 和 `batch_size`；其余按浮点数解析。一个参数不能既出现在
`--search-params` 中又出现在 `--fixed-params` 中。固定值不要求属于 Optuna 的候选
集合，例如可以用 `pos_num_bands=32` 复现旧基线，但必须满足参数自身的有效范围。

严格覆盖18个 Fourier 组合，并保持除 batch size 外与 Trial 53 相同的固定参数：

```powershell
python Experiment/run_optuna.py `
  --data-exp Exp2_Rh12x20_FiveRuns_Closure_LOO_disk_galaxy_model_0_75_ `
  --objective-mode signed_closure `
  --loss l1 `
  --negative-sfr-weight 1.0 `
  --search-params pos_num_bands pos_max_freq `
  --fixed-params base_ch=64 se_reduction=8 p_drop=0 lr=0.0006016003122768778 weight_decay=3.220210353980758e-06 batch_size=256 `
  --sampler grid `
  --sampler-seed 42 `
  --seeds 42
```

即使保留默认的 `--n-trials 600`，上面的单进程 GridSampler 也会在18个格点都运行后
自动停止。`(pos_num_bands=16, pos_max_freq=101)`自然包含在网格中，可作为
“Trial 53其余参数不变、只把 batch size 改为256”的内部对照。

若需要每个 Fourier 组合运行三个训练 seed：

```powershell
--sampler grid --sampler-seed 42 --seeds 42 43 44
```

此时网格大小为 `18×3=54`。seed 是重复实验维度，不是应该挑选最优值的模型超参数；
分析时应按 Fourier 组合聚合三个 seed 的均值、中位数和标准差，而不是采用“最佳 seed”。

查看参数：

```powershell
python Experiment/run_optuna.py --help
```

## 6. 输出与恢复

SQLite 数据库路径包含训练目标：

```text
.cache/<data-exp><objective-mode>_sfrconvnet_hpo_v2.db
.cache/<data-exp><objective-mode>_sfrconvnet_hpo_v2_only_<参数列表>.db
.cache/<data-exp><objective-mode>_sfrconvnet_hpo_v2_only_<参数列表>_fixed_<配置哈希>.db
.cache/<data-exp><objective-mode>_sfrconvnet_hpo_v2_only_<参数列表>_fixed_<配置哈希>_grid.db
```

study 名称同样包含 `<objective-mode>`，因此 direct-SFR 和 signed-closure 不会
恢复到同一个历史 study。study user attributes 还会记录 cache 前缀、目标、条件变量
和 closure 变换，并保存 cache metadata 的 SHA-256 指纹、固定损失函数、负 SFR
惩罚系数、搜索空间版本、实际搜索参数和未搜索参数的固定值；恢复时任一契约变化，
程序都会拒绝继续。采样器、sampler seed 和训练 seed 集合也属于 study 契约。每个
trial 还会记录 `search_params`、`fixed_hyperparameters` 与实际 `training_seed`，便于
事后确认实验条件。

## 7. 注意事项

1. 两种目标应使用相同的 split、HPO 搜索空间和计算预算；
2. CUDA OOM 只剪枝当前 trial；其他数据契约、维度或代码异常会立即中止；
3. batch size `1024` 可能使大模型 trial OOM，此时属于可预期剪枝；
4. HPO 验证 split 必须避免时间泄漏；
5. HPO 不评估测试集。确定超参数后应多随机种子正式训练，最后才使用测试集；
6. 旧版 `<data-exp>sfrconvnet_hpo.db` 可能记录了错误的 direct-SFR trial，应单独归档。

## 8. 2026-09-08：经本地测量保留的性能优化

默认启用 CPU 整批索引、同进程 train/val dataset 复用，并跳过重复模型图。
验证端改为设备内按星系累计，保持“先计算每个星系均值，再对有样本的星系等权平均”的指标。
HPO 不反序列化 test.pt，也不写 checkpoint；仍写 TensorBoard 标量和 Optuna 结果。
trial 的 `performance_settings` 记录本次数据路径、worker、复用和模型图开关。

这些变化没有调整学习目标、closure scale、损失、采样规则、搜索空间或 early stopping。
保持 cache 前缀和固定契约一致时可以沿用原 study；不能为了续跑而把不同前缀/损失的数据库混用。
同进程复用不等于跨进程持久化：重启 HPO 后第一个 trial 会重新加载。
缓存有界且以 PT/metadata 的文件属性变化判断失效，但不要在 HPO 运行期间覆盖数据文件。

恢复普通数据路径并禁用这两项启动优化：

```bash
python Experiment/run_optuna.py --data-exp Exp1_Rh8x8_Closure_auto_ --objective-mode signed_closure --loss l1 --batch-backend standard --num-workers 0 --write-model-graph --no-reuse-datasets
```

这不是恢复整个旧版代码：验证统计向量化和 worker 数 bug 修复仍生效。
不新增强制 CPU 线程数，不在正式入口提供 GPU 常驻和位置编码缓存；它们在本地没有可靠的额外收益。
详细实验结果见 `TRAINING_PERFORMANCE_AUDIT_20260908.md`。Windows/RTX 4060 结论不能直接视为 Linux/A100 加速比。
