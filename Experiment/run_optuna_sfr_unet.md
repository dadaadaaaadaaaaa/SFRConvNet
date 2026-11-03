# run_optuna_sfr_unet.py：逐格 SFRUNet 的 Exp1/Exp2 HPO

与正式训练共用 [train_sfr_unet.py](train_sfr_unet.md)。仅加载 train/val，
不打开 test/out，不用 held-out 星系的标签、指标或尺度选参。
每折的 dataset、逐格尺度仅加载/拟合一次，trial 间复用。

## 所需文件及命令

需上传训练代码、原 `<data_exp>train.pt/val.pt/metadata.json/norm.npz`，
以及 `<cell_exp>train_cells.pt/val_cells.pt`；多折时还需 folds.json。
不需要原始 raw 或 processed HDF5。正式训练则还需要 test/out 及它们的 sidecar。

```powershell
python Experiment/run_optuna_sfr_unet.py --data-exp Exp1_Rh12x20_FiveRuns_Closure_ --cell-exp Exp1_Rh12x20_FiveRuns_Cell_ --device cuda --n-trials 50 --timeout 28800
```

Exp2 单折用完整前缀；多折使用基础前缀与 manifest：

```powershell
python Experiment/run_optuna_sfr_unet.py --data-exp Exp2_Rh12x20_FiveRuns_Closure_ --cell-exp Exp2_Rh12x20_FiveRuns_Cell_ --held-out-galaxy model_0_75 --device cuda --n-trials 50
```

例中前缀须替换为实际文件名，不自动补齐逐格标签。

## 搜索参数

默认只搜索第一轮六项：

| 参数名 | 候选/范围 | 是否默认搜索 |
|---|---|---|
| `lr` | `[1e-5,3e-3]` 对数采样 | 是 |
| `weight_decay` | `[1e-7,1e-3]` 对数采样 | 是 |
| `batch_size` | `32,64,128,256` | 是 |
| `base_ch` | `16,32,64` | 是 |
| `p_drop` | `0,0.1,0.2,0.3` | 是 |
| `bottleneck_depth` | `1,2` | 是 |
| `pos_num_bands` | `2,4,8` | 否 |
| `pos_max_freq` | `2,4,8,16` | 否 |
| `se_reduction` | `4,8,16` | 否 |
| `total_weight` | `0,0.3,1,3` | 否 |
| `negative_weight` | `0.3,1,3` | 否 |

`--search-params` 显式指定子集（空格/逗号分隔，名称用下划线）。未搜索的参数使用训练默认值，
或 JSON/CLI 中的固定值。**列入 search-params 的参数以搜索结果为准，即使 CLI 也提供该值。**
不搜索 seed、split、分辨率、归一化尺度、FiLM、数据采样方式或输入通道。

```powershell
# 固定其他训练参数，穷举位置编码组合。
python Experiment/run_optuna_sfr_unet.py --data-exp PREFIX_ --search-params pos_num_bands pos_max_freq --sampler grid --lr 0.0003 --base-ch 32 --batch-size 64 --n-trials 12 --no-pruning
```

上例为 3×4=12 个离散组合。Grid 不支持连续 lr/weight_decay，须先移出搜索列表并固定。

## HPO 额外参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--search-params` | 上表六项 | 实际参与搜索的参数名 |
| `--n-trials` | 50 | **本次新增**的尝试次数，包括剪枝/失败，不是数据库累计上限 |
| `--timeout` | 28800 | 每折时间预算，秒；不会强行打断正在运行的 trial |
| `--sampler` | `tpe` | `tpe/grid` |
| `--sampler-seed` | 42 | 控制超参数抽样，区别于模型训练 seed |
| `--pruning / --no-pruning` | 开启 | MedianPruner，至少5个启动trial、warmup至少10轮后考虑剪枝 |
| `--held-out-galaxy` | 全部折 | Exp2 筛选，语义与正式启动器相同 |

其余参数与训练核心共用：默认100轮、patience20、固定训练 seed42、CPU线程4。
HPO 不接受 `--resume` 或 `--stop-after-epochs`。不写逐 trial TensorBoard，不导出测试地图。
非有限损失/梯度和 CUDA OOM 标记为剪枝并记录原因；其他异常直接报告，不掩盖接口错误。

## 指标、输出和续跑

study 按固定 `run_macro(Lcell+Ltotal+Lnegative)` 最小化。
即使搜索 total_weight/negative_weight，选模系数仍固定为1，尺度亦固定，不能靠缩小训练权重取得低分。
各 trial 的训练目标保持所采样的权重；两者语义在输出契约里区分。

数据库：`.cache/<前缀>SFRUNet_HPO_v1_<契约摘要>.db`。
契约绑定数据/sidecar/norm/metadata、实现摘要、搜索空间、固定参数、训练预算、seed、选模指标及软件版本。
改变这些条件会创建新的 study，不混合比较。
训练输出在 `Results/HPO_SFRUNet/`；每个 trial 保留 best/last、历史和配置，供复核。

同名 study 配置相同则续跑，sampler 状态保存在 `.cache/<study>/sampler.pt`。
最佳结果逐 trial 更新至：

```text
.cache/<study>/best_trial.json
.cache/<study>/best_config.json
```

最优配置同时包含搜索参数和固定参数，可直接正式训练：

```powershell
python Experiment/run_train_sfr_unet.py --config .cache/STUDY/best_config.json --seeds 42 43 44
```

Ctrl+C 中断后重新运行原 HPO 命令，会继续新增 trial；**不会自动恢复被中断 trial 的半轮训练**。
强制杀进程可能留下 RUNNING 记录，本入口不擅自把它算成完成结果；最佳结果只从 COMPLETE 中选择。
正常关闭释放 SQLite 连接，数据库可移动。不要对同一 SQLite study 启动并行写入；当前仅串行单进程 HPO。
若所有 trial 均剪枝，则报告 completed_trials=0，best_config 为 null，不能继续假定有最佳模型。
