# run_train_sfr_unet.py：正式训练启动器

使用 [train_sfr_unet.py](train_sfr_unet.md) 的全部参数和同一训练核心，额外支持 Exp2 manifest、多 seed。
不读取旧 `.config/ExperimentSetting.json`，避免误继承标量模型或 loss 设置。
仅支持当前 Exp1/Exp2，objective 固定为逐格 signed closure。

## 单实验/单折

```powershell
python Experiment/run_train_sfr_unet.py --data-exp Exp1_Rh12x20_FiveRuns_Closure_ --cell-exp Exp1_Rh12x20_FiveRuns_Cell_ --device cuda
python Experiment/run_train_sfr_unet.py --data-exp Exp2_Rh12x20_Closure_v7_LOO_disk_galaxy_fiducial_ --cell-exp Exp2_Rh12x20_Cell_v7_LOO_disk_galaxy_fiducial_ --device cuda
```

名称只是示例，需与本地文件对应。所有 split 的 `<cell_exp><split>_cells.pt` 必须提前生成。

## Exp2 多折

如果 `<data_exp>metadata.json` 不存在，则查找 `<data_exp>folds.json`，逐折串行训练。
manifest 来自原 `prepare_data.py --experiments exp2`，保留其 split，不生成新划分。

```powershell
python Experiment/run_train_sfr_unet.py --data-exp Exp2_Rh12x20_FiveRuns_Closure_ --cell-exp Exp2_Rh12x20_FiveRuns_Cell_ --held-out-galaxy model_0_75 --device cuda
```

`--held-out-galaxy` 可重复，接受完整名 `disk_galaxy_model_0_75` 或简称 `model_0_75`；省略则运行 manifest 全部折。
当使用 manifest 基础前缀时，sidecar 基础前缀自动追加每折后缀。
例如原折前缀 `BASE_LOO_disk_galaxy_fiducial_`，`--cell-exp CELL_` 对应
`CELL_LOO_disk_galaxy_fiducial_train_cells.pt`。
直接使用单折前缀时，`--cell-exp` 也必须写完整单折前缀。

## Optuna 最佳配置与多个随机种子

```powershell
python Experiment/run_train_sfr_unet.py --config .cache/STUDY/best_config.json --device cuda --seeds 42 43 44
```

`best_config.json` 已包含固定参数和最优搜索参数，不需要手抄 trial 日志。
`--seeds` 默认仅使用配置中的 seed，可给多个不同整数；不能将 seed 作为最优参数挑选。
CLI 覆盖 JSON；新训练总是创建独立输出目录。同一折的 dataset/训练尺度在多个 seed 间复用。
不将 held-out 结果用于选 seed，正式报告应保留所有重复实验。

额外参数只有 `--held-out-galaxy`（默认无，可重复）、`--seeds`（默认配置 seed）；
其余默认值、参数范围、输出与恢复见训练核心文档。`--resume` 只允许单折、单 seed。
首版遇到错误即停止，保留此前已完成的结果；不默默跳过失败折。
