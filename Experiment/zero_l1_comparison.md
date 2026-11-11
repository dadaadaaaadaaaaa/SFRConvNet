# 真实零格子L1与Fiducial Exp2对照（2026-11-12）

新增 `--zero-l1-weight`（JSON `zero_l1_weight`），默认0，允许任意有限非负数。
损失为 `L_existing + zero_l1_weight * mean_snapshot(L_zero)`，其中：

`L_zero = sum_{Y_i==0} abs(Y_hat_i)/cell_scale / max(N_zero,1)`。

仅使用精确零标签，不设置epsilon；按帧零格数归一化，无零格帧为0。
作用于最终物理SFR，不作用于closure或候选q，不再次乘体积；cell_scale沿用train RMS。
relu下等于虚假正SFR惩罚；none下取绝对值，也惩罚零标签处的负预测。
保留candidate辅助监督与固定validation选模指标；选模指标不直接加入新惩罚。
新权重和定义进入loss contract v3、config、checkpoint与HPO身份。
预测文件仍为v2，分析器可继续读取旧预测；旧JSON缺少参数时默认0。
history.train_loss仍保留原5项，新增train_zero_l1_loss；TensorBoard新增train/zero_l1，均为未乘权重的L_zero。
不同损失权重的train总损失不应直接用作性能比较。

## 已有数据可直接复用

本次只改损失；现有同折的train/val/test/out标量`.pt`、`*_cells.pt`、metadata和norm可以继续用，无须重建cache。
下面固定此前Fiducial UNet配置，重新训练0与0.1两个权重的成对多种子实验。
0.1是初始试验值，不是已证明最优；若只试一对，把seeds改为仅42。
使用旧config只继承超参数，不载入旧权重。跨代码版本旧last.pt不强行续训；同版本同契约支持恢复。

## 本地PowerShell正式对照

从项目根目录执行：

```powershell
$pythonExe = "D:/conda_envs/physics_ai/python.exe"
$configPath = "Results/Exp2_Rh12x20_FiveRuns_Closure_v7_LOO_disk_galaxy_fiducial_SFRUNet_seed42_20261104_131515_e66884f5/config.json"
$outputRoot = "Results/ZeroL1_Fiducial_AB"

foreach ($weight in @("0", "0.1")) {
    foreach ($seed in @(42, 43, 44)) {
        & $pythonExe Experiment/run_train_sfr_unet.py `
            --config $configPath `
            --data-exp Exp2_Rh12x20_FiveRuns_Closure_v7_LOO_disk_galaxy_fiducial_ `
            --output-constraint relu --candidate-weight 1 --total-weight 1 `
            --zero-l1-weight $weight --seed $seed `
            --epochs 100 --patience 20 --device cuda --cpu-threads 4 `
            --output-root $outputRoot `
            --run-name "Exp2_LOO_Fiducial_UNet_ZeroL1_${weight}_seed${seed}"
        if ($LASTEXITCODE -ne 0) { throw "训练失败：weight=$weight seed=$seed" }
    }
}
```

每次创建独立时间戳目录，不覆盖旧训练。相同种子是成对比较设计，CUDA运行仍可能有非确定性。
首次启动核对zero_l1_weight、relu、candidate_weight=1、数据前缀与输入通道。

## 批量结果分析

```powershell
Get-ChildItem -LiteralPath $outputRoot -Directory | ForEach-Object {
    if (Test-Path (Join-Path $_.FullName "out_predictions.pt")) {
        & $pythonExe VisualScripts/ResultAnalysis/analyze_sfr_unet.py --result-dir $_.FullName
        if ($LASTEXITCODE -ne 0) { throw "分析失败：$($_.FullName)" }
    }
}
```

每次分析写独立目录。现有Results_stat_Exp2_SFRUNet.ipynb切换RESULT_DIR也可用。
metrics.csv新增ZeroCellFalseSFR_mean（零标签处虚假正SFR逐帧求和的平均）和PositiveCellUnderSFR_mean（正标签处低估量逐帧求和的平均），单位Msun/yr；Model/Baseline均提供。
预测CSV新增model/baseline_zero_cell_false_sfr与model/baseline_positive_cell_under_sfr逐帧列，可检查burst、低SFR和异常帧。
对比同seed的总SFR MAE/RMSE、积分质量、逐格误差、新两项指标，再汇总三个seed；不按out挑seed或调权重。

## 可选：仅用validation搜索零L1权重

```powershell
& $pythonExe Experiment/run_optuna_sfr_unet.py `
    --config $configPath --output-constraint relu --candidate-weight 1 --total-weight 1 `
    --search-params zero_l1_weight --sampler grid --n-trials 6 --no-pruning `
    --epochs 100 --patience 20 --device cuda --cpu-threads 4
```

离散范围`{0,0.01,0.03,0.1,0.3,1}`，默认六项架构/优化器搜索列表不变。
从打印的study导出目录读取best_config.json，再用run_train_sfr_unet.py --config <该路径> --seeds 42 43 44正式训练。
HPO仍只看validation；若采用HPO路线，测试结果留到选定权重后评估。
无需为了本次损失消融重新对两组分别搜索架构。

## 验证

`python -m unittest tests.test_sfr_unet_zero_l1 tests.test_sfr_unet_nonnegative tests.test_sfr_unet_training tests.test_analyze_sfr_unet tests.test_sfr_unet -q`

35项通过，覆盖精确零/极小正标签、不同零格数量、无零格、梯度、单位缩放、默认原算式、Exp2训练/恢复/导出/分析、HPO、CUDA FP32/BF16。
只运行小型可靠性验证，未启动真实数据正式训练，尚无改善结论。
