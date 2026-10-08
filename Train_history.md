### 2026/9/8
Model:Exp1_Rh8x8_Closure_s250_SFRConvNet

--model_type convnet   --objective_mode signed_closure   --negative_sfr_weight 1.0   --loss l1   --base_ch 128   --se_reduction 8   --pos_num_bands 16   --p_drop 0.0   --lr 5.644344534658723e-05   --weight_decay 0.00038791011128247265   --batch_size 128   --stage_depths 2,2,2,2   --pos_max_freq 101   --epochs 90   --patience 15

Test metric loss: 0.035214, per-domain: [0.01627811679329925, 0.03433354309339011, 0.05502939500083259]

Results: Results/Exp1_Rh8x8_Closure_BestTrial61_20260908_082411

### 2026/9/9
Model: Exp1_Rh8x8_Direct_Trial27_seed42_20260909_200536

python Experiment/train.py \
  --cache_dir .cache \
  --data_exp Exp1_Rh8x8_Direct_auto_ \
  --exp_name Exp1_Rh8x8_Direct_Trial27_seed42 \
  --model_type convnet \
  --objective_mode direct_sfr \
  --loss mse \
  --weight_mode none \
  --base_ch 128 \
  --se_reduction 16 \
  --pos_num_bands 16 \
  --pos_max_freq 101 \
  --stage_depths 2,2,2,2 \
  --p_drop 0.5 \
  --lr 0.0018286288968749783 \
  --weight_decay 0.00024006049684757616 \
  --batch_size 64 \
  --epochs 90 \
  --patience 15 \
  --lr_min 0 \
  --lr_warmup_steps 0 \
  --lr_warmup_ratio 0.2 \
  --lr_warmup_start_factor 0 \
  --grad_clip 0.3 \
  --seed 42 \
  --device cuda \
  --batch_backend tensor_cpu \
  --num_workers 0 \
  --sfr_aug_prob 0 \
  --skip_model_graph

  Test metric loss: 0.001008, per-domain: [0.0024703701409317155, 0.00024862027304165717, 0.00030449387454583145]

### 2026/9/12
Model：Exp1_Rh12x20_Closure_BestTrial55_seed42_20260910_111655

### 2026/9/12
Model:Exp2_Rh12x20_FiveRuns_Closure_LOO_Model075_Trial53_Final_20260913_004906

### 2026/9/14
Model(only for HPO): Exp2_Rh12x20_FiveRuns_Closure_LOO_Model075
Fourier 配置（4，4）最稳定，（4，101）和（8，4）次之，都较于（16，101）有明显提升

### 2026/9/19
Model：Exp3_Rh12×20_FiveRuns_Closure_Trial23Params_seed42
泛化内边界为1pc的星系，泛化效果很差，预测值持续高一个数量级

### 2026/9/20
Model：Exp3_Rh12x20_FiveRuns_Direct_Trial61Params_seed42_

--cache_dir .cache   --data_exp Exp3_Rh12x20_FiveRuns_Direct_   --exp_name Exp3_Rh12x20_FiveRuns_Direct_Trial61Params_seed42   --model_type convnet   --objective_mode direct_sfr   --loss mse   --weight_mode none   --base_ch 128   --se_reduction 4   --pos_num_bands 16   --pos_max_freq 32   --p_drop 0.0   --stage_depths 2,2,2,2   --lr 0.00992825268659391   --weight_decay 2.440185070480983e-06   --batch_size 128   --epochs 90   --patience 15   --lr_min 0.0   --lr_warmup_steps 0   --lr_warmup_ratio 0.2   --lr_warmup_start_factor 0.0   --grad_clip 0.3   --sampling_mode run_time_balanced   --batch_backend tensor_cpu   --num_workers 0   --aug_prob 0   --sfr_aug_prob 0   --seed 42   --device auto   --skip_model_graph

Test metric loss: 2.567913, per-domain: [2.567912923192833]