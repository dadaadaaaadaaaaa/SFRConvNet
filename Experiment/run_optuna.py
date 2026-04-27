'''
自动化超参数优化,调用train的main函数,人为设定一个搜索空间,然后每一次训练都从这个空间中采样一个配置进行训练,
最后返回验证集上的最佳性能指标,也就是最优的一组参数组合
只有trial.suggest的变量才会被自动优化,如学习率,模型维度等.其他参数如训练轮数,早停耐心等则保持固定以加速单次训练的速度.
'''
import os
import sys
import torch
import optuna
import gc # 清理内存的库
from functools import partial

# Add the current directory to sys.path so we can import 'train'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import train

# Define the number of trials and timeout (in seconds)
N_TRIALS = 600 # 600次试验
TIMEOUT = 3600 * 8  # 8 hours


def objective(trial, data_exp):
    # 1. Clean up previous runs (memory) 清理内存
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Get base arguments
    #    Pass an empty list to avoid conflict with actual CLI args
    args = train.parse_args()

    # 3. Suggest Hyperparameters
    #    Define your search space here.

    # --- Architecture ---

    # args.model_type = "transformer"
    # args.d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128, 256, 512]) #模型维度
    # # n_layers must match d_model/heads constraint typically, or just simple
    # args.n_layers = trial.suggest_int("n_layers", 2, 12) #层数
    # args.n_heads = trial.suggest_categorical("n_heads", [2, 4, 8]) #注意力头数，必须能整除d_model
    # args.d_ff = trial.suggest_categorical("d_ff", [128, 256, 512, 1024, 2048])

    args.model_type = "convnet"
    args.base_ch = trial.suggest_categorical("base_ch", [32, 64, 128]) #模型的一些超参数
    args.se_reduction = trial.suggest_categorical("se_reduction", [4, 8, 16])

    # --- Positional Encoding ---
    args.pos_num_bands = trial.suggest_categorical("pos_runsnum_bands", [8, 16, 32, 64])

    # --- Dropout ---
    args.p_drop = trial.suggest_float("p_drop", 0.0, 0.5, step=0.1)

    # --- Optimization ---
    args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True) # 学习率
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    # Using larger batch sizes for RTX 5070 Ti since data is small (8x16)
    args.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])

    # --- Training Config (Fixed for HPO speed) ---
    args.epochs = 90  # Keep epochs relatively low for HPO, or use pruning  仅设置90轮来加速单次训练的速度
    args.patience = 15  # Aggressive early stopping for HPO 较高的早停耐心值
    args.num_workers = 0  # Safer on Windows
    args.exp_name = f"HPO_Trial{trial.number:03d}_"  # Unique name for saving results
    args.data_exp = data_exp  # FIXED: Base name for loading data
    args.log_dir = os.path.join("runs", "HPO")
    args.no_save = True

    # # Ensure d_model is divisible by n_heads
    # if args.d_model % args.n_heads != 0:
    #     # Invalid config, prune immediately or adjust
    #     raise optuna.exceptions.TrialPruned(f"d_model {args.d_model} not divisible by n_heads {args.n_heads}")

    # 4. Run Training
    print(f"\n[Optuna] Starting Trial {trial.number} with config: {trial.params}")
    try:
        best_val_loss = train.main(args) # 调用train,返回验证集上的最佳损失值
    except Exception as e:
        print(f"[Optuna] Trial {trial.number} failed with error: {e}")
        # Return a bad value or re-raise
        return float("inf")

    # 5. Return Metric
    return best_val_loss


if __name__ == "__main__":
    # Create storage for persistence (optional)
    data_exp = "Exp5_"
    study_name = f".cache/{data_exp}macnet_vit_hpo"
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
    # 一次完整的参数优化过程被称为一个study

    print("Starting optimization...")
    study.optimize(lambda t: objective(t, data_exp), n_trials=N_TRIALS, timeout=TIMEOUT) #运行优化,最多跑600次或者8小时

    print("\n" + "=" * 50)
    print("Optimization Finished!")
    print("=" * 50)

    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}") # 输出最优秀的一组参数组合
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
