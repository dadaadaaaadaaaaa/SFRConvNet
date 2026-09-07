"""使用 Optuna 为 SFRConvNet 搜索超参数。

训练目标属于 HPO study 的固定实验契约：默认从 cache metadata 的
``prepared_objective`` 自动读取，也可由命令行显式指定。direct-SFR 与
signed-closure 使用不同的数据库和 study 名，禁止静默混合。
"""

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import optuna
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import train


N_TRIALS = 600
TIMEOUT = 3600 * 8
SUPPORTED_OBJECTIVES = ("direct_sfr", "signed_closure")
SEARCH_SPACE_VERSION = 1


def _load_cache_contract(cache_dir: str, data_exp: str) -> dict:
    """读取轻量 metadata，并在启动 HPO 前检查 cache 集合是否完整。"""

    cache_root = Path(cache_dir)
    missing = [
        str(cache_root / f"{data_exp}{split}.pt")
        for split in ("train", "val", "test")
        if not (cache_root / f"{data_exp}{split}.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "HPO 所需 cache 不完整，缺少：\n  - " + "\n  - ".join(missing)
        )

    metadata_path = cache_root / f"{data_exp}metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"找不到 cache metadata：{metadata_path}。"
            "新版 HPO 不再仅凭文件名猜测训练目标，请重新运行 prepare_data.py。"
        )
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 cache metadata：{metadata_path}") from exc

    prepared = metadata.get("prepared_objective", metadata.get("objective_mode"))
    available = tuple(metadata.get("available_objectives", ()))
    if prepared not in SUPPORTED_OBJECTIVES:
        raise RuntimeError(
            f"cache metadata 的 prepared_objective 非法或缺失：{prepared!r}。"
        )
    if not available or any(value not in SUPPORTED_OBJECTIVES for value in available):
        raise RuntimeError(
            f"cache metadata 的 available_objectives 非法或缺失：{available!r}。"
        )
    if prepared not in available:
        raise RuntimeError(
            f"prepared_objective={prepared!r} 不在 available_objectives={available!r} 中。"
        )

    conditioning_name = str(metadata.get("conditioning_variable", "unknown"))
    conditioning_active = bool(metadata.get("conditioning_active", False))
    if conditioning_active and conditioning_name != "log_mass_filter":
        raise RuntimeError(
            "当前 SFRConvNet HPO 只支持 inactive R_h 条件，或活动的 "
            f"log_mass_filter R_m 条件；实际为 {conditioning_name!r}。"
        )
    if not conditioning_active and conditioning_name not in {"none", "unknown"}:
        raise RuntimeError(
            "cache 条件契约自相矛盾："
            f"conditioning_variable={conditioning_name!r}, active=False。"
        )

    fingerprint_payload = dict(metadata)
    metadata_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metadata["prepared_objective"] = prepared
    metadata["available_objectives"] = list(available)
    metadata["conditioning_variable"] = conditioning_name
    metadata["conditioning_active"] = conditioning_active
    metadata["metadata_path"] = str(metadata_path)
    metadata["metadata_fingerprint"] = metadata_fingerprint
    return metadata


def _resolve_objective_mode(requested: str, contract: dict) -> str:
    """解析 auto/显式目标，并验证 cache 是否提供相应监督量。"""

    objective_mode = contract["prepared_objective"] if requested == "auto" else requested
    if objective_mode not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"不支持的 HPO 训练目标：{objective_mode!r}。")
    if objective_mode not in contract["available_objectives"]:
        raise RuntimeError(
            f"cache 不支持 objective_mode={objective_mode!r}；"
            f"可用目标为 {contract['available_objectives']}。"
        )
    if objective_mode == "signed_closure":
        closure_scale = contract.get("closure_scale")
        try:
            closure_scale = float(closure_scale)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"signed-closure cache 缺少合法 closure_scale：{closure_scale!r}。"
            ) from exc
        if not math.isfinite(closure_scale) or closure_scale <= 0:
            raise RuntimeError(
                f"signed-closure cache 的 closure_scale 非法：{closure_scale!r}。"
            )
    return objective_mode


def _study_identity(data_exp: str, objective_mode: str) -> str:
    """构造包含训练目标的 study 身份。"""

    return f"{data_exp}{objective_mode}_sfrconvnet_hpo"


def _bind_study_contract(
    study,
    *,
    data_exp,
    objective_mode,
    contract,
    loss,
    negative_sfr_weight,
):
    """将不可变语义写入 study，并拒绝恢复不兼容的数据库。"""

    expected = {
        "data_exp": data_exp,
        "objective_mode": objective_mode,
        "conditioning_variable": contract["conditioning_variable"],
        "conditioning_active": bool(contract["conditioning_active"]),
        "cache_metadata_sha256": contract["metadata_fingerprint"],
        "loss": loss,
        "negative_sfr_weight": float(negative_sfr_weight),
        "search_space_version": SEARCH_SPACE_VERSION,
    }
    if objective_mode == "signed_closure":
        expected["closure_transform"] = contract.get("closure_transform")
        expected["closure_scale"] = float(contract["closure_scale"])

    for key, value in expected.items():
        previous = study.user_attrs.get(key)
        if previous is not None and previous != value:
            raise RuntimeError(
                f"已有 Optuna study 的 {key}={previous!r}，"
                f"与当前 cache 契约 {value!r} 不一致；请使用新的 study/database。"
            )
        study.set_user_attr(key, value)


def objective(
    trial,
    data_exp,
    *,
    cache_dir=".cache",
    objective_mode="direct_sfr",
    negative_sfr_weight=1.0,
    loss="mse",
    num_workers=0,
    use_film=False,
):
    """执行一个 trial；只有 CUDA OOM 会被剪枝，其他异常立即暴露。"""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args = train.parse_args([])
    args.cache_dir = cache_dir
    args.data_exp = data_exp
    args.objective_mode = objective_mode
    args.negative_sfr_weight = float(negative_sfr_weight)
    args.loss = loss
    args.use_film = bool(use_film)

    args.model_type = "convnet"
    args.base_ch = trial.suggest_categorical("base_ch", [32, 64, 128])
    args.se_reduction = trial.suggest_categorical("se_reduction", [4, 8, 16])
    args.pos_num_bands = trial.suggest_categorical("pos_num_bands", [8, 16, 32, 64])
    args.p_drop = trial.suggest_float("p_drop", 0.0, 0.5, step=0.1)
    args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    args.batch_size = trial.suggest_categorical(
        "batch_size", [64, 128, 256, 512, 1024]
    )

    args.epochs = 90
    args.patience = 15
    args.num_workers = int(num_workers)
    args.exp_name = f"HPO_{objective_mode}_Trial{trial.number:03d}_"
    args.log_dir = os.path.join("runs", "HPO", objective_mode)
    args.no_save = True
    args.skip_test = True

    print(
        f"\n[Optuna] Starting Trial {trial.number}: "
        f"objective={objective_mode}, fixed_loss={loss}, "
        f"negative_sfr_weight={negative_sfr_weight}, params={trial.params}"
    )
    try:
        result = float(train.main(args))
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"CUDA OOM: {exc}") from exc
        raise

    if not math.isfinite(result):
        raise optuna.TrialPruned(
            f"Trial {trial.number} did not produce a finite validation metric."
        )
    return result


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Optuna for an SFR cache variant.")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument(
        "--data-exp",
        default="Exp1_InstantSFR_NoInitialStellarDensity_",
        help="Cache 前缀，例如 Exp1_Rh8x8_Closure_auto_。",
    )
    parser.add_argument(
        "--objective-mode",
        choices=("auto",) + SUPPORTED_OBJECTIVES,
        default="auto",
        help="默认 auto，从 metadata 的 prepared_objective 读取。",
    )
    parser.add_argument("--negative-sfr-weight", type=float, default=1.0)
    parser.add_argument("--loss", choices=("mse", "l1", "huber"), default="mse")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT, help="总时限，单位秒。")
    args = parser.parse_args(argv)
    if args.negative_sfr_weight < 0:
        parser.error("--negative-sfr-weight must be non-negative.")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative.")
    if args.n_trials <= 0:
        parser.error("--n-trials must be positive.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    return args


def main(argv=None):
    cli_args = _parse_args(argv)
    contract = _load_cache_contract(cli_args.cache_dir, cli_args.data_exp)
    objective_mode = _resolve_objective_mode(cli_args.objective_mode, contract)
    use_film = bool(contract["conditioning_active"])

    if cli_args.objective_mode == "auto":
        print(
            "[INFO] objective-mode auto：从 cache metadata 解析为 "
            f"{objective_mode}."
        )
    elif objective_mode != contract["prepared_objective"]:
        print(
            "[WARNING] 显式 HPO 目标与 cache 的 prepared_objective 不同："
            f"requested={objective_mode}, prepared={contract['prepared_objective']}。"
            "该目标在 available_objectives 中，因此允许继续。"
        )

    study_name = _study_identity(cli_args.data_exp, objective_mode)
    database_path = (Path(cli_args.cache_dir) / f"{study_name}.db").resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    storage_name = f"sqlite:///{database_path.as_posix()}"

    print(f"Cache metadata: {contract['metadata_path']}")
    print(f"Resolved learning objective: {objective_mode}")
    if objective_mode == "signed_closure":
        print(
            "Closure contract: "
            f"transform={contract.get('closure_transform')}, "
            f"scale={contract.get('closure_scale')}, "
            f"negative_sfr_weight={cli_args.negative_sfr_weight}"
        )
    print(
        "Conditioning contract: "
        f"variable={contract['conditioning_variable']}, "
        f"active={contract['conditioning_active']}, "
        f"FiLM={'enabled' if use_film else 'disabled'}"
    )
    print(f"Optuna study: {study_name}")
    print(f"Optuna database: {database_path}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )
    _bind_study_contract(
        study,
        data_exp=cli_args.data_exp,
        objective_mode=objective_mode,
        contract=contract,
        loss=cli_args.loss,
        negative_sfr_weight=cli_args.negative_sfr_weight,
    )

    print("Starting optimization...")
    study.optimize(
        lambda trial: objective(
            trial,
            cli_args.data_exp,
            cache_dir=cli_args.cache_dir,
            objective_mode=objective_mode,
            negative_sfr_weight=cli_args.negative_sfr_weight,
            loss=cli_args.loss,
            num_workers=cli_args.num_workers,
            use_film=use_film,
        ),
        n_trials=cli_args.n_trials,
        timeout=cli_args.timeout,
    )

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
    ]
    if not completed:
        print("Optimization finished without a completed finite trial.")
        return 1

    best = study.best_trial
    print("\n" + "=" * 50)
    print("Optimization Finished!")
    print("=" * 50)
    print(f"Best Trial: {best.number}")
    print(f"  Objective: {objective_mode}")
    print(f"  Value: {best.value}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
