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
from Experiment.training_performance import DatasetCache


N_TRIALS = 600
TIMEOUT = 3600 * 8
SUPPORTED_OBJECTIVES = ("direct_sfr", "signed_closure")
SEARCH_SPACE_VERSION = 2
SEARCHABLE_PARAMS = (
    "base_ch",
    "se_reduction",
    "pos_num_bands",
    "pos_max_freq",
    "p_drop",
    "lr",
    "weight_decay",
    "batch_size",
)
INTEGER_SEARCH_PARAMS = {
    "base_ch",
    "se_reduction",
    "pos_num_bands",
    "batch_size",
}
GRID_SEARCH_SPACES = {
    "base_ch": (32, 64, 128),
    "se_reduction": (4, 8, 16),
    "pos_num_bands": (4, 8, 16),
    "pos_max_freq": (2, 4, 8, 16, 32, 101),
    "p_drop": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    "batch_size": (64, 128, 256, 512, 1024),
}


def _resolve_search_params(requested=None) -> tuple[str, ...]:
    """规范化待搜索参数；未指定时启用完整搜索空间。"""

    if requested is None:
        return SEARCHABLE_PARAMS
    if isinstance(requested, str):
        requested = (requested,)

    values = []
    for item in requested:
        values.extend(part.strip() for part in str(item).split(",") if part.strip())
    if not values:
        raise ValueError("search_params 不能为空。")
    if "all" in values:
        if len(values) != 1:
            raise ValueError("search_params='all' 不能与具体参数同时使用。")
        return SEARCHABLE_PARAMS

    unknown = sorted(set(values).difference(SEARCHABLE_PARAMS))
    if unknown:
        raise ValueError(
            f"未知 Optuna 搜索参数：{unknown}；可选值为 {list(SEARCHABLE_PARAMS)}。"
        )
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"Optuna 搜索参数重复：{duplicates}。")
    selected = set(values)
    return tuple(name for name in SEARCHABLE_PARAMS if name in selected)


def _parse_fixed_param_overrides(requested=None) -> dict:
    """把 ``名称=值`` 固定参数解析为规范顺序和训练参数类型。"""

    if requested is None:
        return {}
    if isinstance(requested, dict):
        entries = list(requested.items())
    else:
        if isinstance(requested, str):
            requested = (requested,)
        assignments = []
        for item in requested:
            assignments.extend(
                part.strip() for part in str(item).split(",") if part.strip()
            )
        entries = []
        for assignment in assignments:
            if "=" not in assignment:
                raise ValueError(
                    f"固定参数 {assignment!r} 缺少 '='；应写成 参数名=值。"
                )
            name, raw_value = (part.strip() for part in assignment.split("=", 1))
            if not name or not raw_value:
                raise ValueError(f"固定参数格式非法：{assignment!r}。")
            entries.append((name, raw_value))

    names = [str(name) for name, _ in entries]
    unknown = sorted(set(names).difference(SEARCHABLE_PARAMS))
    if unknown:
        raise ValueError(
            f"未知 Optuna 固定参数：{unknown}；可选值为 {list(SEARCHABLE_PARAMS)}。"
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Optuna 固定参数重复：{duplicates}。")

    parsed = {}
    for name, raw_value in entries:
        try:
            value = int(raw_value) if name in INTEGER_SEARCH_PARAMS else float(raw_value)
        except (TypeError, ValueError) as exc:
            expected = "整数" if name in INTEGER_SEARCH_PARAMS else "浮点数"
            raise ValueError(f"固定参数 {name}={raw_value!r} 不是合法{expected}。") from exc
        if name in INTEGER_SEARCH_PARAMS and value <= 0:
            raise ValueError(f"固定参数 {name} 必须为正整数，实际为 {value}。")
        if name == "pos_max_freq" and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"固定参数 pos_max_freq 必须为有限正数，实际为 {value}。")
        if name == "p_drop" and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"固定参数 p_drop 必须在 [0, 1] 内，实际为 {value}。")
        if name in {"lr", "weight_decay"} and (
            not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"固定参数 {name} 必须为有限非负数，实际为 {value}。")
        parsed[name] = value
    return {name: parsed[name] for name in SEARCHABLE_PARAMS if name in parsed}


def _fixed_hyperparameters(search_params, requested=None) -> dict:
    """返回所有未搜索参数的实际固定值；未覆盖项使用训练默认值。"""

    selected = set(_resolve_search_params(search_params))
    overrides = _parse_fixed_param_overrides(requested)
    overlap = sorted(selected.intersection(overrides))
    if overlap:
        raise ValueError(f"参数不能同时搜索并固定：{overlap}。")
    defaults = train.parse_args([])
    return {
        name: overrides.get(name, getattr(defaults, name))
        for name in SEARCHABLE_PARAMS
        if name not in selected
    }


def _resolve_training_seeds(requested=None) -> tuple[int, ...]:
    """解析模型训练 seed；默认沿用训练入口的 seed。"""

    if requested is None:
        return (int(train.parse_args([]).seed),)
    if isinstance(requested, (str, int)):
        requested = (requested,)

    values = []
    for item in requested:
        values.extend(part.strip() for part in str(item).split(",") if part.strip())
    if not values:
        raise ValueError("training seeds 不能为空。")
    try:
        seeds = tuple(int(value) for value in values)
    except ValueError as exc:
        raise ValueError(f"training seeds 必须为整数，实际为 {values}。") from exc
    duplicates = sorted({seed for seed in seeds if seeds.count(seed) > 1})
    if duplicates:
        raise ValueError(f"training seeds 重复：{duplicates}。")
    invalid = [seed for seed in seeds if not 0 <= seed < 2**32]
    if invalid:
        raise ValueError(f"training seeds 必须在 [0, 2**32) 内，实际为 {invalid}。")
    return seeds


def _build_sampler(sampler_name, search_params, training_seeds, sampler_seed=None):
    """构造 Optuna sampler，并验证 GridSampler 的有限搜索空间。"""

    selected = _resolve_search_params(search_params)
    seeds = _resolve_training_seeds(training_seeds)
    if sampler_seed is not None and not 0 <= sampler_seed < 2**32:
        raise ValueError("sampler seed 必须在 [0, 2**32) 内。")
    if sampler_name == "tpe":
        if len(seeds) != 1:
            raise ValueError(
                "多个 training seed 只支持 grid sampler；"
                "避免 TPE 把随机 seed 当成可优化超参数。"
            )
        return optuna.samplers.TPESampler(seed=sampler_seed)
    if sampler_name != "grid":
        raise ValueError(f"未知 Optuna sampler：{sampler_name!r}。")

    continuous = [name for name in selected if name not in GRID_SEARCH_SPACES]
    if continuous:
        raise ValueError(
            "GridSampler 只能用于具有有限候选集合的参数；"
            f"当前不支持网格化：{continuous}。"
        )
    grid = {name: list(GRID_SEARCH_SPACES[name]) for name in selected}
    if len(seeds) > 1:
        grid["seed"] = list(seeds)
    return optuna.samplers.GridSampler(grid, seed=sampler_seed)


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


def _resolve_sampling_mode(requested: str, contract: dict) -> str:
    """默认采用 cache 为实验声明的训练抽样契约。"""

    configured = str(contract.get("training_sampling", "uniform"))
    sampling_mode = configured if requested == "auto" else requested
    if sampling_mode not in {"uniform", "run_time_balanced"}:
        raise RuntimeError(
            f"不支持的训练 sampling_mode={sampling_mode!r}。"
        )
    return sampling_mode


def _study_identity(
    data_exp: str,
    objective_mode: str,
    search_params=None,
    fixed_params=None,
    sampler_name="tpe",
    training_seeds=None,
    sampling_mode="uniform",
) -> str:
    """构造包含训练目标的 study 身份。"""

    selected = _resolve_search_params(search_params)
    fixed = _fixed_hyperparameters(selected, fixed_params)
    base = f"{data_exp}{objective_mode}_sfrconvnet_hpo_v{SEARCH_SPACE_VERSION}"
    if sampling_mode != "uniform":
        base += f"_{sampling_mode}"
    if selected == SEARCHABLE_PARAMS:
        identity = base
    else:
        identity = f"{base}_only_{'-'.join(selected)}"
    defaults = train.parse_args([])
    nondefault = {
        name: value
        for name, value in fixed.items()
        if value != getattr(defaults, name)
    }
    if nondefault:
        payload = json.dumps(nondefault, sort_keys=True, separators=(",", ":"))
        identity += f"_fixed_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:10]}"
    if sampler_name != "tpe":
        identity += f"_{sampler_name}"
    seeds = _resolve_training_seeds(training_seeds)
    default_seed = int(train.parse_args([]).seed)
    if seeds != (default_seed,):
        payload = json.dumps(seeds, separators=(",", ":"))
        identity += f"_seeds_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:10]}"
    return identity


def _bind_study_contract(
    study,
    *,
    data_exp,
    objective_mode,
    contract,
    loss,
    negative_sfr_weight,
    search_params=None,
    fixed_params=None,
    sampler_name="tpe",
    sampler_seed=None,
    training_seeds=None,
    sampling_mode="uniform",
):
    """将不可变语义写入 study，并拒绝恢复不兼容的数据库。"""

    selected = _resolve_search_params(search_params)
    expected = {
        "data_exp": data_exp,
        "objective_mode": objective_mode,
        "conditioning_variable": contract["conditioning_variable"],
        "conditioning_active": bool(contract["conditioning_active"]),
        "cache_metadata_sha256": contract["metadata_fingerprint"],
        "loss": loss,
        "negative_sfr_weight": float(negative_sfr_weight),
        "sampling_mode": sampling_mode,
        "search_space_version": SEARCH_SPACE_VERSION,
        "search_params": list(selected),
        "fixed_hyperparameters": _fixed_hyperparameters(selected, fixed_params),
        "sampler": sampler_name,
        "sampler_seed": sampler_seed,
        "training_seeds": list(_resolve_training_seeds(training_seeds)),
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
    batch_backend="tensor_cpu",
    write_model_graph=False,
    dataset_cache=None,
    search_params=None,
    fixed_params=None,
    sampler_name="tpe",
    training_seeds=None,
    sampling_mode="uniform",
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
    args.sampling_mode = sampling_mode
    args.batch_backend = batch_backend
    args.skip_model_graph = not write_model_graph

    selected = _resolve_search_params(search_params)
    fixed_hyperparameters = _fixed_hyperparameters(selected, fixed_params)
    seeds = _resolve_training_seeds(training_seeds)
    if len(seeds) == 1:
        args.seed = seeds[0]
    elif sampler_name == "grid":
        args.seed = trial.suggest_categorical("seed", list(seeds))
    else:
        raise ValueError("多个 training seed 只支持 grid sampler。")
    args.model_type = "convnet"
    if "base_ch" in selected:
        args.base_ch = trial.suggest_categorical(
            "base_ch", list(GRID_SEARCH_SPACES["base_ch"])
        )
    if "se_reduction" in selected:
        args.se_reduction = trial.suggest_categorical(
            "se_reduction", list(GRID_SEARCH_SPACES["se_reduction"])
        )
    if "pos_num_bands" in selected:
        args.pos_num_bands = trial.suggest_categorical(
            "pos_num_bands", list(GRID_SEARCH_SPACES["pos_num_bands"])
        )
    if "pos_max_freq" in selected:
        args.pos_max_freq = trial.suggest_categorical(
            "pos_max_freq", list(GRID_SEARCH_SPACES["pos_max_freq"])
        )
    if "p_drop" in selected:
        args.p_drop = trial.suggest_float("p_drop", 0.0, 0.5, step=0.1)
    if "lr" in selected:
        args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    if "weight_decay" in selected:
        args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    if "batch_size" in selected:
        args.batch_size = trial.suggest_categorical(
            "batch_size", list(GRID_SEARCH_SPACES["batch_size"])
        )
    for name, value in fixed_hyperparameters.items():
        setattr(args, name, value)

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
    performance_settings = {
        "batch_backend": batch_backend, "num_workers": int(num_workers),
        "reuse_datasets": dataset_cache is not None, "skip_model_graph": not write_model_graph,
    }
    trial.set_user_attr("performance_settings", performance_settings)
    trial.set_user_attr("search_params", list(selected))
    trial.set_user_attr("fixed_hyperparameters", fixed_hyperparameters)
    trial.set_user_attr("training_seed", int(args.seed))
    try:
        # cache 对象由本次 HPO 进程拥有，既不写磁盘，也不跨进程共享 CUDA tensor。
        result = float(train.main(args, dataset_cache=dataset_cache)) if dataset_cache is not None else float(train.main(args))
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
    parser.add_argument(
        "--sampling-mode",
        choices=("auto", "uniform", "run_time_balanced"),
        default="auto",
        help="auto 从 cache metadata 的 training_sampling 读取。",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-backend", choices=("standard", "tensor_cpu"), default="tensor_cpu",
                        help="默认整批 CPU 索引；standard 恢复普通 DataLoader。")
    parser.add_argument("--write-model-graph", action="store_true",
                        help="逐 trial 写 TensorBoard 模型图；默认只写标量以降低启动开销。")
    parser.add_argument("--no-reuse-datasets", action="store_true",
                        help="关闭同进程 train/val dataset 复用，每个 trial 重新读取 PT。")
    parser.add_argument(
        "--search-params",
        nargs="+",
        default=None,
        metavar="PARAM",
        help=(
            "只搜索指定超参数；支持空格或逗号分隔。省略或指定 all 时搜索全部。"
            f"可选值：{', '.join(SEARCHABLE_PARAMS)}。"
        ),
    )
    parser.add_argument(
        "--fixed-params",
        nargs="+",
        default=None,
        metavar="PARAM=VALUE",
        help=(
            "为未搜索的可选超参数指定固定值；支持空格或逗号分隔。"
            "未指定的固定参数使用 train.py 默认值。"
        ),
    )
    parser.add_argument(
        "--sampler",
        choices=("tpe", "grid"),
        default="tpe",
        help="tpe 为默认自适应采样；grid 穷举有限离散搜索空间。",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="控制 sampler 的抽样/网格顺序，不是模型训练 seed。",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        metavar="SEED",
        help=(
            "模型训练 seed，支持空格或逗号分隔；默认使用 train.py 的 seed。"
            "多个 seed 只支持 grid，并作为重复实验维度完整展开。"
        ),
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT, help="总时限，单位秒。")
    args = parser.parse_args(argv)
    if args.negative_sfr_weight < 0:
        parser.error("--negative-sfr-weight must be non-negative.")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative.")
    if args.batch_backend != "standard" and args.num_workers:
        parser.error("tensor_cpu 不使用 worker；请指定 --num-workers 0 或 --batch-backend standard。")
    if args.n_trials <= 0:
        parser.error("--n-trials must be positive.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    try:
        args.search_params = _resolve_search_params(args.search_params)
        args.fixed_params = _fixed_hyperparameters(
            args.search_params, args.fixed_params
        )
        args.seeds = _resolve_training_seeds(args.seeds)
        _build_sampler(
            args.sampler,
            args.search_params,
            args.seeds,
            args.sampler_seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    cli_args = _parse_args(argv)
    contract = _load_cache_contract(cli_args.cache_dir, cli_args.data_exp)
    objective_mode = _resolve_objective_mode(cli_args.objective_mode, contract)
    sampling_mode = _resolve_sampling_mode(cli_args.sampling_mode, contract)
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

    study_name = _study_identity(
        cli_args.data_exp,
        objective_mode,
        cli_args.search_params,
        cli_args.fixed_params,
        cli_args.sampler,
        cli_args.seeds,
        sampling_mode,
    )
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
    print(f"Search parameters: {', '.join(cli_args.search_params)}")
    print(f"Sampler: {cli_args.sampler} (seed={cli_args.sampler_seed})")
    print(f"Training seeds: {cli_args.seeds}")
    print(f"Training sampling: {sampling_mode}")
    fixed_hyperparameters = cli_args.fixed_params
    if fixed_hyperparameters:
        print(f"Fixed hyperparameters: {fixed_hyperparameters}")

    sampler = _build_sampler(
        cli_args.sampler,
        cli_args.search_params,
        cli_args.seeds,
        cli_args.sampler_seed,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
    )
    _bind_study_contract(
        study,
        data_exp=cli_args.data_exp,
        objective_mode=objective_mode,
        contract=contract,
        loss=cli_args.loss,
        negative_sfr_weight=cli_args.negative_sfr_weight,
        search_params=cli_args.search_params,
        fixed_params=cli_args.fixed_params,
        sampler_name=cli_args.sampler,
        sampler_seed=cli_args.sampler_seed,
        training_seeds=cli_args.seeds,
        sampling_mode=sampling_mode,
    )

    print("Starting optimization...")
    print(f"Performance: batch_backend={cli_args.batch_backend}, workers={cli_args.num_workers}, "
          f"reuse_datasets={not cli_args.no_reuse_datasets}, model_graph={cli_args.write_model_graph}")
    dataset_cache = None if cli_args.no_reuse_datasets else DatasetCache(max_entries=2)
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
            batch_backend=cli_args.batch_backend,
            write_model_graph=cli_args.write_model_graph,
            dataset_cache=dataset_cache,
            search_params=cli_args.search_params,
            fixed_params=cli_args.fixed_params,
            sampler_name=cli_args.sampler,
            training_seeds=cli_args.seeds,
            sampling_mode=sampling_mode,
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
