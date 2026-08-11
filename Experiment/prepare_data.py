'''
1. CachePt数据预处理，整合raw_data_processing中的功能对数据进行预处理
2. 设计了4个实验，盘状星系、剔除一个星系的盘状，剔除低分辨率的盘状、同时剔除低分辨率和一个星系的盘状，测试模型的泛化能力
'''

import argparse
import json
import os
import re
import sys
import torch
import numpy as np

print(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raw_data_processing.galaxy_data import GalDataSet
from raw_data_processing.parse_log_file import load_config
from Experiment.target_transform import SUPPORTED_TARGET_TRANSFORMS, transform_sfr


class CachePt: # 特征过滤、数据集划分、镜像扩增、样本平衡、计算均值和方差、归一化、保存为.pt文件
    def __init__(self, dataset: GalDataSet = None):
        self._dataset = dataset

    # ------------ Properties -----------

    @property
    def dataset(self):
        return self._dataset

    @property
    def input_channels(self):
        return self._dataset.input_channels

    # ------------ Dunder Methods ------------

    def __len__(self): # 通过调用time，返回数据集一共有多少个snapshot
        return len(self._dataset.raw_data["time"])

    def __add__(self, other: "CachePt") -> "CachePt": # 拼接dataset
        if not isinstance(other, CachePt):
            raise TypeError("Can only add CachePt to another CachePt")
        combined_dataset = self._dataset + other._dataset
        return CachePt(combined_dataset)

    # ------------ Methods ------------

    def load(
        self,
        path: str,
        *,
        include_channels=None,
        exclude_channels=None,
        include_zero_targets: bool = True,
        require_target_quantity: str | None = "instantaneous_sfr",
    ):
        self._dataset = GalDataSet().load_data(path)
        target_quantity = self._dataset.target_metadata.get("quantity")
        if require_target_quantity and target_quantity != require_target_quantity:
            raise ValueError(
                f"Processed data target is {target_quantity!r}, expected {require_target_quantity!r}. "
                "Regenerate fine/coarse HDF5 files from the raw hdfra snapshots before preparing caches."
            )
        self._dataset.select_input_channels(include=include_channels, exclude=exclude_channels)
        self._dataset.drop_invalid(
            require_positive_targets=not include_zero_targets,
            require_positive_ks=not include_zero_targets,
        )
        return self

    def split( # 用statify把数据划分为训练集、验证集和测试集
        self,
        train_size: float,
        validation_size: float,
        stratify: str | np.ndarray | None = None,
        random_state: int | None = None,
    ):
        train_set, val_set, test_set = self._dataset.split(
            train_size=train_size,
            validation_size=validation_size,
            stratify=stratify,
            random_state=random_state,
        )
        return CachePt(train_set), CachePt(val_set), CachePt(test_set)

    def process(self, threshold: float = None, mirror: bool = True, balance: str = None):
        data = self._dataset
        if threshold is not None:
            data.filter(threshold) # 过滤掉恒星形成率小于threshold的样本
        if mirror:
            data.mirror_data() # mirror，镜像扩增数据
        if balance is not None: # balance，过采样或欠采样平衡数据
            if balance not in ["oversample", "undersample"]:
                raise ValueError("Balance method must be 'oversample' or 'undersample'")
            data.balance_groups(method=balance)
        return CachePt(data)

    def to_pt(
        self,
        save_path: str,
        mean: np.ndarray | None = None, # 均值
        std: np.ndarray | None = None, # 方差，用于归一化
        *,
        target_transform: str = "log10_1p",
        target_scale: float = 1.0,
        target_floor: float = 1e-9,
    ):
        data = self._dataset
        if mean is not None and std is not None: # 验证集&测试集
            x = data.standardize(mean, std)
        else: # 训练集
            x, mean, std = data.standardize()
        y = transform_sfr(
            data.sfr_target,
            transform=target_transform,
            scale=target_scale,
            floor=target_floor,
        )
        # The fourth PT field is the transformed KS-law SFR. It is not mbh.
        # SFRConvNet may use this scalar as a FiLM conditioner when use_film=True.
        sfr_ks_transformed = transform_sfr(
            data.sfr_ks,
            transform=target_transform,
            scale=target_scale,
            floor=target_floor,
        )
        torch.save(
            (
                x,
                data.coordinate,
                y,
                sfr_ks_transformed,
                data.groups,
                data.n_groups,
                data.time,
            ),
            save_path,
        )
        return mean, std


def _save_norm_stats(
    save_path: str,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    input_channels=(),
    excluded_channels=(),
    target_metadata=None,
    target_transform="log10_1p",
    target_scale=1.0,
    target_floor=1e-9,
    includes_zero_targets=True,
    oversample_groups=True,
    split_metadata=None,
): # 接收CachePt处理训练集时算出来的mean和std，打包保存成npz文件
    norm_path = f"{save_path}norm.npz"
    channel_names = np.asarray(tuple(input_channels), dtype=str)
    excluded_names = np.asarray(tuple(excluded_channels), dtype=str)
    target_metadata = dict(target_metadata or {})
    metadata = {
        "format_version": 2,
        "input_channels": channel_names.tolist(),
        "excluded_channels": excluded_names.tolist(),
        "n_input_channels": int(len(channel_names)),
        "target_quantity": target_metadata.get("quantity", "instantaneous_sfr"),
        "target_name": target_metadata.get("name", "sfr"),
        "target_dataset": target_metadata.get("dataset", "Data-Set-11"),
        "target_reduction": target_metadata.get("reduction", "sum"),
        "target_units": target_metadata.get("units", "Msun/yr"),
        "target_transform": target_transform,
        "target_scale": float(target_scale),
        "target_floor": float(target_floor),
        "includes_zero_targets": bool(includes_zero_targets),
        "oversample_groups": bool(oversample_groups),
        "group_balance": "oversample" if oversample_groups else "none",
        # Describes the fourth tensor stored in every current SFR PT cache.
        "conditioning_variable": "sfr_ks",
        "conditioning_transform": target_transform,
    }
    if split_metadata:
        overlap = set(split_metadata.get("train_groups", ())).intersection(
            split_metadata.get("held_out_groups", ())
        )
        if overlap:
            raise ValueError(
                "Refusing to save invalid split metadata: train_groups and "
                f"held_out_groups overlap: {sorted(overlap)}"
            )
        metadata.update(split_metadata)
    np.savez(
        norm_path,
        mean=mean,
        std=std,
        input_channels=channel_names,
        excluded_channels=excluded_names,
        target_quantity=np.asarray(metadata["target_quantity"]),
        target_name=np.asarray(metadata["target_name"]),
        target_dataset=np.asarray(metadata["target_dataset"]),
        target_reduction=np.asarray(metadata["target_reduction"]),
        target_units=np.asarray(metadata["target_units"]),
        target_transform=np.asarray(metadata["target_transform"]),
        target_scale=np.asarray(metadata["target_scale"]),
        target_floor=np.asarray(metadata["target_floor"]),
        includes_zero_targets=np.asarray(metadata["includes_zero_targets"]),
        oversample_groups=np.asarray(metadata["oversample_groups"]),
        group_balance=np.asarray(metadata["group_balance"]),
        conditioning_variable=np.asarray(metadata["conditioning_variable"]),
        conditioning_transform=np.asarray(metadata["conditioning_transform"]),
    )
    metadata_path = f"{save_path}metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved normalization stats: {norm_path}")
    print(f"[INFO] Saved channel metadata: {metadata_path}")


def _parse_channel_list(value: str):
    return tuple(name.strip() for name in value.split(",") if name.strip())


def _parse_bool(value):
    """Parse explicit command-line booleans such as --oversample=false."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value (true/false), got {value!r}."
    )


def _parse_experiment_tokens(values):
    """Normalize CLI values such as ``exp1 exp2`` or ``exp1,exp2``."""
    normalized = []
    for value in values or ():
        for token in str(value).split(","):
            token = token.strip().lower()
            if not token:
                continue
            match = re.fullmatch(r"(?:exp)?([1-4])", token)
            if not match:
                raise argparse.ArgumentTypeError(
                    f"Unsupported experiment {token!r}; choose exp1, exp2, exp3, or exp4."
                )
            name = f"exp{match.group(1)}"
            if name not in normalized:
                normalized.append(name)
    return tuple(normalized)


def _slugify_group_name(group_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(group_name)).strip("_")
    if not slug:
        raise ValueError(f"Cannot construct a cache name from group {group_name!r}.")
    return slug


def _valid_group_paths(folder_paths):
    """Return configured groups that currently contain at least one HDF5 cache."""
    valid = []
    skipped = []
    for entry in folder_paths:
        path, group_name = entry
        has_h5 = os.path.isdir(path) and any(
            name.lower().endswith(".h5") for name in os.listdir(path)
        )
        if has_h5:
            valid.append([os.path.abspath(path), str(group_name)])
        else:
            skipped.append([os.path.abspath(path), str(group_name)])
    return valid, skipped


def _normalize_held_out_names(requested_names, valid_group_names):
    """Resolve nicknames (``fiducial_4``) or full group names unambiguously."""
    valid_group_names = tuple(valid_group_names)
    resolved = []
    for requested in requested_names or ():
        requested = str(requested).strip()
        if not requested:
            continue
        candidates = [requested]
        if not requested.startswith("disk_galaxy_"):
            candidates.append(f"disk_galaxy_{requested}")
        match = next((name for name in candidates if name in valid_group_names), None)
        if match is None:
            raise ValueError(
                f"Requested held-out galaxy {requested!r} is not available. "
                f"Valid groups: {list(valid_group_names)}"
            )
        if match not in resolved:
            resolved.append(match)
    return tuple(resolved)


def _cache_groups(cache: CachePt) -> tuple[str, ...]:
    return tuple(sorted({str(group) for group in cache.dataset.groups.tolist()}))


def _assert_logo_partition(train_cache, val_cache, test_cache, out_cache, *, train_groups, held_out_group):
    """Adversarial split check using the actual samples, not only requested paths."""
    expected_train = set(train_groups)
    expected_out = {held_out_group}
    actual = {
        "train": set(_cache_groups(train_cache)),
        "val": set(_cache_groups(val_cache)),
        "test": set(_cache_groups(test_cache)),
        "out": set(_cache_groups(out_cache)),
    }
    for split in ("train", "val", "test"):
        if actual[split] != expected_train:
            raise RuntimeError(
                f"Invalid leave-one-galaxy-out fold: {split} groups {sorted(actual[split])} "
                f"do not equal expected in-domain groups {sorted(expected_train)}."
            )
    if actual["out"] != expected_out:
        raise RuntimeError(
            f"Invalid leave-one-galaxy-out fold: out groups {sorted(actual['out'])} "
            f"do not equal held-out group {held_out_group!r}."
        )
    overlap = expected_train.intersection(actual["out"])
    if overlap:
        raise RuntimeError(
            f"Data leakage detected: held-out groups overlap training groups: {sorted(overlap)}"
        )
    return actual


def _cache_prefix(experiment_number: int, cache_tag: str) -> str:
    tag = cache_tag.strip().strip("_")
    return f"Exp{experiment_number}_{tag}_" if tag else f"Exp{experiment_number}_"


def _parse_args():
    parser = argparse.ArgumentParser(description="Prepare SFR training caches.")
    parser.add_argument(
        "--exclude-channels",
        default="initial_stellar_density",
        help="Comma-separated input channels to exclude, for example: initial_stellar_density",
    )
    parser.add_argument(
        "--cache-tag",
        default="InstantSFR_NoInitialStellarDensity",
        help="Tag inserted in cache prefixes. Use InstantSFR with an empty exclusion list for the retained-channel control.",
    )
    parser.add_argument(
        "--target-transform",
        choices=SUPPORTED_TARGET_TRANSFORMS,
        default="log10_1p",
        help="Invertible transform applied identically to true SFR and the KS baseline.",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=1.0,
        help="SFR scale in Msun/yr for log10_1p.",
    )
    parser.add_argument(
        "--target-floor",
        type=float,
        default=1e-9,
        help="Positive floor used only by log10_floor.",
    )
    parser.add_argument(
        "--drop-zero-targets",
        action="store_true",
        help="Drop zero-SFR samples. By default zero-SFR snapshots are retained.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for dataset splitting.")
    parser.add_argument(
        "--only-exp1",
        action="store_true",
        help="Deprecated compatibility alias for --experiments exp1.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        metavar="EXP",
        help=(
            "Experiments to prepare, for example --experiments exp2 or "
            "--experiments exp1 exp2. Defaults to exp1-exp4 for backward compatibility."
        ),
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        metavar="EXP",
        help="Repeatable singular alias, for example --experiment exp3 --experiment exp4.",
    )
    parser.add_argument(
        "--held-out-galaxy",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Limit Exp2 folds to one or more galaxies. Accepts a nickname such as "
            "fiducial_4 or a full name such as disk_galaxy_fiducial_4. "
            "Omit to generate one fold for every available disk galaxy."
        ),
    )
    parser.add_argument(
        "--oversample",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        metavar="BOOL",
        help=(
            "Balance training groups by oversampling (default: true). "
            "Accepts --oversample=true or --oversample=false."
        ),
    )
    args = parser.parse_args()
    args.exclude_channels = _parse_channel_list(args.exclude_channels)
    if args.target_scale <= 0:
        parser.error("--target-scale must be positive.")
    if args.target_floor <= 0:
        parser.error("--target-floor must be positive.")
    if args.exclude_channels and not args.cache_tag:
        parser.error("--cache-tag is required when --exclude-channels is used, to avoid overwriting old caches.")
    try:
        requested = _parse_experiment_tokens((args.experiments or []) + (args.experiment or []))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.only_exp1:
        if requested and requested != ("exp1",):
            parser.error("--only-exp1 cannot be combined with experiments other than exp1.")
        requested = ("exp1",)
    args.experiments = requested or ("exp1", "exp2", "exp3", "exp4")
    return args


def get_folder_paths(configs_path, gal_type, resolution): # 读取config文件，返回指定类型和分辨率的文件夹绝对路径列表。这样代码路径就直接从config文件中读取，不需要硬编码在代码里了。
    configs = load_config(configs_path)
    data_dir = configs["BaseConfig"]["data_dir"]
    raw_data_config = configs["RawDataConfig"]

    folder_abs_paths = []
    for gal_nickname in raw_data_config[gal_type]:
        gal_name = f"{gal_type}_{gal_nickname}"
        folder_abs_path = os.path.join(data_dir, gal_name, resolution)
        folder_abs_paths.append([folder_abs_path, gal_name])
    return folder_abs_paths


def _prepare_random_snapshot_experiment(
    experiment_number,
    folder_paths,
    *,
    args,
    load_options,
    transform_options,
    train_balance,
    split_label="random_snapshot_within_group",
):
    """Prepare the legacy in-domain 60/20/20 reference split."""
    valid_paths, skipped_paths = _valid_group_paths(folder_paths)
    if not valid_paths:
        raise ValueError(f"Experiment {experiment_number} has no valid HDF5 groups.")
    if skipped_paths:
        print(f"  [INFO] Ignoring {len(skipped_paths)} configured groups without processed HDF5 data.")

    cache = CachePt().load(valid_paths, **load_options)
    train_cache, val_cache, test_cache = cache.split(
        train_size=0.6,
        validation_size=0.2,
        stratify="groups",
        random_state=args.seed,
    )
    np.random.seed(args.seed + int(experiment_number) * 1000)
    train_cache = train_cache.process(balance=train_balance)
    val_cache = val_cache.process()
    test_cache = test_cache.process()

    group_names = [entry[1] for entry in valid_paths]
    split_groups = {
        "train": list(_cache_groups(train_cache)),
        "val": list(_cache_groups(val_cache)),
        "test": list(_cache_groups(test_cache)),
    }
    print(f"  Training set size:   {len(train_cache)}")
    print(f"  Validation set size: {len(val_cache)}")
    print(f"  Testing set size:    {len(test_cache)}")

    save_path = os.path.join(".cache", _cache_prefix(experiment_number, args.cache_tag))
    mean, std = train_cache.to_pt(save_path + "train.pt", **transform_options)
    _save_norm_stats(
        save_path,
        mean,
        std,
        input_channels=train_cache.input_channels,
        excluded_channels=args.exclude_channels,
        target_metadata=train_cache.dataset.target_metadata,
        includes_zero_targets=not args.drop_zero_targets,
        oversample_groups=args.oversample,
        split_metadata={
            "experiment": f"exp{experiment_number}",
            "split_strategy": split_label,
            "split_seed": int(args.seed),
            "oversample_seed": int(args.seed + int(experiment_number) * 1000),
            "available_groups": group_names,
            "train_groups": group_names,
            "held_out_groups": [],
            "split_groups": split_groups,
            "group_paths": {entry[1]: entry[0] for entry in valid_paths},
            "sample_counts": {
                "train": len(train_cache),
                "val": len(val_cache),
                "test": len(test_cache),
                "out": 0,
            },
            "cache_prefix": os.path.basename(save_path),
        },
        **transform_options,
    )
    val_cache.to_pt(save_path + "val.pt", mean, std, **transform_options)
    test_cache.to_pt(save_path + "test.pt", mean, std, **transform_options)
    return save_path


def _prepare_exp2_logo_folds(
    folder_paths,
    *,
    args,
    load_options,
    transform_options,
    train_balance,
):
    """Generate one cache quartet per held-out disk galaxy."""
    valid_paths, skipped_paths = _valid_group_paths(folder_paths)
    valid_groups = tuple(entry[1] for entry in valid_paths)
    if skipped_paths:
        print("  Ignoring configured groups without processed HDF5 data:")
        for path, group in skipped_paths:
            print(f"    - {group}: {path}")
    if len(valid_paths) < 2:
        raise ValueError(
            "Exp2 leave-one-galaxy-out requires at least two available processed galaxies; "
            f"found {list(valid_groups)}."
        )

    requested = _normalize_held_out_names(args.held_out_galaxy, valid_groups)
    held_out_groups = requested or valid_groups
    base_prefix = _cache_prefix(2, args.cache_tag)
    folds = []

    for held_out_group in held_out_groups:
        held_out_entry = next(entry for entry in valid_paths if entry[1] == held_out_group)
        in_entries = [entry for entry in valid_paths if entry[1] != held_out_group]
        in_groups = tuple(entry[1] for entry in in_entries)
        fold_index = valid_groups.index(held_out_group)
        fold_name = f"LOO_{_slugify_group_name(held_out_group)}"
        save_path = os.path.join(".cache", f"{base_prefix}{fold_name}_")

        print("\n" + "=" * 72)
        print(f"Processing Exp2 fold: {fold_name}")
        print(f"  In-domain groups: {list(in_groups)}")
        print(f"  Held-out group:   {held_out_group}")

        cache_in = CachePt().load(in_entries, **load_options)
        train_cache, val_cache, test_cache = cache_in.split(
            train_size=0.6,
            validation_size=0.2,
            stratify="groups",
            random_state=args.seed,
        )
        # A fold-specific seed makes manual single-fold preparation identical
        # to the same fold produced as part of an exhaustive run.
        oversample_seed = int(args.seed + fold_index)
        np.random.seed(oversample_seed)
        train_cache = train_cache.process(balance=train_balance)
        val_cache = val_cache.process()
        test_cache = test_cache.process()
        cache_out = CachePt().load([held_out_entry], **load_options).process()

        actual_groups = _assert_logo_partition(
            train_cache,
            val_cache,
            test_cache,
            cache_out,
            train_groups=in_groups,
            held_out_group=held_out_group,
        )
        print("  [PASS] Actual train/val/test groups are disjoint from the held-out group.")
        print(f"  Training set size:                 {len(train_cache)}")
        print(f"  Validation set size:               {len(val_cache)}")
        print(f"  In-distribution test set size:      {len(test_cache)}")
        print(f"  Held-out galaxy test set size:      {len(cache_out)}")

        mean, std = train_cache.to_pt(save_path + "train.pt", **transform_options)
        split_metadata = {
            "experiment": "exp2",
            "split_strategy": "leave_one_galaxy_out",
            "split_seed": int(args.seed),
            "fold_name": fold_name,
            "fold_index": fold_index,
            "n_folds": len(valid_groups),
            "oversample_seed": oversample_seed,
            "available_groups": list(valid_groups),
            "train_groups": list(in_groups),
            "in_domain_groups": list(in_groups),
            "in_distribution_test_groups": list(in_groups),
            "held_out_group": held_out_group,
            "held_out_groups": [held_out_group],
            "out_groups": [held_out_group],
            "split_groups": {key: sorted(value) for key, value in actual_groups.items()},
            "group_paths": {entry[1]: entry[0] for entry in valid_paths},
            "raw_in_domain_groups": list(in_groups),
            "raw_held_out_groups": [held_out_group],
            "sample_counts": {
                "train": len(train_cache),
                "val": len(val_cache),
                "test": len(test_cache),
                "out": len(cache_out),
            },
            "cache_prefix": os.path.basename(save_path),
        }
        _save_norm_stats(
            save_path,
            mean,
            std,
            input_channels=train_cache.input_channels,
            excluded_channels=args.exclude_channels,
            target_metadata=train_cache.dataset.target_metadata,
            includes_zero_targets=not args.drop_zero_targets,
            oversample_groups=args.oversample,
            split_metadata=split_metadata,
            **transform_options,
        )
        val_cache.to_pt(save_path + "val.pt", mean, std, **transform_options)
        test_cache.to_pt(save_path + "test.pt", mean, std, **transform_options)
        cache_out.to_pt(save_path + "out.pt", mean, std, **transform_options)
        folds.append(split_metadata)

    manifest_path = os.path.join(".cache", f"{base_prefix}folds.json")
    manifest = {
        "format_version": 1,
        "experiment": "exp2",
        "split_strategy": "leave_one_galaxy_out",
        "available_groups": list(valid_groups),
        "requested_held_out_groups": list(requested),
        "fold_count": len(folds),
        "folds": [
            {
                "fold_name": fold["fold_name"],
                "cache_prefix": fold["cache_prefix"],
                "train_groups": fold["train_groups"],
                "held_out_groups": fold["held_out_groups"],
            }
            for fold in folds
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"\n[INFO] Saved Exp2 fold manifest: {manifest_path}")
    return manifest_path


# Define global paths for external imports (e.g. by notebooks)
try:
    _resolution = "coarse"
    _configs_dir = ".config"
    Exp_eg_folder_paths = get_folder_paths(_configs_dir, "elliptical_galaxy", _resolution) # 特定分辨率的椭圆星系的路径
    Exp_dg_folder_paths = get_folder_paths(_configs_dir, "disk_galaxy", _resolution) # 特定分辨率的盘状星系的路径
except Exception as e:
    # This might happen if CWD is not project root during import
    Exp_eg_folder_paths = []
    Exp_dg_folder_paths = []


if __name__ == "__main__":
    args = _parse_args()
    excluded_channels = args.exclude_channels
    include_zero_targets = not args.drop_zero_targets
    load_options = {
        "exclude_channels": excluded_channels,
        "include_zero_targets": include_zero_targets,
        "require_target_quantity": "instantaneous_sfr",
    }
    transform_options = {
        "target_transform": args.target_transform,
        "target_scale": args.target_scale,
        "target_floor": args.target_floor,
    }
    train_balance = "oversample" if args.oversample else None
    np.random.seed(args.seed)
    os.makedirs(".cache", exist_ok=True)
    resolution = "coarse"
    configs_dir = ".config"

    # Use the global variables (renamed to match notebook import expectation)
    exp_eg_folder_paths = Exp_eg_folder_paths # 特定分辨率的椭圆星系的路径
    exp_dg_folder_paths = Exp_dg_folder_paths # 特定分辨率的盘状星系的路径

    print(f"Requested experiments: {', '.join(args.experiments)}")

    # Exp1 remains the in-domain random-snapshot reference experiment.
    if "exp1" in args.experiments:
        print("\nProcessing Experiment 1 (in-domain random snapshot split)...")
        _prepare_random_snapshot_experiment(
            1,
            exp_dg_folder_paths,
            args=args,
            load_options=load_options,
            transform_options=transform_options,
            train_balance=train_balance,
        )

    # Exp2 is now exhaustive leave-one-galaxy-out over available HDF5 groups.
    if "exp2" in args.experiments:
        print("\nProcessing Experiment 2 (leave-one-galaxy-out)...")
        _prepare_exp2_logo_folds(
            exp_dg_folder_paths,
            args=args,
            load_options=load_options,
            transform_options=transform_options,
            train_balance=train_balance,
        )

    # Preserve legacy Exp3 semantics while making selection explicit.
    if "exp3" in args.experiments:
        print("\nProcessing Experiment 3...")
        exp3_paths = [entry for entry in exp_dg_folder_paths if entry[1] != "disk_galaxy_low"]
        _prepare_random_snapshot_experiment(
            3,
            exp3_paths,
            args=args,
            load_options=load_options,
            transform_options=transform_options,
            train_balance=train_balance,
            split_label="random_snapshot_without_low_resolution_group",
        )

    # Preserve legacy Exp4's configured third-group holdout. Exp2 should be
    # preferred for new multi-fold generalization studies.
    if "exp4" in args.experiments:
        print("\nProcessing Experiment 4 (legacy single holdout)...")
        valid_nolow, _ = _valid_group_paths(
            [entry for entry in exp_dg_folder_paths if entry[1] != "disk_galaxy_low"]
        )
        if len(valid_nolow) < 3:
            raise ValueError(
                "Legacy Exp4 requires at least three available non-low-resolution groups."
            )
        legacy_args = argparse.Namespace(**vars(args))
        legacy_args.held_out_galaxy = [valid_nolow[2][1]]
        print(
            "  [INFO] Exp4 retains its historical third-available-group holdout. "
            "Use Exp2 for exhaustive leave-one-galaxy-out."
        )
        # Exp4 keeps its historical prefix, so prepare its one fold explicitly.
        held = valid_nolow[2]
        in_entries = [entry for entry in valid_nolow if entry[1] != held[1]]
        cache_in = CachePt().load(in_entries, **load_options)
        train_cache, val_cache, test_cache = cache_in.split(
            train_size=0.6,
            validation_size=0.2,
            stratify="groups",
            random_state=args.seed,
        )
        train_cache = train_cache.process(balance=train_balance)
        val_cache = val_cache.process()
        test_cache = test_cache.process()
        cache_out = CachePt().load([held], **load_options).process()
        actual_groups = _assert_logo_partition(
            train_cache,
            val_cache,
            test_cache,
            cache_out,
            train_groups=[entry[1] for entry in in_entries],
            held_out_group=held[1],
        )
        save_path = os.path.join(".cache", _cache_prefix(4, args.cache_tag))
        mean, std = train_cache.to_pt(save_path + "train.pt", **transform_options)
        _save_norm_stats(
            save_path,
            mean,
            std,
            input_channels=train_cache.input_channels,
            excluded_channels=excluded_channels,
            target_metadata=train_cache.dataset.target_metadata,
            includes_zero_targets=include_zero_targets,
            oversample_groups=args.oversample,
            split_metadata={
                "experiment": "exp4",
                "split_strategy": "legacy_single_group_holdout",
                "split_seed": int(args.seed),
                "available_groups": [entry[1] for entry in valid_nolow],
                "train_groups": [entry[1] for entry in in_entries],
                "held_out_group": held[1],
                "held_out_groups": [held[1]],
                "out_groups": [held[1]],
                "split_groups": {key: sorted(value) for key, value in actual_groups.items()},
                "group_paths": {entry[1]: entry[0] for entry in valid_nolow},
                "sample_counts": {
                    "train": len(train_cache), "val": len(val_cache),
                    "test": len(test_cache), "out": len(cache_out),
                },
                "cache_prefix": os.path.basename(save_path),
            },
            **transform_options,
        )
        val_cache.to_pt(save_path + "val.pt", mean, std, **transform_options)
        test_cache.to_pt(save_path + "test.pt", mean, std, **transform_options)
        cache_out.to_pt(save_path + "out.pt", mean, std, **transform_options)
