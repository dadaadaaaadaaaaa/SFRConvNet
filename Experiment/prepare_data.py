'''
1. CachePt数据预处理，整合raw_data_processing中的功能对数据进行预处理
2. 设计了4个实验，盘状星系、剔除一个星系的盘状，剔除低分辨率的盘状、同时剔除低分辨率和一个星系的盘状，测试模型的泛化能力
'''

import argparse
import json
import os
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
        y_baseline = transform_sfr(
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
                # data.mbh,
                y_baseline,
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
    }
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
        help="Prepare only Exp1 caches; useful for feature-ablation runs.",
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

    exp_all_folder_paths = exp_eg_folder_paths + exp_dg_folder_paths # 所有星系的路径

    # ==========================================
    # Modifications for SFR Architecture
    # ==========================================
    # As SFR (Star Formation Rate) physics model is highly relevant and tailored to Disk Galaxies with gas disks.
    # We will exclude Elliptical Galaxies from standard training pools to prevent physical domain confusion.

    # 1. Main Default Experiment (Exp1_) - Disk Galaxies Only (Fiducial setup)
    print("Processing Experiment 1 (Disk Galaxies)...")
    save_path = os.path.join(".cache", _cache_prefix(1, args.cache_tag))
    cache = CachePt().load(exp_dg_folder_paths, **load_options)
    train_cache, val_cache, test_cache = cache.split(
        train_size=0.6,
        validation_size=0.2,
        stratify="groups",
        random_state=args.seed,
    )
    
    train_cache = train_cache.process(balance=train_balance)
    val_cache = val_cache.process()
    test_cache = test_cache.process()
    
    print(f"  Training set size:   {len(train_cache)}")
    print(f"  Validation set size: {len(val_cache)}")
    print(f"  Testing set size:    {len(test_cache)}")

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
        **transform_options,
    )
    val_cache.to_pt(save_path + "val.pt", mean, std, **transform_options)
    test_cache.to_pt(save_path + "test.pt", mean, std, **transform_options)

    if args.only_exp1:
        print("[INFO] --only-exp1 requested; skipping Experiments 2-4.")
        sys.exit(0)

    import random

    # 2. Generalization Experiment (Exp2_) - In/Out Sample on Disk Galaxies
    # (Hold out one disk galaxy to test generalization)
    exp_dg_folder_paths_in = exp_dg_folder_paths.copy()
    
    if len(exp_dg_folder_paths_in) > 1:
        # 随机剔除一个盘星系作为留出集（设定固定随机种子以保证复现性与缓存一致）
        random.seed(42)
        held_out_idx = random.randint(0, len(exp_dg_folder_paths_in) - 1)
        held_out_dg = exp_dg_folder_paths_in.pop(held_out_idx)
        exp_dg_folder_paths_out = [held_out_dg]
        
        print(f"Processing Experiment 2 (Disk Galaxies Generalization: Hold-out test)...")
        print(f"  Hold-out Galaxy: {held_out_dg[1]}")
        save_path = os.path.join(".cache", _cache_prefix(2, args.cache_tag))
        
        try:
            cache_in = CachePt().load(exp_dg_folder_paths_in, **load_options)
            train_cache, val_cache, test_cache = cache_in.split(
                train_size=0.6,
                validation_size=0.2,
                stratify="groups",
                random_state=args.seed,
            )

            train_cache = train_cache.process(balance=train_balance)
            val_cache = val_cache.process()
            test_cache = test_cache.process()
            
            print(f"  Training set size:   {len(train_cache)}")
            print(f"  Validation set size: {len(val_cache)}")
            print(f"  Testing set size:    {len(test_cache)}")

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
                **transform_options,
            )
            val_cache.to_pt(save_path + "val.pt", mean, std, **transform_options)
            test_cache.to_pt(save_path + "test.pt", mean, std, **transform_options)

            # Process out-of-distribution (held out) test set
            cache_out = CachePt().load(exp_dg_folder_paths_out, **load_options)
            cache_out = cache_out.process()
            print(f"  Out-set size (Generalization): {len(cache_out)}")
            cache_out.to_pt(save_path + "out.pt", mean, std, **transform_options)
        except ValueError as e:
            print(f"  [Skip] Skipped Experiment 2: {e}")
    else:
        print("[Skip] Not enough Disk Galaxies to do hold-out Exp2_")

    # Create disk galaxies without low-res data
    exp_dg_folder_paths_nolow = exp_dg_folder_paths.copy() # 实验3：测试强行剔除低分辨率数据后，模型预测的精度会不会有提升
    try:
        if len(exp_dg_folder_paths_nolow) > 5:
            exp_dg_folder_paths_nolow.pop(5)
        print(f"Processing Experiment 3...")
        save_path = os.path.join(".cache", _cache_prefix(3, args.cache_tag))
        cache = CachePt().load(exp_dg_folder_paths_nolow, **load_options)
        train_cache, val_cache, test_cache = cache.split(
            train_size=0.6,
            validation_size=0.2,
            stratify="groups",
            random_state=args.seed,
        )
        train_cache = train_cache.process(balance=train_balance)
        val_cache = val_cache.process()
        test_cache = test_cache.process()
        print(f"  Training set size:   {len(train_cache)}")
        print(f"  Validation set size: {len(val_cache)}")
        print(f"  Testing set size:    {len(test_cache)}")
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
            **transform_options,
        )
        val_cache.to_pt(save_path + "val.pt", mean, std, **transform_options)
        test_cache.to_pt(save_path + "test.pt", mean, std, **transform_options)
    except Exception as e:
        print(f"  [Skip] Skipped Experiment 3: {e}")

    # Create in- and out-datasets for disk galaxies without low-res data
    print(f"Processing Experiment 4...") # 实验4：在实验2的基础上，测试强行剔除低分辨率数据后，模型预测的精度会不会有提升
    save_path = os.path.join(".cache", _cache_prefix(4, args.cache_tag))
    exp_dg_folder_paths_nolow_in = exp_dg_folder_paths_nolow.copy()
    
    try:
        exp_dg_folder_paths_nolow_in.pop(2)
        exp_dg_folder_paths_nolow_out = [exp_dg_folder_paths_nolow[2]]
        
        cache_in = CachePt().load(exp_dg_folder_paths_nolow_in, **load_options)
        train_cache, val_cache, test_cache = cache_in.split(
            train_size=0.6,
            validation_size=0.2,
            stratify="groups",
            random_state=args.seed,
        )
        train_cache = train_cache.process(balance=train_balance)
        val_cache = val_cache.process()
        test_cache = test_cache.process()
        print(f"  Training set size:   {len(train_cache)}")
        print(f"  Validation set size: {len(val_cache)}")
        print(f"  Testing set size:    {len(test_cache)}")
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
            **transform_options,
        )
        val_cache.to_pt(save_path + "val.pt", mean, std, **transform_options)
        test_cache.to_pt(save_path + "test.pt", mean, std, **transform_options)

        cache_out = CachePt().load(exp_dg_folder_paths_nolow_out, **load_options)
        cache_out = cache_out.process()
        print(f"  Out-set size: {len(cache_out)}")
        cache_out.to_pt(save_path + "out.pt", mean, std, **transform_options)
    except Exception as e:
        print(f"  [Skip] Skipped Experiment 4: {e}")
