'''
1. CachePt数据预处理，整合raw_data_processing中的功能对数据进行预处理
2. 设计了7个实验，椭圆星系、盘状星系、所有星系、剔除一个星系的椭圆、盘状，剔除低分辨率的椭圆、同时剔除低分辨率和一个星系的盘状，测试模型的泛化能力
'''

import os
import sys
import torch
import numpy as np

print(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raw_data_processing.galaxy_data import GalDataSet
from raw_data_processing.parse_log_file import load_config


class CachePt: # 特征过滤、数据集划分、镜像扩增、样本平衡、计算均值和方差、归一化、保存为.pt文件
    def __init__(self, dataset: GalDataSet = None):
        self._dataset = dataset

    # ------------ Properties -----------

    @property
    def dataset(self):
        return self._dataset

    # ------------ Dunder Methods ------------

    def __len__(self): # 通过调用time，返回数据集一共有多少个snapshot
        return len(self._dataset.raw_data["time"])

    def __add__(self, other: "CachePt") -> "CachePt": # 拼接dataset
        if not isinstance(other, CachePt):
            raise TypeError("Can only add CachePt to another CachePt")
        combined_dataset = self._dataset + other._dataset
        return CachePt(combined_dataset)

    # ------------ Methods ------------

    def load(self, path: str):
        self._dataset = GalDataSet().load_data(path) # 读取数据并创建GalDataSet对象
        self._dataset.drop_invalid() # 去除无效数据（如0或负数或过大）
        return self

    def split( # 用statify把数据划分为训练集、验证集和测试集
        self,
        train_size: float,
        validation_size: float,
        stratify: str | np.ndarray | None = None,
    ):
        train_set, val_set, test_set = self._dataset.split(
            train_size=train_size, validation_size=validation_size, stratify=stratify
        )
        return CachePt(train_set), CachePt(val_set), CachePt(test_set)

    def process(self, threshold: float = None, mirror: bool = True, balance: str = None):
        data = self._dataset
        if threshold is not None:
            data.filter(threshold) # 过滤掉吸积率小于threshold的样本
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
    ):
        data = self._dataset
        if mean is not None and std is not None: # 验证集&测试集
            x = data.standardize(mean, std)
        else: # 训练集
            x, mean, std = data.standardize()
        torch.save(
            (
                x,
                data.coordinate,
                data.y,
                data.mbh,
                data.y_baseline,
                data.groups,
                data.n_groups,
                data.time,
            ),
            save_path,
        )
        return mean, std


def _save_norm_stats(save_path: str, mean: np.ndarray, std: np.ndarray): # 接收CachePt处理训练集时算出来的mean和std，打包保存成npz文件
    norm_path = f"{save_path}norm.npz"
    np.savez(norm_path, mean=mean, std=std)
    print(f"[INFO] Saved normalization stats: {norm_path}")


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
    resolution = "coarse"
    configs_dir = ".config"

    # Use the global variables (renamed to match notebook import expectation)
    exp_eg_folder_paths = Exp_eg_folder_paths # 特定分辨率的椭圆星系的路径
    exp_dg_folder_paths = Exp_dg_folder_paths # 特定分辨率的盘状星系的路径

    exp_all_folder_paths = exp_eg_folder_paths + exp_dg_folder_paths # 所有星系的路径

    # Create each type galaxy cache
    for i, path in enumerate([exp_eg_folder_paths, exp_dg_folder_paths, exp_all_folder_paths]): # 进行实验1-3，分别训练椭圆星系、盘状星系和所有星系
        print(f"Processing Experiment {i+1}...")
        save_path = os.path.join(".cache", f"Exp{i+1}_")

        cache = CachePt().load(path)
        train_cache, val_cache, test_cache = cache.split(train_size=0.6, validation_size=0.2, stratify="groups")

        train_cache = train_cache.process(balance="oversample")
        val_cache = val_cache.process()
        test_cache = test_cache.process()
        print(f"  Training set size:   {len(train_cache)}")
        print(f"  Validation set size: {len(val_cache)}")
        print(f"  Testing set size:    {len(test_cache)}")

        mean, std = train_cache.to_pt(save_path + "train.pt")
        _save_norm_stats(save_path, mean, std)
        val_cache.to_pt(save_path + "val.pt", mean, std)
        test_cache.to_pt(save_path + "test.pt", mean, std)

    # Create in- and out-datasets for each galaxy type
    exp_eg_folder_paths_in = exp_eg_folder_paths.copy()
    exp_dg_folder_paths_in = exp_dg_folder_paths.copy()
    exp_eg_folder_paths_in.pop(2) # 故意把某个椭圆星系的数据从训练集中剔除，最后用这个星系的数据作为测试集，验证模型的泛化能力
    exp_dg_folder_paths_in.pop(2) # 与上面同理
    exp_eg_folder_paths_out = [exp_eg_folder_paths[2]]
    exp_dg_folder_paths_out = [exp_dg_folder_paths[2]]

    for i, (in_paths, out_path) in enumerate( # 进行实验4和5，分别训练椭圆星系、盘状星系和所有星系，且每种类型都剔除一个星系作为测试集
        [
            (exp_eg_folder_paths_in, exp_eg_folder_paths_out),
            (exp_dg_folder_paths_in, exp_dg_folder_paths_out),
        ]
    ):
        print(f"Processing Experiment {i+4}...")
        save_path = os.path.join(".cache", f"Exp{i+4}_")

        try:
            cache_in = CachePt().load(in_paths)
            train_cache, val_cache, test_cache = cache_in.split(train_size=0.6, validation_size=0.2, stratify="groups")

            train_cache = train_cache.process(balance="oversample")
            val_cache = val_cache.process()
            test_cache = test_cache.process()
            print(f"  Training set size:   {len(train_cache)}")
            print(f"  Validation set size: {len(val_cache)}")
            print(f"  Testing set size:    {len(test_cache)}")

            mean, std = train_cache.to_pt(save_path + "train.pt")
            _save_norm_stats(save_path, mean, std)
            val_cache.to_pt(save_path + "val.pt", mean, std)
            test_cache.to_pt(save_path + "test.pt", mean, std)

            cache_out = CachePt().load(out_path)

            cache_out = cache_out.process()
            print(f"  Out-set size: {len(cache_out)}")

            cache_out.to_pt(save_path + "out.pt", mean, std)
        except ValueError as e:
            print(f"  [Skip] Skipped Experiment {i+4}: {e}")

    # Create disk galaxies without low-res data
    exp_dg_folder_paths_nolow = exp_dg_folder_paths.copy() # 实验6：测试强行剔除低分辨率数据后，模型预测的精度会不会有提升
    exp_dg_folder_paths_nolow.pop(5)
    print(f"Processing Experiment 6...")
    save_path = os.path.join(".cache", f"Exp6_")
    cache = CachePt().load(exp_dg_folder_paths_nolow)
    train_cache, val_cache, test_cache = cache.split(train_size=0.6, validation_size=0.2, stratify="groups")
    train_cache = train_cache.process(balance="oversample")
    val_cache = val_cache.process()
    test_cache = test_cache.process()
    print(f"  Training set size:   {len(train_cache)}")
    print(f"  Validation set size: {len(val_cache)}")
    print(f"  Testing set size:    {len(test_cache)}")
    mean, std = train_cache.to_pt(save_path + "train.pt")
    _save_norm_stats(save_path, mean, std)
    val_cache.to_pt(save_path + "val.pt", mean, std)
    test_cache.to_pt(save_path + "test.pt", mean, std)

    # Create in- and out-datasets for disk galaxies without low-res data
    print(f"Processing Experiment 7...") # 实验7：在实验4的基础上，测试强行剔除低分辨率数据后，模型预测的精度会不会有提升
    save_path = os.path.join(".cache", f"Exp7_")
    exp_dg_folder_paths_nolow_in = exp_dg_folder_paths_nolow.copy()
    exp_dg_folder_paths_nolow_in.pop(2)
    exp_dg_folder_paths_nolow_out = [exp_dg_folder_paths_nolow[2]]
    cache_in = CachePt().load(exp_dg_folder_paths_nolow_in)
    train_cache, val_cache, test_cache = cache_in.split(train_size=0.6, validation_size=0.2, stratify="groups")
    train_cache = train_cache.process(balance="oversample")
    val_cache = val_cache.process()
    test_cache = test_cache.process()
    print(f"  Training set size:   {len(train_cache)}")
    print(f"  Validation set size: {len(val_cache)}")
    print(f"  Testing set size:    {len(test_cache)}")
    mean, std = train_cache.to_pt(save_path + "train.pt")
    _save_norm_stats(save_path, mean, std)
    val_cache.to_pt(save_path + "val.pt", mean, std)
    test_cache.to_pt(save_path + "test.pt", mean, std)

    cache_out = CachePt().load(exp_dg_folder_paths_nolow_out)
    cache_out = cache_out.process()
    print(f"  Out-set size: {len(cache_out)}")
    cache_out.to_pt(save_path + "out.pt", mean, std)