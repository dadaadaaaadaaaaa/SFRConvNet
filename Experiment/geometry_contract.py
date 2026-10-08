"""训练与结果分析共用的二维几何输入契约检查。"""

import numpy as np

from raw_data_processing.geometry_features import (
    SUPPORTED_GEOMETRY_INPUT_MODES,
    geometry_input_channels,
    geometry_input_contract,
)


def validate_geometry_contract(contract, input_channels, log_transform_mask=None):
    """检查固定单位、几何通道顺序及禁止重复 log1p 的约定。"""
    if contract is None:
        contract = geometry_input_contract("none")
    if not isinstance(contract, dict):
        raise RuntimeError("几何输入契约必须是结构化字典。")
    mode = contract.get("mode")
    if mode not in SUPPORTED_GEOMETRY_INPUT_MODES:
        raise RuntimeError(f"未知几何输入模式：{mode!r}")
    if contract != geometry_input_contract(mode):
        raise RuntimeError("几何输入契约与固定 kpc 单位、公式或变换约定不一致。")
    names = tuple(input_channels)
    expected = geometry_input_channels(mode)
    reserved = set(geometry_input_channels("absolute_radius_widths"))
    actual = tuple(name for name in names if name in reserved)
    if actual != expected:
        raise RuntimeError(f"几何通道与模式不一致：{actual!r}，预期 {expected!r}。")
    if expected:
        if log_transform_mask is None:
            raise RuntimeError("几何 cache 缺少完整的训练集 log1p 掩码。")
        mask = np.asarray(log_transform_mask, dtype=bool).reshape(-1)
        if len(mask) != len(names):
            raise RuntimeError("几何 cache 缺少完整的训练集 log1p 掩码。")
        if any(mask[names.index(name)] for name in expected):
            raise RuntimeError("已经 log10 的几何通道不得再次使用 log1p。")
    return contract


def validate_geometry_ablation(datasets, expected_mode=None):
    """训练入口检查各 split 契约一致，以及消融名称没有误用旧 cache。"""
    contracts = [dataset.geometry_input_contract for dataset in datasets.values()]
    if not contracts or any(value != contracts[0] for value in contracts[1:]):
        raise RuntimeError("train/validation/test 的几何输入契约不一致。")
    contract = contracts[0]
    if expected_mode is not None and contract["mode"] != expected_mode:
        raise RuntimeError(
            f"消融要求 geometry_input_mode={expected_mode!r}，"
            f"cache 实际为 {contract['mode']!r}；请先生成对应独立 cache。"
        )
    return contract
