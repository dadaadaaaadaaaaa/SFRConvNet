"""二维逐格 SFR 监督的纯数组契约；标签为每格积分贡献，不是 SFR 密度。"""

from hashlib import sha256

import numpy as np

from raw_data_processing.metadata_schema import canonical_json


CELL_SCHEMA = "macnet_cell_sfr_supervision"


def array_digest(values):
    """固定 dtype 和字节序，使不同平台得到相同的数组摘要。"""
    array = np.ascontiguousarray(values, dtype="<f8")
    return sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def validate_cell_arrays(target, baseline, closure, *, shape, totals):
    """核验逐格恒等式及三种总量；closure 抵消误差使用 Y/B 的量级。"""
    arrays = [np.asarray(a, dtype=np.float64) for a in (target, baseline, closure)]
    if len(shape) != 2 or any(a.shape != tuple(shape) for a in arrays):
        raise ValueError("逐格 Y/B/C 必须与二维气体场 shape 一致。")
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("逐格 Y/B/C 存在非有限值。")
    y, b, c = arrays
    if np.any(y < 0) or np.any(b < 0):
        raise ValueError("逐格 Y/B 必须非负；只有 closure 允许负值。")
    if not np.allclose(c, y - b, rtol=1e-12, atol=1e-12):
        raise ValueError("逐格监督违反 C_i=Y_i-B_i。")
    scale = max(float(np.sum(y)), float(np.sum(b)), 1.0)
    for name, array, total in zip(("Y", "B", "C"), arrays, totals):
        if not np.isfinite(total) or abs(float(np.sum(array)) - total) > 1e-11 * scale:
            raise ValueError(f"逐格 {name} 求和与原标量监督不一致。")
    return arrays


def build_cell_contract(metadata, target, products, *, shape, target_total):
    """将局部标签、原始身份和物理 baseline 绑定，保持原 sample_uid 不变。"""
    operator = metadata.get("derivation", {}).get("operator_id")
    if operator not in {"identity_fine", "spatial_block"}:
        raise ValueError("逐格监督当前仅支持 identity_fine 和 spatial_block；R_m 需输运权重。")
    target_meta = metadata.get("target", {})
    if target_meta.get("dataset") != "Data-Set-11" or target_meta.get("reduction") != "sum":
        raise ValueError("逐格监督必须来自 Data-Set-11 的每格积分 SFR，归约为 sum。")
    if target_meta.get("units") != "MACER_mass_code/Gyr":
        raise ValueError("逐格监督仅支持二维 MACER_mass_code/Gyr 单位。")
    if "data_compatible" not in products:
        raise ValueError("逐格监督缺少 data_compatible baseline。")
    modes = {}
    for mode, product in products.items():
        baseline = np.asarray(product["cell_sfr_baseline"], dtype=np.float64)
        closure = np.asarray(target, dtype=np.float64) - baseline
        validate_cell_arrays(target, baseline, closure, shape=shape,
                             totals=(target_total, product["baseline_total"], product["closure_true"]))
        modes[mode] = {
            "physics_product_uid": product["physics_product_uid"],
            "baseline_sha256": array_digest(baseline),
            "closure_sha256": array_digest(closure),
        }
    contract = {
        "schema_name": CELL_SCHEMA, "schema_version": 1,
        "quantity": "cell_integrated_instantaneous_sfr",
        "units": "MACER_mass_code/Gyr", "to_msun_per_yr": 0.025,
        "axis_order": ["theta", "r"], "shape": list(shape),
        "label_source": "Data-Set-11", "label_reduction": "sum_without_volume",
        "baseline_definition": "S(R(U_HR)); not R(S(U_HR))",
        "closure_definition": "C_i=Y_i-B_i",
        "is_model_input": False, "canonical_mode": "data_compatible",
        "identity": dict(metadata["identity"]), "operator_id": operator,
        "target_sha256": array_digest(target), "modes": modes,
    }
    contract["product_uid"] = "macnetcell:" + sha256(canonical_json(contract).encode()).hexdigest()
    return contract
