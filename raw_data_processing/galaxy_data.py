"""MACER snapshot 读取、粗粒化、SFR 监督产品与机器学习数据集的总入口。

本模块从 HDF4 和日志构造 ``GalData``，保存权威网格体积和完整 metadata，
统一调度 R_h/R_m 算子并生成 ``Y、B、C_true``；``GalDataSet`` 则负责多
snapshot 加载、通道选择、标准化、分组及划分。作者：Peng Cheng。
"""

import os
import warnings

import h5py
import numpy as np

from typing import Callable, List, Tuple

try:
    from raw_data_processing.metadata_schema import (
        build_derived_metadata,
        build_fine_metadata,
        build_physics_product_uid,
        metadata_from_json,
        metadata_to_json,
        sha256_file,
        validate_complete_metadata,
    )
except ModuleNotFoundError:  # 兼容直接运行 raw_data_processing/galaxy_data.py。
    from metadata_schema import (
        build_derived_metadata,
        build_fine_metadata,
        build_physics_product_uid,
        metadata_from_json,
        metadata_to_json,
        sha256_file,
        validate_complete_metadata,
    )

try:
    from raw_data_processing.coarsening_common import (
        GAS_INTERNAL_ENERGY_DENSITY_DATASET,
        GAS_INTERNAL_ENERGY_DENSITY_FIELD,
        cooling_rate_density_from_time,
        temperature_from_internal_energy_density,
    )
except ModuleNotFoundError:  # 兼容直接运行 raw_data_processing/galaxy_data.py。
    from coarsening_common import (
        GAS_INTERNAL_ENERGY_DENSITY_DATASET,
        GAS_INTERNAL_ENERGY_DENSITY_FIELD,
        cooling_rate_density_from_time,
        temperature_from_internal_energy_density,
    )

try:
    from pyhdf.SD import SD
except ImportError:  # Processed HDF5 files and unit tests do not require HDF4.
    SD = None


def _validate_file_path(file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    return file_path


def _validate_macer_thermodynamic_correspondence(correspondence: dict) -> None:
    """确保旧字段名 ``gas_energy`` 只对应 Data-Set-10 内能密度。"""

    sources = [
        original_name
        for original_name, new_name in correspondence.items()
        if new_name == GAS_INTERNAL_ENERGY_DENSITY_FIELD
    ]
    if sources != [GAS_INTERNAL_ENERGY_DENSITY_DATASET]:
        raise ValueError(
            "MACER 'gas_energy' means gas internal-energy density and must map "
            f"exclusively from {GAS_INTERNAL_ENERGY_DENSITY_DATASET}; got {sources}. "
            "Data-Set-6 is temperature, not internal-energy density."
        )


def _validate_macer_dataset_labels(handle, datasets: set[str]) -> None:
    """用 HDF ``long_name`` 防止条件输出造成 Data-Set 编号整体错位。"""

    expected_tokens = {
        "Data-Set-5": ("DENSITY",),
        "Data-Set-8": ("A1",),
        "Data-Set-9": ("A2",),
        "Data-Set-10": ("GAS ENERGY",),
        "Data-Set-11": ("SFR",),
        "Data-Set-15": ("VOL",),
        "Data-Set-17": ("COOLT",),
    }
    for dataset_name in sorted(datasets.intersection(expected_tokens)):
        try:
            long_name = str(handle.select(dataset_name).attributes()["long_name"])
        except Exception as exc:
            raise ValueError(
                f"无法验证 {dataset_name} 的 HDF long_name；拒绝猜测字段编号。"
            ) from exc
        normalized = long_name.upper()
        if not any(token in normalized for token in expected_tokens[dataset_name]):
            raise ValueError(
                f"{dataset_name} 的 long_name={long_name!r} 与预期语义不符；"
                "该运行可能因条件输出导致 Data-Set 编号偏移。"
            )


def _validate_half_open_range(value, config_key: str) -> tuple[int, int]:
    """Validate a JSON-style ``[start, end]`` half-open integer range."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"'{config_key}' must be a two-item [start, end] range.")
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, np.integer))
        or not isinstance(end, (int, np.integer))
    ):
        raise ValueError(f"'{config_key}' boundaries must be integers, got {value!r}.")
    start, end = int(start), int(end)
    if start < 0 or start >= end:
        raise ValueError(f"'{config_key}' must satisfy 0 <= start < end, got {value!r}.")
    return start, end


def _snapshot_index_from_filename(file_name: str) -> int:
    """Extract the non-negative integer suffix from an ``hdfra.<index>`` name."""
    if not file_name.startswith("hdfra."):
        raise ValueError(f"Snapshot filename must start with 'hdfra.': {file_name!r}.")
    suffix = file_name.rsplit(".", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"Snapshot filename has a non-numeric index: {file_name!r}.")
    return int(suffix)


def _build_snapshot_schedule(hdfra_paths: List[str], gal_config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return physical times and output indices aligned with sorted snapshot paths.

    Supported modes:
    - ``sequence`` (the default): use every integer in ``range`` in order.
    - ``concatenate``: concatenate the integer intervals in ``ranges``.
    - ``filename_index``: recover the global index from each flattened filename.

    ``filename_index`` deliberately preserves gaps caused by missing snapshots;
    a file named ``hdfra.05065`` is always assigned index 5065 regardless of its
    position in the list.
    """
    if not hdfra_paths:
        raise ValueError("Cannot build a snapshot schedule for an empty file list.")

    try:
        snapshot_dt = float(gal_config["snapshot_dt"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("'snapshot_dt' must be provided as a positive finite number.") from exc
    try:
        offset = float(gal_config.get("offset", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("'offset' must be a finite number.") from exc
    if not np.isfinite(snapshot_dt) or snapshot_dt <= 0:
        raise ValueError(f"'snapshot_dt' must be positive and finite, got {snapshot_dt!r}.")
    if not np.isfinite(offset):
        raise ValueError(f"'offset' must be finite, got {offset!r}.")

    mode = gal_config.get("mode", "sequence")
    if mode in (None, "sequence", "sequential"):
        if "range" not in gal_config:
            raise ValueError("Sequential snapshot mode requires a 'range' entry.")
        start, end = _validate_half_open_range(gal_config["range"], "range")
        physical_indices = np.arange(start, end, dtype=np.int64)
        # Preserve the legacy output naming convention for non-flattened data.
        output_indices = np.arange(len(hdfra_paths), dtype=np.int64)
    elif mode == "concatenate":
        ranges = gal_config.get("ranges")
        if not isinstance(ranges, (list, tuple)) or not ranges:
            raise ValueError("Concatenate snapshot mode requires a non-empty 'ranges' list.")
        validated_ranges = [
            _validate_half_open_range(value, f"ranges[{i}]") for i, value in enumerate(ranges)
        ]
        physical_indices = np.concatenate(
            [np.arange(start, end, dtype=np.int64) for start, end in validated_ranges]
        )
        if len(np.unique(physical_indices)) != len(physical_indices):
            raise ValueError("Concatenate snapshot ranges overlap and produce duplicate physical indices.")
        output_indices = np.arange(len(hdfra_paths), dtype=np.int64)
    elif mode == "filename_index":
        if "ranges" in gal_config:
            raise ValueError("filename_index mode uses singular 'range', not 'ranges'.")
        if "range" not in gal_config:
            raise ValueError("filename_index snapshot mode requires a nominal 'range' entry.")
        start, end = _validate_half_open_range(gal_config["range"], "range")
        physical_indices = np.asarray(
            [_snapshot_index_from_filename(file_name) for file_name in hdfra_paths],
            dtype=np.int64,
        )
        if len(np.unique(physical_indices)) != len(physical_indices):
            raise ValueError("Flattened snapshot filenames contain duplicate global indices.")
        if np.any(np.diff(physical_indices) <= 0):
            raise ValueError("Flattened snapshot filenames must be sorted by increasing global index.")
        invalid = physical_indices[(physical_indices < start) | (physical_indices >= end)]
        if invalid.size:
            preview = ", ".join(str(value) for value in invalid[:5])
            raise ValueError(
                f"Snapshot indices lie outside the configured range [{start}, {end}): {preview}."
            )
        output_indices = physical_indices.copy()
    else:
        raise ValueError(
            f"Unsupported snapshot mode {mode!r}; use 'sequence', 'concatenate', or 'filename_index'."
        )

    if len(physical_indices) != len(hdfra_paths):
        raise ValueError(
            "Snapshot count does not match its configured time indices: "
            f"found {len(hdfra_paths)} files but generated {len(physical_indices)} indices. "
            "Use filename_index mode for flattened data with missing snapshots."
        )
    time_seq = physical_indices.astype(np.float64) * snapshot_dt + offset
    if not np.all(np.isfinite(time_seq)):
        raise ValueError("Computed snapshot times contain NaN or infinity.")
    return time_seq, output_indices


def _snapshot_output_name(snapshot_index: int) -> str:
    """Return the processed HDF5 filename for a logical/global snapshot index."""
    snapshot_index = int(snapshot_index)
    if snapshot_index < 0:
        raise ValueError(f"Snapshot output index must be non-negative, got {snapshot_index}.")
    return f"{snapshot_index:05d}.h5"


def _reject_stale_filename_index_outputs(folder_path: str, expected_names: set[str]) -> None:
    """Prevent a rerun from silently mixing old enumerated and global-index outputs."""
    existing_names = {
        name for name in os.listdir(folder_path) if name.lower().endswith(".h5")
    }
    unexpected = sorted(existing_names.difference(expected_names))
    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise RuntimeError(
            f"Output folder '{folder_path}' contains {len(unexpected)} stale/unexpected HDF5 files "
            f"({preview}). Move or remove that processed output before running filename_index mode."
        )


# =============== Galaxy Data Class ===============


class GalData:
    def __init__(self, ndim: int = 2, coordinate_mode: str = "polar"):
        """
        Initializes the GalData object.
        Args:
            ndim (int): Dimensionality of the data, either 2 or 3. Default is 2.
            coordinate_mode (str): Coordinate system mode, either 'polar'('p') or 'cartesian'('c'). Default is 'polar'.
        """
        coordinate_mode = coordinate_mode.lower()
        self._coordinate_mode = {
            "polar": "polar",
            "p": "polar",
            "cartesian": "cartesian",
            "c": "cartesian",
        }.get(coordinate_mode)
        if self._coordinate_mode is None:
            raise ValueError("Invalid coordinate mode. Choose 'polar' or 'cartesian'.")

        if ndim not in [2, 3]:
            raise ValueError("ndim must be either 2 or 3.")
        self.ndim = ndim
        self.time = None
        self._data_hdfra = None
        self._data_log = None
        self._target_metadata = {}
        self._sfr_products = {}
        self._corr = None
        self._axes = None
        self._axis_edges = None
        self._coord = None
        self._grid_volume = None
        self._volume_source = None
        self._coarsening_metadata = {}
        self._coarsening_diagnostics = {}
        self._metadata = {}
        self._pending_source_metadata = None

    # -------- Properties --------

    @property
    def snapshot(self) -> dict:
        if isinstance(self._data_hdfra, dict):
            data = dict(self._data_hdfra)
            if "cooling_time" in data and "cooling_rate_density" not in data:
                data["cooling_rate_density"] = cooling_rate_density_from_time(
                    data["gas_energy"], data["cooling_time"]
                )
            return data

        data = {}
        for original_name, new_name in self._corr.items():
            arr = self._data_hdfra.select(original_name)[:]
            data[new_name] = arr[0, ...]
        data["temperature"] = temperature_from_internal_energy_density(
            data["gas_energy"], data["density"]
        )
        if "cooling_time" in data:
            data["cooling_rate_density"] = cooling_rate_density_from_time(
                data["gas_energy"], data["cooling_time"]
            )
        return data

    @property
    def snapshot_array(self) -> np.ndarray:     # 把snapshot字典里的所有值提取出来打包成numpy多维数组，方便后续矩阵运算
        return np.array(list(self.snapshot.values()))

    @property
    def target_metadata(self) -> dict:
        return self._target_metadata.copy()

    @property
    def sfr_products(self) -> dict:
        """返回不含逐格数组的 SFR 监督产品摘要。"""

        summary = {}
        for mode, product in self._sfr_products.items():
            summary[mode] = {
                key: value
                for key, value in product.items()
                if key != "cell_sfr_baseline"
            }
        return metadata_from_json(metadata_to_json(summary)) if summary else {}

    @property
    def sfr_macer_baseline(self) -> float:
        """默认数据兼容 MACER 算子在当前分辨率上的总 SFR。"""

        return float(self._sfr_products["data_compatible"]["baseline_total"])

    @property
    def sfr_closure_true(self) -> float:
        """有符号真闭合项 ``Y-B``。"""

        return float(self._sfr_products["data_compatible"]["closure_true"])

    @property
    def coarsening_metadata(self) -> dict:
        """Serializable description of the coarse-graining operator."""
        return self._coarsening_metadata.copy()

    @property
    def coarsening_diagnostics(self) -> dict:
        """Diagnostic arrays kept separate from the model input channels."""
        return {
            name: np.asarray(values).copy()
            for name, values in self._coarsening_diagnostics.items()
        }

    @property
    def metadata(self) -> dict:
        """返回该 snapshot 的版本化身份、来源、几何和字段契约。"""

        return metadata_from_json(metadata_to_json(self._metadata)) if self._metadata else {}

    @property
    def parent_hr_uid(self) -> str | None:
        """返回所有同源粗粒化变体共享的 HR snapshot UID。"""

        return self._metadata.get("identity", {}).get("parent_hr_uid")

    @property
    def sample_uid(self) -> str | None:
        """返回当前具体数据产品的稳定 UID。"""

        return self._metadata.get("identity", {}).get("sample_uid")

    @property
    def raw_axes(self) -> list: # 返回原始的坐标轴数组。
        self.__validate_coords()
        return self._axes

    @property
    def broadcast_coords(self) -> np.ndarray: # 返回“广播”（Broadcast）后的坐标数组。这是一种节省内存的 NumPy 结构，用来对齐数据矩阵的形状。
        self.__validate_coords()
        return self._coord

    @property
    def coordinate(self) -> Tuple[np.ndarray, ...]: # 用 np.meshgrid 生成的完整的坐标网格矩阵。
        self.__validate_coords()
        return np.meshgrid(*self._axes)

    def axes(self) -> list: # 下面是极坐标系和直角坐标系的互相转换
        self.__validate_coords()
        return self._axes

    @property
    def r(self) -> np.ndarray: # 极坐标系下的r
        self.__validate_coords()
        if self._coordinate_mode == "polar":
            return self._coord[0]
        else:  # Cartesian -> spherical radius
            return np.sqrt(sum(c**2 for c in self._coord))

    @property
    def theta(self) -> np.ndarray: # 极坐标系下的theta
        self.__validate_coords()
        if self._coordinate_mode == "polar":
            return self._coord[1]
        else:
            if self.ndim == 2:
                return np.arctan2(self._coord[1], self._coord[0])
            else:
                return np.arccos(self._coord[2] / self.r)

    @property
    def phi(self) -> np.ndarray: # 极坐标系下的phi
        if self.ndim == 2:
            raise ValueError("Phi coordinate not available in 2D data.")
        self.__validate_coords()
        if self._coordinate_mode == "polar":
            return self._coord[2]
        else:
            return np.arctan2(self._coord[1], self._coord[0])

    @property
    def x(self) -> np.ndarray: # 直角坐标系下的x
        self.__validate_coords()
        if self._coordinate_mode == "cartesian":
            return self._coord[0]
        if self.ndim == 2:
            return self.r * np.sin(self.theta)
        else:
            return self.r * np.sin(self.theta) * np.cos(self.phi)

    @property
    def y(self) -> np.ndarray: # 直角坐标系下的y
        self.__validate_coords()
        if self._coordinate_mode == "cartesian":
            return self._coord[1]
        if self.ndim == 2:
            return self.r * np.cos(self.theta)
        else:
            return self.r * np.sin(self.theta) * np.sin(self.phi)

    @property
    def z(self) -> np.ndarray: # 直角坐标系下的z
        if self.ndim == 2:
            raise ValueError("Z coordinate not available in 2D data.")
        self.__validate_coords()
        if self._coordinate_mode == "cartesian":
            return self._coord[2]
        else:
            return self.r * np.cos(self.theta)

    @property
    def grid_volume(self) -> np.ndarray: # 计算单位网格的体积元dV
        """Authoritative per-cell volume supplied by MACER Data-Set-15.

        Cell volume must never be reconstructed from cell centres.  That is
        not conservative on MACER's logarithmic spherical-polar mesh and was
        the source of the legacy ``dr**2 * dtheta`` volume error.
        """
        if self._grid_volume is None:
            raise ValueError(
                "Authoritative grid volume is unavailable. Regenerate this processed HDF5 "
                "file from the raw HDFRA snapshot so Data-Set-15 is preserved."
            )
        return self._grid_volume

    @property
    def gas_mass(self) -> float: # 网格密度*网格体积=网格质量
        return self.snapshot["density"] * self.grid_volume

    # @property
    # def mbh(self) -> float: # 提取该snapshot的黑洞质量
    #     return self._data_log["mbh"]

    @property
    # def mdot_macer(self) -> float: # 提取吸积率
    #     return self._data_log["mdot_macer"]
    def sfr_target(self) -> float: # 提取恒星形成率
        return self._data_log["sfr_target"]

    # @property
    # def mdot_edd(self) -> float: # 提取爱丁顿吸积率
    #     return self._data_log["mdot_edd"]

    # -------- Calculation Methods --------

    # def mdot_bondi(self, r_acc: float = 1.0, G: float = 112.0, gamma: float = 5 / 3) -> float:
    #     """
    #     Calculates the Bondi accretion rate.
    #     Args:
    #         r_max (float): 计算恒星形成率的星系有效半径区域 (kpc)。默认 10.0 kpc。
    #         A_ks (float): KS 定律的经验常数，单位随质量系、引力单位而变。
    #         N_ks (float): KS 定律的幂指数，通常取 1.4 左右。
    #     """
    #     mask = self.r < r_max
    #     if not np.any(mask):
    #         r_values = np.sort(np.unique(self.r.reshape(-1)))
    #         idx = np.searchsorted(r_values, r_acc)
    #         if idx >= len(r_values):
    #             idx = len(r_values) - 1
    #         effective_r_acc = float(r_values[max(idx, 0)])
    #         mask = self.r <= effective_r_acc
    #
    #         if not np.any(mask):
    #             raise ValueError("No cells found within r_acc even after expanding to the smallest available radius.")
    #
    #         warnings.warn(
    #             (
    #                 "No grid cells lie within the requested r_acc=%.3f kpc; "
    #                 "using the innermost available radius at %.3f kpc instead."
    #             )
    #             % (r_acc, effective_r_acc),
    #             RuntimeWarning,
    #         )
    #
    #     # Broadcast mask to match the shape of density and other snapshot data
    #     density_shape = self.snapshot["density"].shape
    #     mask_broadcasted = np.broadcast_to(mask, density_shape)
    #
    #     rho = self.snapshot["density"][mask_broadcasted]
    #     e = self.snapshot["gas_energy"][mask_broadcasted]
    #     cs = np.sqrt((gamma * gamma1) * e / rho) # 计算声速
    #     m = self.gas_mass[mask_broadcasted]
    #     M_bh = self.mbh
    #
    #     # Weighted averages (default: mass-weighted)
    #     rho_inf = np.sum(m * rho) / np.sum(m)
    #     cs_inf = np.sum(m * cs) / np.sum(m)
    #
    #     return 4 * np.pi * lambda_c * G**2 * M_bh**2 * rho_inf / cs_inf**3 # 计算邦迪吸积率

    def sfr_ks(self, r_max: float = 1.0, A_ks: float = 0.906, N_ks: float = 1.4) -> float:
        """
        根据全局 Kennicutt-Schmidt (KS) 定律计算参考的恒星形成率。
        SFR_KS = A_ks * (Sigma_gas)^N_ks * Area
        Args:
            r_max (float): 计算恒星形成率的星系有效半径区域 (kpc)。默认 10.0 kpc。
            A_ks (float): KS 定律的经验常数，单位随质量系、引力单位而变。
            N_ks (float): KS 定律的幂指数，通常取 1.4 左右。
        """
        mask = self.r < r_max
        if not np.any(mask):
            return 0.0

        # 对于极坐标或柱坐标，r 的形状可能是 (Nr, 1)，通过 np.broadcast_to 确保 mask 形状和 gas_mass 完全一致
        # 以防出现 IndexError: boolean index did not match indexed array...
        mask = np.broadcast_to(mask, self.gas_mass.shape)
        
        # 取出所有给定半径部分的气体质量总和
        total_gas_mass = np.sum(self.gas_mass[mask])
        
        # 计算有效投影面积 (Area = pi * R_{eff}^2) 和气体表面密度 (Sigma_gas)
        area = np.pi * (r_max ** 2)
        sigma_gas = total_gas_mass / area
        
        # 计算全局 SFR (KS定律给出的是面密度，乘以面积得总SFR)
        sfr_estimated = A_ks * (sigma_gas ** N_ks) * area
        
        return float(sfr_estimated)

    def compute_macer_sfr_products(
        self,
        *,
        time_step: float | None = None,
        context_selection: str | None = None,
    ) -> "GalData":
        """计算两套 MACER SFR 基线，并构造当前样本的 ``Y、B、Y-B``。

        默认基线为数据兼容版本。``paper_physical`` 版本只作为物理对照保留。
        这些量保存在独立监督产品中，不会进入 ``snapshot`` 模型输入通道。
        """

        try:
            from raw_data_processing.macer_sfr_operator import calculate_macer_sfr
        except ModuleNotFoundError:
            from macer_sfr_operator import calculate_macer_sfr

        required_fields = {"density", "gas_energy", "cooling_time", "radial_acceleration"}
        missing_fields = sorted(required_fields.difference(self.snapshot))
        if missing_fields:
            raise KeyError(f"构造 MACER SFR baseline 缺少字段: {missing_fields}")
        if self._data_log is None or "mbh" not in self._data_log:
            raise KeyError("构造数据兼容 MACER SFR baseline 需要 log_data['mbh']。")
        if "sfr_target" not in self._data_log:
            raise KeyError("构造 closure 需要从 Data-Set-11 得到的 sfr_target。")

        mbh = float(self._data_log["mbh"])
        if not np.isfinite(mbh) or mbh <= 0:
            raise ValueError(f"黑洞质量必须为正有限值，得到 {mbh!r}。")
        if time_step is None:
            candidate = float(self._data_log.get("dt", 0.0))
            time_step = candidate if np.isfinite(candidate) and candidate >= 0 else 0.0
            time_step_source = "log_data.dt" if time_step > 0 else "continuous_limit"
        else:
            time_step = float(time_step)
            time_step_source = "explicit_argument"
        if not np.isfinite(time_step) or time_step < 0:
            raise ValueError("SFR 算子的 time_step 必须是非负有限值。")

        target = float(self.sfr_target)
        common = {
            "density": self.snapshot["density"],
            "internal_energy_density": self.snapshot["gas_energy"],
            "cooling_time": self.snapshot["cooling_time"],
            "cell_volume": self.grid_volume,
            "radius": self.r,
            "time_step": time_step,
        }
        results = {
            "data_compatible": calculate_macer_sfr(
                **common,
                mode="data_compatible",
                black_hole_mass=mbh,
            ),
            "paper_physical": calculate_macer_sfr(
                **common,
                mode="paper_physical",
                radial_acceleration=self.snapshot["radial_acceleration"],
            ),
        }
        selection_code = int(
            round(float(self._data_log.get("log_context_selection_code", 0)))
        )
        selection_names = {
            0: "window_mean",
            1: "linear_interpolation_empty_window",
            2: "nearest_endpoint_outside_log_range",
        }
        inferred_selection = selection_names.get(
            selection_code, f"unknown_code_{selection_code}"
        )
        if context_selection is None:
            context_selection = inferred_selection
        snapshot_time = float(self.time)
        context = {
            "black_hole_mass": mbh,
            "black_hole_mass_units": "2.5e7_Msun",
            "black_hole_mass_source": f"log_data.mbh:{context_selection}",
            "time_step": time_step,
            "time_step_units": "Gyr",
            "time_step_source": time_step_source,
            "log_context_selection": str(context_selection),
            "log_context_selection_code": selection_code,
            "log_context_scope_gyr": float(
                self._data_log.get("log_context_scope", 0.0)
            ),
            "log_context_window_count": int(
                round(float(self._data_log.get("log_context_window_count", -1)))
            ),
            "log_context_left_time_gyr": float(
                self._data_log.get("log_context_left_time", snapshot_time)
            ),
            "log_context_right_time_gyr": float(
                self._data_log.get("log_context_right_time", snapshot_time)
            ),
            "log_context_bracket_width_gyr": float(
                self._data_log.get("log_context_bracket_width", 0.0)
            ),
            "log_context_interpolation_weight": float(
                self._data_log.get("log_context_interpolation_weight", 0.0)
            ),
            "log_context_fallback_used": selection_code != 0,
            "snapshot_time_gyr": snapshot_time,
        }
        products = {}
        metadata_operators = {}
        for mode, result in results.items():
            parameters = dict(result.parameters)
            product_uid = None
            if self.sample_uid:
                product_uid = build_physics_product_uid(
                    self.sample_uid,
                    operator_id=str(parameters["operator_id"]),
                    operator_version=int(parameters["operator_version"]),
                    operator_parameters=parameters,
                    context=context,
                )
            baseline = float(result.total_sfr)
            products[mode] = {
                "baseline_total": baseline,
                "closure_true": target - baseline,
                "target_total": target,
                "units": self._target_metadata.get("units", "MACER_mass_code/Gyr"),
                "physics_product_uid": product_uid,
                "parameters": parameters,
                "context": dict(context),
                "cell_sfr_baseline": np.asarray(result.cell_sfr, dtype=np.float64),
            }
            metadata_operators[mode] = {
                "physics_product_uid": product_uid,
                "parameters": parameters,
                "context": dict(context),
            }
        self._sfr_products = products
        if self._metadata:
            self._metadata["supervision"] = {
                "canonical_target": "sfr_target",
                "canonical_baseline_mode": "data_compatible",
                "closure_definition": "sfr_target-sfr_macer_baseline",
                "signed_closure": True,
                "target_units": self._target_metadata.get(
                    "units", "MACER_mass_code/Gyr"
                ),
                "sfr_operators": metadata_operators,
            }
            validate_complete_metadata(self._metadata)
        return self

    # -------- Public Methods -------- # 用户直接调用的接口

    def read_hdfra( # 使用 pyhdf 库读取 HDF4 格式的Snapshot
        self,
        hdfra_path: str,
        time: float,
        log_path: str,
        func_parse_log: Callable,
        scope: float,
        target_config: dict | None = None,
        source_metadata: dict | None = None,
    ) -> "GalData":
        """
        Reads snapshot data and associated BH accretion rates from specified paths.
        Args:
            hdfra_path (str): Path to the hdfra.xxxxx file including the simulation snapshot.
            time (float): The simulation time corresponding to the snapshot to be read (Gyr).
            log_path (str): Path to the log file including the BH accretion accretion rate.
            func_parse_log (Callable): Function to parse the log file into readable data.
            scope (float): Time scope for filtering log data. If provided, only entries within [time - scope, time + scope] are retained.
        """
        self._load_snapshot_hdfra(hdfra_path, time)
        datasets_to_validate = {
            name
            for name, semantic in (self._corr or {}).items()
            if name in {"Data-Set-5", "Data-Set-10", "Data-Set-17"}
            or semantic in {"radial_acceleration", "theta_acceleration"}
        }
        datasets_to_validate.add("Data-Set-15")
        if target_config is not None:
            datasets_to_validate.add(str(target_config["dataset"]))
        _validate_macer_dataset_labels(self._data_hdfra, datasets_to_validate)
        self._load_log(log_path, func_parse_log, scope)
        self._pending_source_metadata = dict(source_metadata) if source_metadata is not None else None
        if target_config is not None:
            self._set_target_from_snapshot(target_config)
        return self

    def attach_fine_source_metadata(self, source_metadata: dict) -> "GalData":
        """为已重建的 legacy fine HDF5 补充完整源身份，不改变任何物理数组。"""

        if self._coarsening_metadata:
            raise ValueError("只能给 fine snapshot 附加源身份，不能重写粗粒化结果的身份。")
        self.__validate_coords()
        self._metadata = build_fine_metadata(
            source=source_metadata,
            snapshot_fields=tuple(self.snapshot),
            axes=self._axes,
            axis_edges=self._axis_edges,
            coordinate_mode=self._coordinate_mode,
            ndim=self.ndim,
            target_metadata=self._target_metadata,
            volume_source=self._volume_source,
        )
        validate_complete_metadata(self._metadata)
        return self

    def _set_target_from_snapshot(self, target_config: dict) -> "GalData":
        """Extract a scalar supervised target without exposing it as input."""
        required = {"dataset", "name", "reduction"}
        missing = required.difference(target_config)
        if missing:
            raise ValueError(f"Target configuration is missing required keys: {sorted(missing)}")

        dataset_name = str(target_config["dataset"])
        target_name = str(target_config["name"])
        reduction = str(target_config["reduction"]).lower()
        if self._corr is not None and dataset_name in self._corr:
            raise ValueError(
                f"Target dataset '{dataset_name}' is also present in correspondence; "
                "this would leak the label into model inputs."
            )

        try:
            target_field = np.asarray(self._data_hdfra.select(dataset_name)[:], dtype=np.float64)
        except Exception as exc:
            raise KeyError(f"Could not read target dataset '{dataset_name}' from the snapshot.") from exc
        target_field = np.squeeze(target_field)
        if not np.all(np.isfinite(target_field)):
            raise ValueError(f"Target dataset '{dataset_name}' contains NaN or Inf values.")

        if reduction == "sum":
            target_value = float(np.sum(target_field, dtype=np.float64))
        elif reduction == "mean":
            target_value = float(np.mean(target_field, dtype=np.float64))
        else:
            raise ValueError(f"Unsupported target reduction '{reduction}'. Use 'sum' or 'mean'.")
        if target_value < 0:
            raise ValueError(f"Snapshot target '{target_name}' must be non-negative, got {target_value}.")

        if self._data_log is None:
            self._data_log = {}
        self._data_log[target_name] = target_value
        self._data_log["sfr_target"] = target_value
        self._target_metadata = {
            "dataset": dataset_name,
            "name": target_name,
            "reduction": reduction,
            "quantity": "instantaneous_sfr",
            "units": str(target_config.get("units", "MACER_mass_code/Gyr")),
        }
        for key in ("mass_unit_msun", "time_unit_gyr", "to_msun_per_yr"):
            if key in target_config:
                self._target_metadata[key] = float(target_config[key])
        return self

    def _as_coarsening_input(self):
        """Build the pure-array input contract used by all coarsening modules."""
        self.__validate_coords()
        try:
            from raw_data_processing.coarsening_types import CoarseningInput
        except ModuleNotFoundError:
            from coarsening_types import CoarseningInput
        return CoarseningInput(
            snapshot=self.snapshot,
            grid_volume=np.asarray(self.grid_volume, dtype=np.float64),
            axes=tuple(np.asarray(axis) for axis in self._axes),
            axis_edges=tuple(np.asarray(edges) for edges in self._axis_edges),
            coordinate_mode=self._coordinate_mode,
            ndim=self.ndim,
        )

    def _from_coarsening_result(
        self,
        result,
        *,
        conservation_rtol: float,
        conservation_atol: float,
    ) -> "GalData":
        """Assemble a GalData object and validate an array-only result."""
        source = self._as_coarsening_input()
        try:
            from raw_data_processing.coarsening_common import validate_coarsening_conservation
        except ModuleNotFoundError:
            from coarsening_common import validate_coarsening_conservation
        conservation_report = validate_coarsening_conservation(
            source, result, rtol=conservation_rtol, atol=conservation_atol
        )
        coarse = GalData(ndim=self.ndim, coordinate_mode=self._coordinate_mode)
        coarse.time = self.time
        coarse._corr = self._corr.copy() if self._corr is not None else {}
        coarse._data_log = self._data_log.copy() if self._data_log is not None else None
        coarse._target_metadata = self._target_metadata.copy()
        coarse._data_hdfra = {
            name: np.asarray(values) for name, values in result.snapshot.items()
        }
        coarse._grid_volume = np.asarray(result.grid_volume, dtype=np.float64)
        coarse._volume_source = str(result.volume_source)
        coarse._axes = [np.asarray(axis) for axis in result.axes]
        coarse._axis_edges = [np.asarray(edges) for edges in result.axis_edges]
        coarse._coord = tuple(np.meshgrid(*coarse._axes))
        coarse._coarsening_metadata = dict(result.metadata)
        coarse._coarsening_diagnostics = {
            name: np.asarray(values) for name, values in result.diagnostics.items()
        }
        coarse._coarsening_diagnostics.update(
            {name: np.asarray(value) for name, value in conservation_report.items()}
        )
        coarse._metadata = build_derived_metadata(
            self._metadata,
            operator_metadata=coarse._coarsening_metadata,
            axes=coarse._axes,
            axis_edges=coarse._axis_edges,
            coordinate_mode=coarse._coordinate_mode,
            volume_source=coarse._volume_source,
        )
        if {
            "cooling_time", "radial_acceleration"
        }.issubset(coarse.snapshot) and coarse._data_log is not None and {
            "mbh", "sfr_target"
        }.issubset(coarse._data_log):
            coarse.compute_macer_sfr_products()
        return coarse

    def coarsen(
        self,
        operator: str,
        new_shape: Tuple[int, ...],
        *,
        conservation_rtol: float = 1e-6,
        conservation_atol: float = 1e-10,
        **operator_kwargs,
    ) -> "GalData":
        """Dispatch a named coarse-graining operator through one stable interface."""
        normalized = str(operator).lower().replace("-", "_")
        source = self._as_coarsening_input()
        if normalized in {"spatial", "spatial_block", "rh", "r_h"}:
            try:
                from raw_data_processing.coarsening_spatial import coarsen_spatial_block
            except ModuleNotFoundError:
                from coarsening_spatial import coarsen_spatial_block
            result = coarsen_spatial_block(
                source,
                tuple(new_shape),
                conservation_rtol=conservation_rtol,
                conservation_atol=conservation_atol,
                **operator_kwargs,
            )
        elif normalized in {
            "rm_v0", "r_m_v0", "axisymmetric_rm_v0",
            "adaptive_mass_axisymmetric_v0",
        }:
            try:
                from raw_data_processing.coarsening_rm_v0 import coarsen_mass_v0
            except ModuleNotFoundError:
                from coarsening_rm_v0 import coarsen_mass_v0
            result = coarsen_mass_v0(source, tuple(new_shape), **operator_kwargs)
        elif normalized in {
            "rm_v1", "r_m_v1", "axisymmetric_rm_v1",
            "adaptive_mass_axisymmetric_v1",
        }:
            try:
                from raw_data_processing.coarsening_rm_v1 import coarsen_mass_v1
            except ModuleNotFoundError:
                from coarsening_rm_v1 import coarsen_mass_v1
            result = coarsen_mass_v1(source, tuple(new_shape), **operator_kwargs)
        else:
            raise ValueError(
                f"Unknown coarsening operator {operator!r}; choose spatial_block, "
                "axisymmetric_rm_v0, or axisymmetric_rm_v1."
            )
        return self._from_coarsening_result(
            result,
            conservation_rtol=conservation_rtol,
            conservation_atol=conservation_atol,
        )

    def rescale(
        self,
        new_shape: Tuple[int, ...],
        *,
        conservation_rtol: float = 1e-6,
        conservation_atol: float = 1e-10,
    ) -> "GalData":
        """Backward-compatible interface for regular spatial-block R_h."""
        return self.coarsen(
            "spatial_block",
            new_shape,
            conservation_rtol=conservation_rtol,
            conservation_atol=conservation_atol,
        )

    def rescale_mass_adaptive(
        self,
        new_shape: Tuple[int, ...],
        *,
        mass_filter: float,
        kernel: str = "top_hat",
        initial_neighbors: int = 32,
        balance_tolerance: float = 1e-6,
        balance_max_iterations: int = 5000,
        conservation_rtol: float = 1e-6,
        conservation_atol: float = 1e-10,
    ) -> "GalData":
        """Backward-compatible interface for axisymmetric R_m-v0."""
        return self.coarsen(
            "axisymmetric_rm_v0",
            new_shape,
            mass_filter=mass_filter,
            kernel=kernel,
            initial_neighbors=initial_neighbors,
            balance_tolerance=balance_tolerance,
            balance_max_iterations=balance_max_iterations,
            conservation_rtol=conservation_rtol,
            conservation_atol=conservation_atol,
        )

    def set_corr(self, corr: List[Tuple[str, str]] | Tuple[str, str]) -> "GalData": # 坐标轴重映射，允许手动指定轴的对应关系，如设定第一个轴是theta
        """
        Sets the correlation mapping for hdfra file data to new names.
        Note: This class requires the presence of 'density' and 'gas_energy' in the snapshot data.
        Args:
            corr (List[Tuple[str, str]] | Tuple[str, str]): List of tuples or a single tuple where each tuple contains
                the original dataset name and the new variable name.
        """
        if isinstance(corr, dict):
            self._corr = dict(corr)
        elif isinstance(corr, tuple) and len(corr) == 2 and all(isinstance(x, str) for x in corr):
            self._corr = dict([corr])
        else:
            self._corr = dict(corr)
        _validate_macer_thermodynamic_correspondence(self._corr)
        return self

    def set_coord(self, coords: List[str] | str) -> "GalData":
        """
        Sets the correlation mapping for coordinate data in hdfra file to new names.
        Args:
            coords (List[str] | str): List of coordinate names or a single coordinate name.
        Note:
            the order of coordinates in the list must match the order of the coordinate system.
            2D polar: ["r", "theta"]
            2D cartesian: ["x", "y"]
            3D polar: ["r", "theta", "phi"]
            3D cartesian: ["x", "y", "z"]
            "theta" is the polar angle in 2D/3D, and "phi" is the azimuthal angle in 3D.
        """
        if isinstance(coords, str):
            coords = [coords]
        if len(coords) != self.ndim:
            raise ValueError(f"Coordinate count mismatch: expected {self.ndim}, got {len(coords)}.")
        self._axes = [self._data_hdfra.select(c)[:] for c in coords]
        self._axis_edges = [
            self._infer_axis_edges(axis, is_radial=(self._coordinate_mode == "polar" and i == 0))
            for i, axis in enumerate(self._axes)
        ]
        # np.meshgrid(r, theta) returns arrays in snapshot order (theta, r).
        self._coord = tuple(np.meshgrid(*self._axes))
        try:
            volume = np.asarray(self._data_hdfra.select("Data-Set-15")[:], dtype=np.float64)
        except Exception as exc:
            raise KeyError(
                "Raw HDFRA snapshot is missing authoritative cell volume Data-Set-15."
            ) from exc
        volume = np.squeeze(volume)
        expected_shape = self.snapshot["density"].shape
        if volume.shape != expected_shape:
            raise ValueError(
                f"Data-Set-15 shape {volume.shape} does not match snapshot shape {expected_shape}."
            )
        if not np.all(np.isfinite(volume)) or np.any(volume <= 0):
            raise ValueError("Data-Set-15 must contain only finite, strictly positive volumes.")
        self._grid_volume = volume
        self._volume_source = "Data-Set-15"
        if self._pending_source_metadata is not None:
            self._metadata = build_fine_metadata(
                source=self._pending_source_metadata,
                snapshot_fields=tuple(self.snapshot),
                axes=self._axes,
                axis_edges=self._axis_edges,
                coordinate_mode=self._coordinate_mode,
                ndim=self.ndim,
                target_metadata=self._target_metadata,
                volume_source=self._volume_source,
            )
            validate_complete_metadata(self._metadata)
        return self

    def save_h5(self, file_path: str) -> None: # 将Galdata保存为HDF5格式
        """
        Saves core data of GalData instance to an HDF5 file.
        Args:
            file_path (str): Path to the output HDF5 file.
        """
        if not self._sfr_products and {
            "cooling_time", "radial_acceleration"
        }.issubset(self.snapshot) and self._data_log is not None and {
            "mbh", "sfr_target"
        }.issubset(self._data_log):
            self.compute_macer_sfr_products()
        with h5py.File(file_path, "w") as f:
            f.attrs["processed_format_version"] = 6
            snap_grp = f.create_group("snapshot")
            for name, data in self.snapshot.items():
                dataset = snap_grp.create_dataset(name, data=data)
                if name == GAS_INTERNAL_ENERGY_DENSITY_FIELD:
                    dataset.attrs["physical_quantity"] = "gas_internal_energy_density"
                    dataset.attrs["source_dataset"] = GAS_INTERNAL_ENERGY_DENSITY_DATASET
                    dataset.attrs["integrated_quantity"] = "gas_internal_energy"
                    dataset.attrs["integration_weight"] = "geometry/cell_volume"
                    dataset.attrs["temperature_relation"] = "temperature=93*gas_energy/density"
                elif name == "cooling_time":
                    dataset.attrs["physical_quantity"] = "gas_cooling_time"
                    dataset.attrs["source_dataset"] = "Data-Set-17"
                    dataset.attrs["coarsening_rule"] = (
                        "derive_from_gas_energy/cooling_rate_density"
                    )
                elif name == "cooling_rate_density":
                    dataset.attrs["physical_quantity"] = "gas_cooling_rate_density"
                    dataset.attrs["definition"] = "gas_energy/cooling_time"
                    dataset.attrs["conserved_integral"] = (
                        "sum(cooling_rate_density*cell_volume)"
                    )
                elif name in {"radial_acceleration", "theta_acceleration"}:
                    dataset.attrs["physical_quantity"] = "gravitational_acceleration"
                    dataset.attrs["basis"] = "spherical"
                    dataset.attrs["source_dataset"] = (
                        "Data-Set-8" if name == "radial_acceleration" else "Data-Set-9"
                    )

            coord_grp = f.create_group("coordinates")
            for i, ax in enumerate(self._axes):
                coord_grp.create_dataset(f"axis_{i}", data=ax)
                coord_grp.create_dataset(f"axis_{i}_edges", data=self._axis_edges[i])
            coord_grp.attrs["coordinate_mode"] = self._coordinate_mode
            coord_grp.attrs["ndim"] = self.ndim

            geometry_grp = f.create_group("geometry")
            geometry_grp.create_dataset("cell_volume", data=self.grid_volume)
            geometry_grp.attrs["volume_source"] = self._volume_source

            log_grp = f.create_group("log_data")
            for key, value in self._data_log.items():
                log_grp.create_dataset(key, data=value)
            for key, value in self._target_metadata.items():
                log_grp.attrs[f"target_{key}"] = value

            f.attrs["time"] = self.time

            if self._metadata:
                metadata_grp = f.create_group("metadata")
                metadata_grp.attrs["schema_name"] = self._metadata.get("schema_name", "")
                metadata_grp.attrs["schema_version"] = int(
                    self._metadata.get("schema_version", 0)
                )
                metadata_grp.attrs["metadata_completeness"] = self._metadata.get(
                    "metadata_completeness", "unknown"
                )
                identity = self._metadata.get("identity", {})
                if identity.get("parent_hr_uid"):
                    metadata_grp.attrs["parent_hr_uid"] = identity["parent_hr_uid"]
                if identity.get("sample_uid"):
                    metadata_grp.attrs["sample_uid"] = identity["sample_uid"]
                metadata_grp.create_dataset(
                    "contract_json",
                    data=metadata_to_json(self._metadata),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )

            corr_grp = f.create_group("correlation")
            corr_grp.attrs["length"] = len(self._corr)
            for i, (orig, new) in enumerate(self._corr.items()):
                corr_grp.attrs[f"{i}_orig"] = orig
                corr_grp.attrs[f"{i}_new"] = new

            if self._coarsening_metadata or self._coarsening_diagnostics:
                coarse_grp = f.create_group("coarsening")
                for key, value in self._coarsening_metadata.items():
                    if isinstance(value, dict):
                        coarse_grp.attrs[key] = "__json__:" + metadata_to_json(value)
                    else:
                        coarse_grp.attrs[key] = value
                for name, values in self._coarsening_diagnostics.items():
                    coarse_grp.create_dataset(name, data=np.asarray(values))

            if self._sfr_products:
                sfr_grp = f.create_group("sfr_supervision")
                sfr_grp.attrs["schema_version"] = 1
                sfr_grp.attrs["default_mode"] = "data_compatible"
                sfr_grp.attrs["target_total"] = float(self.sfr_target)
                sfr_grp.attrs["target_units"] = self._target_metadata.get(
                    "units", "MACER_mass_code/Gyr"
                )
                canonical = self._sfr_products["data_compatible"]
                canonical_grp = sfr_grp.create_group("canonical")
                canonical_grp.create_dataset("Y_alpha", data=float(self.sfr_target))
                canonical_grp.create_dataset(
                    "B_alpha", data=float(canonical["baseline_total"])
                )
                canonical_grp.create_dataset(
                    "C_alpha_true", data=float(canonical["closure_true"])
                )
                canonical_grp.attrs["definition"] = "C_alpha_true=Y_alpha-B_alpha"
                for mode, product in self._sfr_products.items():
                    mode_grp = sfr_grp.create_group(mode)
                    mode_grp.attrs["baseline_total"] = float(product["baseline_total"])
                    mode_grp.attrs["closure_true"] = float(product["closure_true"])
                    mode_grp.attrs["units"] = str(product["units"])
                    if product.get("physics_product_uid"):
                        mode_grp.attrs["physics_product_uid"] = product[
                            "physics_product_uid"
                        ]
                    mode_grp.create_dataset(
                        "cell_sfr_baseline",
                        data=np.asarray(product["cell_sfr_baseline"], dtype=np.float64),
                    )
                    mode_grp.create_dataset(
                        "parameters_json",
                        data=metadata_to_json(product["parameters"]),
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
                    mode_grp.create_dataset(
                        "context_json",
                        data=metadata_to_json(product["context"]),
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )

            # print(f"GalData saved to HDF5: {file_path}")

    def load_h5(self, file_path: str) -> "GalData": # 打开已经保存数据的HDF5文件
        """

        Loads GalData instance from an HDF5 file.
        Args:
            file_path (str): Path to the input HDF5 file.
        """
        with h5py.File(file_path, "r") as f:
            ndim = f["coordinates"].attrs["ndim"]
            coordinate_mode = f["coordinates"].attrs["coordinate_mode"]
            if isinstance(coordinate_mode, bytes):
                coordinate_mode = coordinate_mode.decode()
            self.ndim = int(ndim)
            if coordinate_mode not in {"polar", "cartesian"}:
                raise ValueError(f"Invalid stored coordinate mode {coordinate_mode!r}.")
            self._coordinate_mode = coordinate_mode

            self.time = f.attrs["time"]
            self._metadata = {}
            self._sfr_products = {}
            self._pending_source_metadata = None
            if "metadata" in f and "contract_json" in f["metadata"]:
                payload = f["metadata"]["contract_json"][()]
                self._metadata = metadata_from_json(payload)
                if self._metadata.get("metadata_completeness") == "complete":
                    validate_complete_metadata(self._metadata)

            if "sfr_supervision" in f:
                sfr_grp = f["sfr_supervision"]
                stored_target = float(sfr_grp.attrs["target_total"])
                canonical_grp = sfr_grp.get("canonical")
                if canonical_grp is None:
                    raise ValueError("sfr_supervision 缺少 canonical Y/B/C 数据组。")
                canonical_y = float(canonical_grp["Y_alpha"][()])
                canonical_b = float(canonical_grp["B_alpha"][()])
                canonical_c = float(canonical_grp["C_alpha_true"][()])
                if not np.isclose(
                    canonical_c, canonical_y - canonical_b, rtol=1e-12, atol=1e-12
                ):
                    raise ValueError("canonical SFR 产品违反 C_alpha_true=Y_alpha-B_alpha。")
                for mode in (name for name in sfr_grp if name != "canonical"):
                    mode_grp = sfr_grp[mode]
                    baseline = float(mode_grp.attrs["baseline_total"])
                    closure = float(mode_grp.attrs["closure_true"])
                    if not np.isclose(
                        closure, stored_target - baseline, rtol=1e-12, atol=1e-12
                    ):
                        raise ValueError(
                            f"SFR supervision {mode!r} violates closure=target-baseline."
                        )
                    product_uid = mode_grp.attrs.get("physics_product_uid")
                    if isinstance(product_uid, bytes):
                        product_uid = product_uid.decode()
                    self._sfr_products[mode] = {
                        "baseline_total": baseline,
                        "closure_true": closure,
                        "target_total": stored_target,
                        "units": str(
                            mode_grp.attrs.get(
                                "units", sfr_grp.attrs.get("target_units", "")
                            )
                        ),
                        "physics_product_uid": product_uid,
                        "parameters": metadata_from_json(
                            mode_grp["parameters_json"][()]
                        ),
                        "context": metadata_from_json(mode_grp["context_json"][()]),
                        "cell_sfr_baseline": mode_grp["cell_sfr_baseline"][()],
                    }
                default_product = self._sfr_products.get("data_compatible")
                if default_product is None or not (
                    np.isclose(canonical_y, stored_target, rtol=1e-12, atol=1e-12)
                    and np.isclose(
                        canonical_b,
                        default_product["baseline_total"],
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    and np.isclose(
                        canonical_c,
                        default_product["closure_true"],
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ):
                    raise ValueError(
                        "canonical Y/B/C 与 data_compatible 默认产品不一致。"
                    )
                supervision = self._metadata.get("supervision", {})
                declared_operators = supervision.get("sfr_operators", {})
                for mode, product in self._sfr_products.items():
                    if not declared_operators:
                        break
                    declared = declared_operators.get(mode)
                    if not isinstance(declared, dict):
                        raise ValueError(f"metadata 未声明 SFR 产品模式 {mode!r}。")
                    if (
                        metadata_to_json(declared.get("parameters", {}))
                        != metadata_to_json(product["parameters"])
                        or metadata_to_json(declared.get("context", {}))
                        != metadata_to_json(product["context"])
                    ):
                        raise ValueError(
                            f"metadata 与 sfr_supervision/{mode} 的参数或上下文不一致。"
                        )
                    expected_product_uid = build_physics_product_uid(
                        self.sample_uid,
                        operator_id=str(product["parameters"]["operator_id"]),
                        operator_version=product["parameters"]["operator_version"],
                        operator_parameters=product["parameters"],
                        context=product["context"],
                    )
                    if (
                        product.get("physics_product_uid") != expected_product_uid
                        or declared.get("physics_product_uid") != expected_product_uid
                    ):
                        raise ValueError(f"SFR 产品 {mode!r} 的 physics_product_uid 不一致。")

            self._data_hdfra = {}
            for name in f["snapshot"]:
                self._data_hdfra[name] = f["snapshot"][name][()]
            gas_energy_dataset = f["snapshot"].get(GAS_INTERNAL_ENERGY_DENSITY_FIELD)
            stored_semantics = None
            if gas_energy_dataset is not None:
                stored_semantics = gas_energy_dataset.attrs.get("physical_quantity")
                if isinstance(stored_semantics, bytes):
                    stored_semantics = stored_semantics.decode()
                if stored_semantics not in (None, "gas_internal_energy_density"):
                    raise ValueError(
                        f"Processed file '{file_path}' assigns incompatible semantics "
                        f"{stored_semantics!r} to 'gas_energy'."
                    )

            axis_names = sorted(
                (name for name in f["coordinates"] if name.startswith("axis_") and not name.endswith("_edges")),
                key=lambda name: int(name.split("_")[1]),
            )
            self._axes = [f["coordinates"][name][()] for name in axis_names]
            edge_names = [f"axis_{i}_edges" for i in range(len(self._axes))]
            if not all(name in f["coordinates"] for name in edge_names):
                raise ValueError(
                    f"Processed file '{file_path}' lacks stored cell edges. Regenerate it from raw HDFRA."
                )
            self._axis_edges = [f["coordinates"][name][()] for name in edge_names]
            for i, (axis, edges) in enumerate(zip(self._axes, self._axis_edges)):
                if edges.shape != (len(axis) + 1,) or not np.all(np.isfinite(edges)):
                    raise ValueError(
                        f"Stored axis_{i}_edges must be finite and have length len(axis_{i}) + 1."
                    )
                if np.any(np.diff(edges) <= 0):
                    raise ValueError(f"Stored axis_{i}_edges must be strictly increasing.")
            self._coord = tuple(np.meshgrid(*self._axes))

            if "geometry" not in f or "cell_volume" not in f["geometry"]:
                raise ValueError(
                    f"Processed file '{file_path}' lacks authoritative cell_volume. "
                    "Regenerate it from raw HDFRA Data-Set-15."
                )
            self._grid_volume = np.asarray(f["geometry"]["cell_volume"][()], dtype=np.float64)
            self._volume_source = f["geometry"].attrs.get("volume_source", "unknown")
            if isinstance(self._volume_source, bytes):
                self._volume_source = self._volume_source.decode()
            expected_shape = next(iter(self._data_hdfra.values())).shape
            if self._grid_volume.shape != expected_shape:
                raise ValueError(
                    f"Stored cell_volume shape {self._grid_volume.shape} does not match snapshot shape {expected_shape}."
                )
            if not np.all(np.isfinite(self._grid_volume)) or np.any(self._grid_volume <= 0):
                raise ValueError("Stored cell_volume must be finite and strictly positive.")

            if "log_data" in f:
                self._data_log = {k: f["log_data"][k][()] for k in f["log_data"]}
                self._target_metadata = {
                    key[len("target_") :]: value.decode() if isinstance(value, bytes) else value
                    for key, value in f["log_data"].attrs.items()
                    if key.startswith("target_")
                }
                if self._sfr_products:
                    stored_target = float(
                        self._sfr_products["data_compatible"]["target_total"]
                    )
                    if not np.isclose(
                        stored_target,
                        float(self._data_log["sfr_target"]),
                        rtol=1e-12,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            "sfr_supervision target 与 log_data/sfr_target 不一致。"
                        )

            corr_grp = f["correlation"]
            self._corr = dict(
                (corr_grp.attrs[f"{i}_orig"], corr_grp.attrs[f"{i}_new"]) for i in range(corr_grp.attrs["length"])
            )
            if self._corr:
                _validate_macer_thermodynamic_correspondence(self._corr)
            elif stored_semantics != "gas_internal_energy_density":
                raise ValueError(
                    f"Processed file '{file_path}' has no raw-field correspondence and "
                    "does not explicitly identify 'gas_energy' as internal-energy density."
                )

            self._coarsening_metadata = {}
            self._coarsening_diagnostics = {}
            if "coarsening" in f:
                coarse_grp = f["coarsening"]
                self._coarsening_metadata = {
                    key: (
                        metadata_from_json(value.decode()[len("__json__:"):])
                        if isinstance(value, bytes) and value.decode().startswith("__json__:")
                        else metadata_from_json(value[len("__json__:"):])
                        if isinstance(value, str) and value.startswith("__json__:")
                        else value.decode() if isinstance(value, bytes) else value
                    )
                    for key, value in coarse_grp.attrs.items()
                }
                self._coarsening_diagnostics = {
                    name: coarse_grp[name][()] for name in coarse_grp
                }

        # print(f"GalData loaded from HDF5: {file_path}")
        return self

    # -------- Private Methods --------

    def _load_snapshot_hdfra(self, path: str, time: float) -> "GalData":
        """
        Reads snapshot data from the specified path.
        Args:
            path (str): Path to the hdfra.xxxxx file including the simulation snapshot.
            time (float): The simulation time corresponding to the snapshot to be read (Gyr).
        """
        # Validate file path
        path = _validate_file_path(path)
        if SD is None:
            raise ImportError(
                "Reading raw MACER HDF4 snapshots requires the optional 'pyhdf' package. "
                "Loading processed HDF5 files does not."
            )
        self.time = time
        # Convert path to relative path to avoid pyhdf C-library issues with Chinese chars on Windows
        rel_path = os.path.relpath(path)
        self._data_hdfra = SD(rel_path)
        return self

    def _load_log(self, path: str, f: Callable, scope: float) -> "GalData":
        """
        Reads BH accretion rates from the specified log file.
        Args:
            path (str): Path to the log file including the BH accretion accretion rate.
            f (Callable): Function to parse the log file into readable data.
                - Accept parameters (path: str, time: float, scope: float).
                - Return a structured data object (e.g., pandas DataFrame) containing at least the columns 'time', 'mbh', 'mdot_macer', and 'mdot_edd'.
            scope (float, optional): Time scope for filtering log data. If provided, only entries within [time - scope, time + scope] are retained.
        """
        path = _validate_file_path(path)
        if self.time is None:
            raise ValueError("Simulation time not set. Please call load_snapshot_hdfra() first.")
        parsed = f(path, time=self.time, scope=scope)
        self._data_log = dict(parsed.items()) if hasattr(parsed, "items") else dict(parsed)
        self._data_log["log_context_scope"] = float(scope)
        return self

    @staticmethod
    def _block_reduce_sum(data: np.ndarray, new_shape: Tuple[int, ...]) -> np.ndarray:
        """Compatibility wrapper; numerical implementation lives in coarsening_common."""
        try:
            from raw_data_processing.coarsening_common import block_reduce_sum
        except ModuleNotFoundError:
            from coarsening_common import block_reduce_sum
        return block_reduce_sum(data, new_shape)

    @staticmethod
    def _infer_axis_edges(axis: np.ndarray, *, is_radial: bool) -> np.ndarray:
        """Infer fine-grid edges once; stored edges are used thereafter.

        MACER provides cell centres and authoritative volumes. Radial centres
        are logarithmic, so geometric midpoints preserve their native topology;
        angular/Cartesian axes use arithmetic midpoints.
        """
        axis = np.asarray(axis, dtype=np.float64)
        if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
            raise ValueError("Coordinate axes must be finite one-dimensional arrays of length >= 2.")
        if np.any(np.diff(axis) <= 0):
            raise ValueError("Coordinate axes must be strictly increasing.")
        if is_radial:
            if np.any(axis <= 0):
                raise ValueError("Logarithmic radial centres must be strictly positive.")
            interior = np.sqrt(axis[:-1] * axis[1:])
            first = axis[0] ** 2 / interior[0]
            last = axis[-1] ** 2 / interior[-1]
        else:
            interior = 0.5 * (axis[:-1] + axis[1:])
            first = axis[0] - (interior[0] - axis[0])
            last = axis[-1] + (axis[-1] - interior[-1])
        edges = np.concatenate(([first], interior, [last]))
        if np.any(np.diff(edges) <= 0):
            raise ValueError("Inferred coordinate edges are not strictly increasing.")
        return edges
    # ------- Validation Methods -------

    def __validate_coords(self) -> None:
        if self._coord is None:
            raise ValueError("Coordinates not set. Please call set_coord() first.")


# =============== Galaxy DataSet Class =============== 将多个galdata对象聚合为张量形式，为了对接机器学习框架


class GalDataSet:
    def __init__(self):
        """
        Initializes the GalDataSet object for managing multiple GalData instances.
        """
        self._raw_data = None
        self._group_labels = None
        self.folder_path = None
        self._x_keys = None
        self._coord_keys = None
        self._target_metadata = {}
        self._coarsening_signature = None
        self._coordinate_reference = None

    # --------- Properties -------- 将零散的天体物理快照数据，无缝转化为 Python 机器学习框架可以直接摄入的标准张量格式

    @property
    def raw_data(self) -> dict:
        return self._raw_data

    @property
    def x(self) -> np.ndarray: 
        '''
        提取_x_keys (如密度、能量、气体质量等) 对应的所有数据。原始组装出来的数据形状是 (Channels, Batch_size, Height, Width)。通过 moveaxis 变换后，
        形状变成了 (Batch_size, Channels, Height, Width)。这是计算机视觉中卷积神经网络 (CNN) 最标准的输入格式。它将物理场直接转化为了多通道的“图像”。
        '''
        self.__verify_data_loaded() # 确保数据已经成功加载到内存中
        return np.moveaxis(np.array([self._raw_data[k] for k in self._x_keys]), 0, 1)

    @property
    def input_channels(self) -> tuple[str, ...]:
        """Names of the physical-field channels, in the same order as ``x``."""
        self.__verify_data_loaded()
        return tuple(self._x_keys)

    @property
    def target_metadata(self) -> dict:
        self.__verify_data_loaded()
        return self._target_metadata.copy()

    @property
    def sample_uid(self) -> np.ndarray:
        """返回与样本轴对齐的衍生数据产品 UID。"""

        self.__verify_data_loaded()
        return np.asarray(self._raw_data["sample_uid"], dtype=str)

    @property
    def parent_hr_uid(self) -> np.ndarray:
        """返回与样本轴对齐的 HR 母 snapshot UID。"""

        self.__verify_data_loaded()
        return np.asarray(self._raw_data["parent_hr_uid"], dtype=str)

    @property
    def mass_filter(self) -> np.ndarray:
        """返回 R_m 的目标质量尺度；非 R_m 样本为 NaN。"""

        self.__verify_data_loaded()
        return np.asarray(self._raw_data["mass_filter"], dtype=np.float64)

    @property
    def log_mass_filter(self) -> np.ndarray:
        """返回模型可用的 log10(M_filter) 分辨率条件。"""

        values = self.mass_filter
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(
                "log_mass_filter 仅适用于具有正有限 mass_filter 的完整 R_m 数据集。"
            )
        return np.log10(values)

    @property
    def h_eff(self) -> np.ndarray:
        """返回预留的 H_eff 场，目前不会自动加入模型输入。"""

        self.__verify_data_loaded()
        return np.asarray(self._raw_data["h_eff"], dtype=np.float64)

    @property
    def n_eff(self) -> np.ndarray:
        """返回预留的 N_eff 场，目前不会自动加入模型输入。"""

        self.__verify_data_loaded()
        return np.asarray(self._raw_data["n_eff"], dtype=np.float64)

    @property
    def y(self) -> np.ndarray: # 提取恒星形成率并计算对数
        self.__verify_data_loaded()
        sfr_target = np.array(self._raw_data["sfr_target"])

        return np.log10(sfr_target + 1e-9) # 添加微小数防止log(0)例外

    # @property
    # def mbh(self) -> np.ndarray: # 将黑洞质量转化为一维数组
    #     self.__verify_data_loaded()
    #     return np.array(self._raw_data["mbh"])

    @property
    def coordinate(self) -> np.ndarray: # 提取 _coord_keys 对应的数据，同样使用了 np.moveaxis(..., 0, 1)，从而返回与特征张量 x 完全对齐的坐标网格张量
        self.__verify_data_loaded()
        return np.moveaxis(np.array([self._raw_data[k] for k in self._coord_keys]), 0, 1)

    @property
    def time(self) -> np.ndarray: # 返回每个样本对应的模拟时间戳数组
        self.__verify_data_loaded()
        return np.array(self._raw_data["time"])

    # @property
    # def mdot_edd(self) -> np.ndarray: # 将爱丁顿吸积率转化为一维数组
    #     self.__verify_data_loaded()
    #     return np.array(self._raw_data["mdot_edd"])

    @property
    def sfr_target(self) -> np.ndarray: # 将恒星形成率转化为一维数组
        self.__verify_data_loaded()
        return np.array(self._raw_data["sfr_target"])

    @property
    def sfr_ks(self) -> np.ndarray:
        """Physical (untransformed) Kennicutt-Schmidt SFR baseline."""
        self.__verify_data_loaded()
        return np.array(self._raw_data["sfr_ks"])

    @property
    def sfr_macer_baseline(self) -> np.ndarray:
        """默认数据兼容 MACER 物理基线 ``B_alpha``。"""

        self.__verify_data_loaded()
        return np.asarray(self._raw_data["sfr_macer_baseline"], dtype=np.float64)

    @property
    def sfr_closure_true(self) -> np.ndarray:
        """允许为负的真闭合项 ``C_alpha=Y_alpha-B_alpha``。"""

        self.__verify_data_loaded()
        return np.asarray(self._raw_data["sfr_closure_true"], dtype=np.float64)

    @property
    def physics_product_uid(self) -> np.ndarray:
        self.__verify_data_loaded()
        return np.asarray(self._raw_data["physics_product_uid"], dtype=str)

    @property
    def groups(self) -> np.ndarray: # 将内部的 _group_labels 列表转化为字符串数组返回。这对应了原始数据属于哪个模拟子目录（如不同的星系初始条件或 AGN 反馈参数）。它是后续进行分层切分和样本均衡的核心依据。
        self.__verify_data_loaded()
        if self._group_labels is None:
            raise ValueError("Group labels not initialized. Please call load_data() first.")
        return np.asarray(self._group_labels, dtype=str)

    @property
    def n_groups(self) -> int: # 利用 np.unique 统计 groups 中不重复的标签数量，快速获知当前数据集中混合了多少种不同物理环境的星系数据。
        self.__verify_data_loaded()
        return len(np.unique(self.groups))

    # @property
    # def y_baseline(self) -> np.ndarray: # 计算邦迪吸积率和爱丁顿吸积率的对数比值，是一个物理基准，用来评判机器预测的y的好坏。
    #     self.__verify_data_loaded()
    #     mdot_bondi = np.array(self._raw_data["mdot_bondi"])
    #     mdot_edd = np.array(self._raw_data["mdot_edd"])
    #     return np.log10(mdot_bondi / mdot_edd)

    @property
    def y_baseline(self) -> np.ndarray:
        """Deprecated legacy log view of the KS baseline.

        Current SFR caches do not use this property. ``prepare_data.py`` reads
        physical ``sfr_ks`` and applies the same configured ``transform_sfr``
        function used for the target. Prefer ``sfr_ks`` in new code.
        """
        self.__verify_data_loaded()
        sfr_ks_val = np.array(self._raw_data["sfr_ks"])
        return np.log10(sfr_ks_val + 1e-9)

    # --------- Dunder Methods -------- 重载 Python 的内置操作符或函数，让自定义对象（比如这里的 GalDataSet）能够像 Python 原生的列表或数字一样去行动。

    def __len__(self): # 这段代码允许使得可以直接在外部脚本中使用 length = len(dataset) 来获取数据集的大小。
        self.__verify_data_loaded()
        return len(self._raw_data[self._x_keys[0]])
    
    def __add__(self, other: "GalDataSet") -> "GalDataSet": # 使让数据集支持 + 拼接
        self.__verify_data_loaded()
        other.__verify_data_loaded()

        if self._target_metadata != other._target_metadata:
            raise ValueError(
                "Cannot combine datasets with different target semantics: "
                f"{self._target_metadata} != {other._target_metadata}"
            )
        if self._coarsening_signature != other._coarsening_signature:
            raise ValueError(
                "Cannot combine datasets produced by different coarsening operators: "
                f"{self._coarsening_signature} != {other._coarsening_signature}"
            )
        if self._coordinate_reference is not None and other._coordinate_reference is not None:
            if len(self._coordinate_reference) != len(other._coordinate_reference) or any(
                left.shape != right.shape or not np.array_equal(left, right)
                for left, right in zip(self._coordinate_reference, other._coordinate_reference)
            ):
                raise ValueError("Cannot combine datasets defined on different coordinate grids.")
        if self._x_keys != other._x_keys or self._coord_keys != other._coord_keys:
            raise ValueError("Cannot combine datasets with different input channels or coordinates.")

        new_dataset = GalDataSet()
        new_dataset._x_keys = self._x_keys
        new_dataset._coord_keys = self._coord_keys
        new_dataset._target_metadata = self._target_metadata.copy()
        new_dataset._coarsening_signature = self._coarsening_signature
        new_dataset._coordinate_reference = self._coordinate_reference

        new_dataset._raw_data = {}
        for key in self._raw_data:
            new_dataset._raw_data[key] = self._raw_data[key] + other._raw_data[key] # 利用 Python 原生的列表加法 list1 + list2，把两个数据集的每一个物理量列表拼接在一起。

        new_dataset._group_labels = self._group_labels + other._group_labels
        return new_dataset

    # -------- Public Methods --------

    def select_input_channels(
        self,
        *,
        include=None,
        exclude=None,
    ) -> "GalDataSet":
        """
        Select channels exposed through ``x`` while retaining all raw fields.

        Supplying neither argument preserves the historical all-channel
        behavior. Coordinates, targets, baselines, and metadata are unchanged.
        """
        self.__verify_data_loaded()
        if include is not None and exclude is not None:
            raise ValueError("Specify either include or exclude, not both.")

        available = list(self._x_keys)
        if include is not None:
            requested = list(dict.fromkeys(include))
            unknown = [name for name in requested if name not in available]
            if unknown:
                raise KeyError(f"Unknown input channel(s): {unknown}. Available: {available}")
            selected = requested
        elif exclude is not None:
            excluded = set(exclude)
            unknown = sorted(excluded.difference(available))
            if unknown:
                raise KeyError(f"Unknown input channel(s): {unknown}. Available: {available}")
            selected = [name for name in available if name not in excluded]
        else:
            selected = available

        if not selected:
            raise ValueError("At least one input channel must be retained.")
        self._x_keys = selected
        return self

    def load_data(
        self,
        folder_path,
        *,
        allow_mixed_coarsening: bool = False,
        require_complete_metadata: bool = False,
    ) -> "GalDataSet":
        """
        整个数据集的入口。它接收一个或多个文件夹路径，遍历里面所有的 HDF5 数据文件。
        它把每个文件里的物理场（如密度、能量）、坐标网格、黑洞质量（mbh）、实际吸积率（mdot_macer 和 mdot_edd）提取出来，分别追加到内部的 _raw_data 字典中。
        在读取时，它会实时调用 GalData 的 mdot_bondi() 方法计算邦迪吸积率并存入数据集。同时，它会用 _group_labels 记录每个切片属于哪个星系文件夹，为后续的分组提供依据。
        """
        """
        Loads data from the specified folder path(s).
        Args:
            folder_path (str | list[str]): Path to the folder(s) containing the data files.
        """
        # Handle folder paths
        folder_path = folder_path if isinstance(folder_path, list) else [folder_path]
        
        valid_folder_path = []
        for fp in folder_path:
            # 兼容 fp 是一维列表 [path] 或是二维列表 [path, group_idx] 的情况
            path = fp[0] if isinstance(fp, (list, tuple)) else fp
            if os.path.exists(path) and any(f.endswith('.h5') for f in os.listdir(path)):
                valid_folder_path.append(fp)
            else:
                print(f"  [Warning] Missing or empty data directory: {path}. Skipping...")
                
        if not valid_folder_path:
            raise ValueError("No valid data directories found with .h5 files to load.")
            
        self.folder_path = [fp[0] if isinstance(fp, (list, tuple)) else fp for fp in valid_folder_path]

        # Initialize raw data dictionary
        self._group_labels = []
        data = GalData()
        
        # 寻找第一个合法的 .h5 文件用来初始化数据集结构
        first_file = sorted(
            f for f in os.listdir(self.folder_path[0]) if f.endswith(".h5")
        )[0]
        data_path = os.path.join(self.folder_path[0], first_file)
        
        data.load_h5(data_path)
        self._target_metadata = data.target_metadata
        def processing_signature(sample: GalData) -> tuple:
            derivation = sample.metadata.get("derivation", {})
            supervision = sample.metadata.get("supervision", {})
            operator_parameters = {
                mode: entry.get("parameters", {})
                for mode, entry in supervision.get("sfr_operators", {}).items()
            }
            return (
                str(derivation.get("operator_id", "unknown")),
                str(derivation.get("operator_version", "unknown")),
                metadata_to_json(derivation.get("operator_parameters", {})),
                str(supervision.get("canonical_baseline_mode", "none")),
                metadata_to_json(operator_parameters),
            )

        self._coarsening_signature = processing_signature(data)
        self._coordinate_reference = tuple(
            np.asarray(axis).copy() for axis in data.raw_axes
        )
        self._x_keys = list(data.snapshot.keys())
        self._coord_keys = [f"coordinate_{i}" for i in range(len(data.coordinate))]
        self._x_keys += ["gas_mass", "grid_volume"] 
        keys = (
            self._x_keys
            + self._coord_keys
            + [
                # "mbh",
                "sfr_target",
                # "mdot_edd",
                "time",
                "sfr_ks",
                "sfr_macer_baseline",
                "sfr_closure_true",
                "physics_product_uid",
                "sample_uid",
                "parent_hr_uid",
                "operator_id",
                "mass_filter",
                "h_eff",
                "n_eff",
                "sample_metadata",
                # "mdot_bondi",
            ]
        )
        self._raw_data = {k: [] for k in keys}

        # Load data from all specified folder paths
        for fp_info in valid_folder_path:
            # 安全地解包 fp 和 group_idx
            fp = fp_info[0] if isinstance(fp_info, (list, tuple)) else fp_info
            group_idx = fp_info[1] if isinstance(fp_info, (list, tuple)) and len(fp_info) > 1 else None
            
            for file_name in sorted(os.listdir(fp)):
                if not file_name.endswith('.h5'):
                    continue
                file_path = os.path.join(fp, file_name)
                data = GalData().load_h5(file_path)
                if require_complete_metadata:
                    validate_complete_metadata(data.metadata)
                if data.target_metadata != self._target_metadata:
                    raise ValueError(
                        f"Inconsistent target metadata in '{file_path}': "
                        f"{data.target_metadata} != {self._target_metadata}"
                    )
                signature = processing_signature(data)
                if signature != self._coarsening_signature and not allow_mixed_coarsening:
                    raise ValueError(
                        f"Inconsistent coarsening operator in '{file_path}': "
                        f"{signature} != {self._coarsening_signature}. Do not mix R_h/R_m "
                        "or different mass_filter values silently."
                    )
                current_axes = tuple(np.asarray(axis) for axis in data.raw_axes)
                if len(current_axes) != len(self._coordinate_reference) or any(
                    left.shape != right.shape or not np.array_equal(left, right)
                    for left, right in zip(self._coordinate_reference, current_axes)
                ):
                    raise ValueError(
                        f"Inconsistent coordinate grid in '{file_path}'. "
                        "Do not concatenate same-shaped tensors with different physical geometry."
                    )
                for key, value in data.snapshot.items():
                    self._raw_data[key].append(value)
                for i, coord in enumerate(data.coordinate):
                    self._raw_data[f"coordinate_{i}"].append(coord)
                self._raw_data["gas_mass"].append(data.gas_mass)
                self._raw_data["grid_volume"].append(data.grid_volume)
                # self._raw_data["mbh"].append(data.mbh)
                self._raw_data["sfr_target"].append(data.sfr_target)
                # self._raw_data["mdot_edd"].append(data.mdot_edd)
                self._raw_data["time"].append(data.time)
                # self._raw_data["mdot_bondi"].append(data.mdot_bondi()) # 计算邦迪吸积率
                self._raw_data["sfr_ks"].append(data.sfr_ks()) # 计算KS定律SFR
                default_product = data.sfr_products.get("data_compatible")
                if default_product is None:
                    self._raw_data["sfr_macer_baseline"].append(np.nan)
                    self._raw_data["sfr_closure_true"].append(np.nan)
                    self._raw_data["physics_product_uid"].append("")
                else:
                    self._raw_data["sfr_macer_baseline"].append(
                        float(default_product["baseline_total"])
                    )
                    self._raw_data["sfr_closure_true"].append(
                        float(default_product["closure_true"])
                    )
                    self._raw_data["physics_product_uid"].append(
                        default_product.get("physics_product_uid") or ""
                    )
                output_shape = data.snapshot["density"].shape
                diagnostics = data.coarsening_diagnostics
                self._raw_data["sample_uid"].append(data.sample_uid or "")
                self._raw_data["parent_hr_uid"].append(data.parent_hr_uid or "")
                self._raw_data["operator_id"].append(
                    data.coarsening_metadata.get(
                        "operator",
                        data.metadata.get("derivation", {}).get("operator_id", "unknown"),
                    )
                )
                self._raw_data["mass_filter"].append(
                    float(data.coarsening_metadata.get("mass_filter", np.nan))
                )
                self._raw_data["h_eff"].append(
                    np.asarray(
                        diagnostics.get(
                            "effective_support_scale", np.full(output_shape, np.nan)
                        ),
                        dtype=np.float64,
                    )
                )
                self._raw_data["n_eff"].append(
                    np.asarray(
                        diagnostics.get(
                            "effective_source_count", np.full(output_shape, np.nan)
                        ),
                        dtype=np.float64,
                    )
                )
                self._raw_data["sample_metadata"].append(data.metadata)
                self._group_labels.append(group_idx) # 为每个galaxydata添加group标签

        return self

    def drop_invalid( # 专门剔除“有毒”数据。在天体物理模拟中，黑洞附近极端的物理条件有时会导致数值溢出，产生 NaN（非数）或 Inf（无穷大）。
        self,
        *,
        require_positive_targets: bool = True, 
        # require_positive_mbh: bool = True, # 要求黑洞质量必须大于0
        # require_positive_bondi: bool = True, # 要求吸积率必须大于0
        require_positive_ks: bool = True, # 要求KS恒星形成率必须大于0
    ) -> "GalDataSet":
        """
        Removes samples whose target-related scalars contain NaN/Inf (or non-positive values if requested).
        Args:
            require_positive_targets (bool): Enforce mdot_macer and mdot_edd to be strictly positive.
            require_positive_mbh (bool): Enforce mbh to be strictly positive.
        """
        self.__verify_data_loaded()
        sfr_target = np.asarray(self._raw_data["sfr_target"], dtype=float) # 将rawdata中的数据转化为一维数组
        # mdot_edd = np.asarray(self._raw_data["mdot_edd"], dtype=float)
        # mdot_bondi = np.asarray(self._raw_data["mdot_bondi"], dtype=float)
        sfr_ks_val = np.asarray(self._raw_data["sfr_ks"], dtype=float)
        # mbh = np.asarray(self._raw_data["mbh"], dtype=float)

        mask = np.isfinite(sfr_target) & np.isfinite(sfr_ks_val)
        # mask = np.isfinite(sfr_target) & np.isfinite(mdot_edd) & np.isfinite(mdot_bondi) & np.isfinite(mbh)
        if require_positive_targets:
            mask &= (sfr_target > 0.0)
        else:
            mask &= (sfr_target >= 0.0)
        # if require_positive_mbh:
        #     mask &= mbh > 0.0
        # if require_positive_bondi:
        #     mask &= mdot_bondi > 0.0
        if require_positive_ks:
            mask &= sfr_ks_val > 0.0
        else:
            mask &= sfr_ks_val >= 0.0

        removed = np.count_nonzero(~mask)
        if removed:
            if removed == len(mask):
                raise ValueError("All samples are invalid after filtering; check the raw dataset.")
            self.__subset(np.flatnonzero(mask))
        return self

    def exclude_raw_snapshot_indices(self, indices) -> "GalDataSet":
        """按源 snapshot index 精确排除初始化输出，不误删晚期零 SFR。"""

        self.__verify_data_loaded()
        excluded = {int(value) for value in indices}
        if not excluded:
            return self
        raw_indices = []
        for metadata in self._raw_data["sample_metadata"]:
            source = metadata.get("source", {}) if isinstance(metadata, dict) else {}
            if "raw_snapshot_index" not in source:
                raise ValueError(
                    "按 raw_snapshot_index 排除样本需要完整 source metadata；"
                    "请重新生成 processed HDF5。"
                )
            raw_indices.append(int(source["raw_snapshot_index"]))
        keep = np.asarray(
            [value not in excluded for value in raw_indices], dtype=bool
        )
        if not np.any(keep):
            raise ValueError(
                f"排除 raw_snapshot_index={sorted(excluded)} 后没有剩余样本。"
            )
        return self.__subset(np.flatnonzero(keep))

    def filter(self, threshold: float) -> "GalDataSet": # 过滤掉y小于一定值的样本，需要输入一个threshold
        """
        Filters out data points where the target variable y is below the specified threshold.
        Args:
            threshold (float): Threshold value for filtering y.
        """
        self.__verify_data_loaded()
        mask = self.y >= threshold
        indices = np.flatnonzero(mask)
        # If no samples pass the threshold, retain an empty dataset consistently
        return self.__subset(indices)

    def mirror_data(self) -> "GalDataSet":
        """
        它利用 NumPy 的 np.fliplr 和 np.hstack 将物理场和坐标场沿着第一个轴（通常是横轴或方位角）进行镜像拼接。
        对于极坐标下的盘状星系，这相当于利用物理对称性把半个星系的模拟数据补全成整个星系，让模型学到更完整的空间结构。
        """
        """
        Mirrors the data along the first coordinate axis (e.g., x-axis).
        """
        self.__verify_data_loaded()

        for key in self._x_keys:
            self._raw_data[key] = [np.hstack([np.fliplr(x), x]) for x in self._raw_data[key]]

        for key in self._coord_keys:
            self._raw_data[key] = [np.hstack([np.fliplr(x), x]) for x in self._raw_data[key]]
        return self

    def split( # 数据集划分，将一整块数据集切分为训练集、验证集和测试集。
        self,
        train_size: float,
        validation_size: float = 0.0,
        stratify: str | np.ndarray | None = None,
        *,
        n_bins: int = 10,
        random_state: int | None = None,
        shuffle: bool = True,
        group_by_parent_uid: bool = False,
    ):
        """
        Split dataset into train / validation / test sets.

        *** stratify options:
            None        : random split
            "groups"    : stratify by self._group_labels
            "y"         : stratify by y (equal-frequency bins)
            array_like  : external 1D labels
        """
        self.__verify_data_loaded()

        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1.")
        if validation_size < 0 or train_size + validation_size >= 1:
            raise ValueError("validation_size must be non-negative and train_size + validation_size < 1.")

        rng = np.random.default_rng(random_state)
        indices = np.arange(len(self))

        labels = self._build_stratified_labels(stratify, n_bins) # private method 中构建了_build_stratified_labels函数，用于构建stratify,这是对数据分成训练集、验证集和测试集的依据
        if group_by_parent_uid:
            parent_uid = self.parent_hr_uid
            if np.any(parent_uid == ""):
                raise ValueError(
                    "按 parent_hr_uid 划分需要所有样本具有完整 metadata。"
                )
            member_map: dict[str, list[int]] = {}
            for sample_index, uid in enumerate(parent_uid.tolist()):
                member_map.setdefault(uid, []).append(sample_index)
            grouped_members = list(member_map.values())
            indices = np.arange(len(grouped_members))
            if labels is not None:
                grouped_labels = []
                for members in grouped_members:
                    member_labels = np.asarray(labels)[members]
                    if not np.all(member_labels == member_labels[0]):
                        raise ValueError(
                            "同一 parent_hr_uid 的多分辨率变体具有不一致的分层标签。"
                        )
                    grouped_labels.append(member_labels[0])
                labels = np.asarray(grouped_labels)
        train_idx, val_idx, test_idx = [], [], []
        if labels is None:
            tr, va, te = self._split_group(indices, train_size, validation_size, rng, shuffle)
            train_idx.append(tr)
            val_idx.append(va)
            test_idx.append(te)
        else:
            for lv in np.unique(labels):
                if labels.dtype == object:
                    mask = np.array([x == lv for x in labels])
                else:
                    mask = labels == lv
                group_idx = indices[mask]
                tr, va, te = self._split_group(group_idx, train_size, validation_size, rng, shuffle)
                if tr.size:
                    train_idx.append(tr)
                if validation_size > 0 and va.size:
                    val_idx.append(va)
                if te.size:
                    test_idx.append(te)

        train_idx = np.concatenate(train_idx) if train_idx else np.empty(0, dtype=int)
        val_idx = np.concatenate(val_idx) if validation_size > 0 and val_idx else np.empty(0, dtype=int)
        test_idx = np.concatenate(test_idx) if test_idx else np.empty(0, dtype=int)

        if shuffle:
            rng.shuffle(train_idx)
            if validation_size > 0:
                rng.shuffle(val_idx)
            rng.shuffle(test_idx)

        if group_by_parent_uid:
            def expand_group_indices(group_indices):
                if len(group_indices) == 0:
                    return np.empty(0, dtype=int)
                return np.concatenate(
                    [np.asarray(grouped_members[index], dtype=int) for index in group_indices]
                )

            train_idx = expand_group_indices(train_idx)
            val_idx = expand_group_indices(val_idx)
            test_idx = expand_group_indices(test_idx)

        train_set = self.__subset_by_idx(train_idx)
        val_set = self.__subset_by_idx(val_idx) if validation_size > 0 else None
        test_set = self.__subset_by_idx(test_idx)

        return (train_set, val_set, test_set) if validation_size > 0 else (train_set, test_set)

    def standardize( #消除不同物理量量纲差异带来的影响，加速神经网络收敛。
        self, mean: np.ndarray = None, std: np.ndarray = None, log_transform: bool = True
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardizes the input features x to have zero mean and unit variance.
        Args:
            mean (np.ndarray, optional): Precomputed mean for each feature. If None, computed from data.
            std (np.ndarray, optional): Precomputed standard deviation for each feature. If None, computed from data.
            log_transform (bool): Whether to apply log1p transformation before standardization.
        """
        self.__verify_data_loaded()
        x = self.x.copy()  # Use copy to avoid modifying original data
        if log_transform: # 对非负物理场应用log（1+x）变换，压缩极端数值
            # Safely apply log1p only to non-negative channels (scalars like density, temperature, etc.)
            # Velocities and other signed quantities should remain linear.
            # We check the minimum value across all samples for each channel.
            for i in range(x.shape[1]):
                if np.min(x[:, i]) >= 0:
                    x[:, i] = np.log1p(x[:, i])

        if mean is None or std is None:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            std = np.std(x, axis=(0, 2, 3), keepdims=True)
            std_safe = np.where(std == 0, 1.0, std)
            x = (x - mean) / std_safe
            return x, mean, std_safe
        elif mean is not None and std is not None: # 标准化，使得测试集上应用完全相同的缩放比例。
            std_safe = np.where(std == 0, 1, std)
            x = (x - mean) / std_safe
            return x
        else:
            raise ValueError("Both mean and std must be provided for standardization.")

    def balance_groups(self, method: str = "oversample") -> "GalDataSet":
        """
        解决样本不平衡问题。假设 A 星系模拟了 1000 个快照，B 星系只模拟了 100 个，不处理的话模型会偏向 A 星系。
        用oversample来增加少样本星系的快照，用undersample来随机去除多样本星系的快照
        """
        """
        Balance samples across groups by random over- or under-sampling.

        Args:
            method: "oversample" (with replacement) or "undersample" (without replacement)
        """
        self.__verify_data_loaded()

        if method not in {"oversample", "undersample"}:
            raise ValueError("method must be 'oversample' or 'undersample'")

        # Collect indices by normalized group label
        def normalize_label(label):
            arr = np.asarray(label)
            return arr.item() if arr.ndim == 0 else tuple(arr.tolist())

        indices_by_group = {}
        for i, label in enumerate(self._group_labels):
            key = normalize_label(label)
            indices_by_group.setdefault(key, []).append(i)

        groups = list(indices_by_group.keys())
        group_sizes = np.array([len(indices_by_group[g]) for g in groups], dtype=int)

        # Determine target size per group. Oversampling must retain every
        # original sample once and draw only the missing remainder with
        # replacement. Re-drawing the entire group with replacement would
        # discard about 37% of the largest group's unique samples on average.
        target_size = group_sizes.max() if method == "oversample" else group_sizes.min()
        sampled_indices = []
        for group in groups:
            group_indices = np.asarray(indices_by_group[group], dtype=int)
            if method == "oversample":
                n_extra = target_size - len(group_indices)
                if n_extra > 0:
                    extra_indices = np.random.choice(
                        group_indices,
                        size=n_extra,
                        replace=True,
                    )
                    selected = np.concatenate([group_indices, extra_indices])
                else:
                    selected = group_indices.copy()
            else:
                selected = np.random.choice(
                    group_indices,
                    size=target_size,
                    replace=False,
                )
            sampled_indices.append(selected)

        all_indices = np.concatenate(sampled_indices)
        np.random.shuffle(all_indices)

        return self.__subset(all_indices)

    # ------- Private Methods -------

    def _build_stratified_labels(self, stratify: str | np.ndarray | None, n_bins: int) -> np.ndarray | None: #statify可以是str，也可以是n维数组，也可以是none。
        """
        Constructs stratification labels based on the specified method.
        """
        if stratify is None: # 如果stratify是none，则对数据随机切分
            return None

        if isinstance(stratify, str):
            if stratify == "groups": # 如果stratify是groups，则对数据按星系组别进行切分    
                items = [
                    tuple(np.asarray(g).tolist()) if np.asarray(g).ndim > 0 else np.asarray(g).item()
                    for g in self._group_labels
                ]
                res = np.empty(len(items), dtype=object)
                res[:] = items
                return res

            if stratify == "y": # 如果stratify是y，则按目标变量y（吸积率比值对数）进行切分，切分的依据是分位数
                y = self.y
                quantiles = np.linspace(0, 1, n_bins + 1) # 生成n_bins+1个分位数
                edges = np.unique(np.quantile(y, quantiles))
                if edges.size < 2:
                    edges = np.array([y.min() - 1e-9, y.max() + 1e-9])
                return np.digitize(y, edges[1:-1], right=False)

            raise ValueError("stratify must be None, 'groups', 'y', or a 1D array.")

        labels = np.asarray(stratify) # stratify也可以是用户传入的自定义的一维数组，比如金属丰度
        if labels.shape != (len(self),):
            raise ValueError("Provided stratify array must be 1D and match dataset length.")
        return labels

    def _split_group( 
        self,
        group_idx: np.ndarray,
        train_size: float,
        validation_size: float,
        rng: np.random.Generator,
        shuffle: bool,
    ):
        """
        Splits the provided group indices into train, validation, and test sets.
        """
        if shuffle:
            rng.shuffle(group_idx)

        n = len(group_idx)
        n_train = int(train_size * n)
        n_val = int(validation_size * n) if validation_size > 0 else 0

        # guard against rounding overflow
        excess = n_train + n_val - n
        if excess > 0:
            reduce_val = min(n_val, excess)
            n_val -= reduce_val
            n_train = max(0, n_train - (excess - reduce_val))

        train = group_idx[:n_train] # 进行切片
        val = group_idx[n_train : n_train + n_val]
        test = group_idx[n_train + n_val :]
        return train, val, test

    # ------- Validation Methods -------

    def __verify_data_loaded(self) -> None: # 验证数据是否加载
        if self._raw_data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

    def __subset(self, indices: np.ndarray) -> "GalDataSet": # 原地修改，只保留索引为keep的样本
        """
        Retains only the samples specified by indices across all stored fields.
        """
        keep = indices.tolist()
        for key, values in self._raw_data.items():
            self._raw_data[key] = [values[i] for i in keep]
        if self._group_labels is not None:
            self._group_labels = [self._group_labels[i] for i in keep]
        return self

    def __subset_by_idx(self, idx: np.ndarray) -> "GalDataSet": # 创建一个新副本
        """
        Returns a new GalDataSet instance containing only the samples specified by idx.
        """
        ds = GalDataSet()
        ds._x_keys = self._x_keys.copy()
        ds._coord_keys = self._coord_keys.copy()
        ds._target_metadata = self._target_metadata.copy()
        ds._coarsening_signature = self._coarsening_signature
        ds._coordinate_reference = self._coordinate_reference
        ds._raw_data = {k: [self._raw_data[k][i] for i in idx] for k in self._raw_data}
        ds._group_labels = [self._group_labels[i] for i in idx]
        ds.folder_path = self.folder_path.copy() if self.folder_path is not None else None
        return ds


if __name__ == "__main__":
    import argparse
    from tqdm.contrib import tenumerate

    try:
        from raw_data_processing.parse_log_file import load_config, ll_eg, ll_dg
    except ModuleNotFoundError:
        from parse_log_file import load_config, ll_eg, ll_dg

    configs = load_config(".config") # 从外部加载.config文件
    parser = argparse.ArgumentParser(description="Convert raw MACER snapshots to fine/coarse HDF5 files.")
    parser.add_argument(
        "--galaxy",
        action="append",
        default=[],
        help="Process only this generated name (for example disk_galaxy_fiducial). May be repeated.",
    )
    parser.add_argument(
        "--coarsener",
        choices=("spatial_block", "axisymmetric_rm_v0", "axisymmetric_rm_v1"),
        default="spatial_block",
        help="Coarse-graining operator applied after each fine snapshot.",
    )
    parser.add_argument(
        "--coarse-shape", nargs=2, type=int, default=(8, 8),
        metavar=("N_THETA", "N_R"),
    )
    parser.add_argument(
        "--mass-filter", type=float, default=None,
        help="Required for R_m; units are density*Data-Set-15 volume.",
    )
    parser.add_argument(
        "--kernel", choices=("top_hat", "cubic_spline", "wendland_c2"),
        default=None,
    )
    parser.add_argument(
        "--coarse-dirname", default=None,
        help="Defaults to coarse for R_h and coarse_<operator> for R_m.",
    )
    parser.add_argument("--initial-neighbors", type=int, default=None)
    parser.add_argument("--balance-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--balance-max-iterations", type=int, default=30000,
        help="Rm-v0 IPFP limit (default 30000; ignored by v1).",
    )
    parser.add_argument(
        "--v1-max-neighbors", type=int, default=4096,
        help="Maximum adaptive KNN count for Rm-v1.",
    )
    parser.add_argument("--v1-max-runtime-seconds", type=float, default=60.0)
    parser.add_argument("--v1-solver-max-iterations", type=int, default=300)
    parser.add_argument("--v1-polish-max-iterations", type=int, default=500)
    cli_args = parser.parse_args()
    if any(value <= 0 for value in cli_args.coarse_shape):
        parser.error("--coarse-shape values must be positive.")
    if cli_args.coarsener != "spatial_block" and (
        cli_args.mass_filter is None or cli_args.mass_filter <= 0
    ):
        parser.error("--mass-filter must be positive for axisymmetric R_m operators.")
    if cli_args.coarsener == "spatial_block" and cli_args.kernel is not None:
        parser.error("--kernel applies only to axisymmetric R_m operators.")
    if cli_args.coarsener == "axisymmetric_rm_v0" and cli_args.kernel == "wendland_c2":
        parser.error("R_m-v0 supports top_hat or cubic_spline, not wendland_c2.")
    if cli_args.coarsener == "axisymmetric_rm_v1" and cli_args.kernel not in (None, "wendland_c2"):
        parser.error("R_m-v1 uses the fixed wendland_c2 kernel.")
    if cli_args.balance_tolerance <= 0 or cli_args.balance_max_iterations <= 0:
        parser.error("Balance tolerance and iteration limits must be positive.")
    if cli_args.v1_max_neighbors < 2 or cli_args.v1_max_runtime_seconds <= 0:
        parser.error("v1 neighbour and runtime limits must be positive.")
    selected_galaxies = set(cli_args.galaxy)

    raw_data_dir = configs["BaseConfig"]["raw_data_dir"]
    data_dir = configs["BaseConfig"]["data_dir"]

    ndim = 2 # 2D模拟（MACER2D）
    coordinate_mode = "polar" # 采用极坐标系

    for gal_type, gal_group in configs["RawDataConfig"].items(): # 便利所有星系类型和组
        # if gal_type == "elliptical_galaxy":
        #     continue  # Skip elliptical galaxies for 2D processing
        for gal_nickname, gal_config in gal_group.items():
            gal_name = f"{gal_type}_{gal_nickname}" # 星系是椭圆星系还是盘星系

            if selected_galaxies and gal_name not in selected_galaxies:
                continue

            hdfra_folder = os.path.join(raw_data_dir, gal_config["folder_name"], "data")
            if not os.path.exists(hdfra_folder):
                # 如果 "data" 子文件夹不存在，尝试直接在 folder_name 寻找（适配您当前的文件结构）
                hdfra_folder = os.path.join(raw_data_dir, gal_config["folder_name"])
            
            if not os.path.exists(hdfra_folder):
                print(f"  [Warning] Data folder not found for {gal_name} at {hdfra_folder}. Skipping...")
                continue
                
            hdfra_files = [f for f in os.listdir(hdfra_folder) if f.startswith("hdfra.")]
            if not hdfra_files:
                print(f"  [Warning] No hdfra files found for {gal_name} in {hdfra_folder}. Skipping...")
                continue
                
            def _snapshot_sort_key(name):
                suffix = name.rsplit(".", 1)[-1]
                return (0, int(suffix)) if suffix.isdigit() else (1, name)

            hdfra_paths = sorted(hdfra_files, key=_snapshot_sort_key)
            log_path = os.path.join(data_dir, f"{gal_name}.parquet")
            func_parse_log = ll_eg if "elliptical" in gal_type else ll_dg # 按照星系类型（椭圆or盘）选择不同的日志解析函数
            hdfra_config = configs["HdfraConfig"][gal_type]
            corr = hdfra_config["correspondence"]
            target_config = hdfra_config.get("target")
            if target_config is not None and target_config["dataset"] in corr:
                raise ValueError(
                    f"Target dataset {target_config['dataset']!r} for {gal_type} is also listed "
                    "in correspondence, which would create direct label leakage."
                )

            # Build a one-to-one mapping from each sorted input file to its
            # physical time and processed output index. filename_index mode
            # reads the global index from hdfra.<index>, so missing snapshots
            # remain real time gaps instead of shifting all later times.
            time_seq, output_indices = _build_snapshot_schedule(hdfra_paths, gal_config)
            scope = float(gal_config["snapshot_dt"]) * 0.5

            save_path_fine = os.path.join(data_dir, gal_name, "fine")
            os.makedirs(save_path_fine, exist_ok=True)
            coarse_dirname = cli_args.coarse_dirname or (
                "coarse" if cli_args.coarsener == "spatial_block"
                else f"coarse_{cli_args.coarsener}"
            )
            save_path_coarse = os.path.join(data_dir, gal_name, coarse_dirname)
            os.makedirs(save_path_coarse, exist_ok=True)

            stable_from_start = gal_name in [
                "disk_galaxy_fiducial",
                "disk_galaxy_supplement",
                "disk_galaxy_fiducial_4",
                "disk_galaxy_fiducial_7",
            ]
            if gal_config.get("mode") == "filename_index":
                expected_output_names = {
                    _snapshot_output_name(snapshot_index)
                    for index, snapshot_index in enumerate(output_indices)
                    if stable_from_start or index > 45
                }
                _reject_stale_filename_index_outputs(save_path_fine, expected_output_names)
                _reject_stale_filename_index_outputs(save_path_coarse, expected_output_names)

            for index, file_name in tenumerate(hdfra_paths, desc=f"Processing {gal_name}"):
                if not stable_from_start and index <= 45: # 跳过最初的45个不稳定的快照
                    continue  # Skip first unstable snapshots

                file_path = os.path.join(hdfra_folder, file_name)
                output_name = _snapshot_output_name(output_indices[index])
                file = GalData(ndim=ndim, coordinate_mode=coordinate_mode) # 创建GalData类
                file.set_corr(corr).read_hdfra(
                    hdfra_path=file_path,
                    time=time_seq[index],
                    log_path=log_path,
                    func_parse_log=func_parse_log,
                    scope=scope,
                    target_config=target_config,
                    source_metadata={
                        "simulation_code": "MACER2D",
                        "galaxy_type": gal_type,
                        "galaxy_id": gal_name,
                        "run_id": gal_name,
                        "source_series_id": gal_config["folder_name"],
                        "raw_snapshot_index": int(output_indices[index]),
                        "source_file_id": file_name,
                        "source_file_sha256": sha256_file(file_path),
                        "source_file_size_bytes": int(os.path.getsize(file_path)),
                        "time": float(time_seq[index]),
                        "time_units": "Gyr",
                    },
                )
                file.set_coord(["fakeDim2", "fakeDim1"]) # 创建坐标轴，并用set_coord重新指定物理网格的坐标轴映射
                file.save_h5(os.path.join(save_path_fine, output_name)) # 将原始分辨率数据的HDF5文件保存至fine文件夹

                operator_kwargs = {}
                if cli_args.mass_filter is not None:
                    operator_kwargs["mass_filter"] = cli_args.mass_filter
                if cli_args.kernel is not None and cli_args.coarsener == "axisymmetric_rm_v0":
                    operator_kwargs["kernel"] = cli_args.kernel
                if cli_args.coarsener == "axisymmetric_rm_v0":
                    operator_kwargs.update(
                        balance_tolerance=cli_args.balance_tolerance,
                        balance_max_iterations=cli_args.balance_max_iterations,
                    )
                    if cli_args.initial_neighbors is not None:
                        operator_kwargs["initial_neighbors"] = cli_args.initial_neighbors
                elif cli_args.coarsener == "axisymmetric_rm_v1":
                    operator_kwargs.update(
                        balance_tolerance=cli_args.balance_tolerance,
                        max_neighbors=cli_args.v1_max_neighbors,
                        max_runtime_seconds=cli_args.v1_max_runtime_seconds,
                        solver_max_iterations=cli_args.v1_solver_max_iterations,
                        polish_max_iterations=cli_args.v1_polish_max_iterations,
                    )
                    if cli_args.initial_neighbors is not None:
                        operator_kwargs["initial_neighbors"] = cli_args.initial_neighbors
                file_coarse = file.coarsen(
                    cli_args.coarsener,
                    new_shape=tuple(cli_args.coarse_shape),
                    **operator_kwargs,
                )
                file_coarse.save_h5(os.path.join(save_path_coarse, output_name))
