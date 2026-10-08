"""从 processed fine HDF5 构建 R_h、R_m-v0 或 R_m-v1 coarse HDF5。

该脚本是 ``fine -> coarse`` 的统一入口。它先严格验证 fine metadata 与
``identity_fine`` 来源，再调用 ``GalData.coarsen``，因此三种粗粒化方法共用
同一套守恒检查、SFR baseline 和 ``Y/B/C_true`` 数据契约。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from raw_data_processing.galaxy_data import GalData
    from raw_data_processing.metadata_schema import validate_complete_metadata
    from raw_data_processing.parse_log_file import load_config
except ModuleNotFoundError:  # 兼容从 raw_data_processing 目录直接运行。
    from galaxy_data import GalData
    from metadata_schema import validate_complete_metadata
    from parse_log_file import load_config


SUPPORTED_COARSENERS = (
    "spatial_3d",
    "spatial_block",
    "axisymmetric_rm_v0",
    "axisymmetric_rm_v1",
)


def _processed_sort_key(path: Path) -> tuple[int, int | str]:
    """优先按 HDF5 文件名的整数 stem 排序。"""

    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name)


def _configured_galaxy_names(configs: dict) -> list[str]:
    """返回配置中的所有生成名；非 HDFRA 运行随后由 fine 目录是否存在决定。"""

    return [
        f"{galaxy_type}_{nickname}"
        for galaxy_type, group in configs["RawDataConfig"].items()
        for nickname in group
    ]


def _validate_fine_input(galaxy: GalData, source_path: Path) -> None:
    """拒绝 legacy/incomplete 或已经粗粒化的输入，防止 coarse→coarse。"""

    metadata = galaxy.metadata
    validate_complete_metadata(metadata)
    operator_id = metadata.get("derivation", {}).get("operator_id")
    if operator_id != "identity_fine":
        raise ValueError(
            f"{source_path} 不是 fine identity 产品：operator_id={operator_id!r}。"
        )
    required_fields = {
        "density",
        "gas_energy",
        "cooling_time",
        "cooling_rate_density",
        "radial_acceleration",
    }
    missing = sorted(required_fields.difference(galaxy.snapshot))
    if missing:
        raise ValueError(f"{source_path} 缺少构造 LR baseline 所需字段：{missing}")
    # 访问属性会同时确认 canonical supervision 已存在且可读。
    _ = galaxy.sfr_target
    _ = galaxy.sfr_macer_baseline
    _ = galaxy.sfr_closure_true


def _operator_kwargs(
    *,
    coarsener: str,
    mass_filter: float | None,
    kernel: str | None,
    initial_neighbors: int | None,
    balance_tolerance: float,
    balance_max_iterations: int,
    v1_max_neighbors: int,
    v1_max_runtime_seconds: float,
    v1_solver_max_iterations: int,
    v1_polish_max_iterations: int,
) -> dict:
    """把稳定 CLI 参数映射到 ``GalData.coarsen`` 的算子专用参数。"""

    kwargs: dict[str, object] = {}
    if mass_filter is not None:
        kwargs["mass_filter"] = mass_filter
    if coarsener == "axisymmetric_rm_v0":
        if kernel is not None:
            kwargs["kernel"] = kernel
        kwargs.update(
            balance_tolerance=balance_tolerance,
            balance_max_iterations=balance_max_iterations,
        )
        if initial_neighbors is not None:
            kwargs["initial_neighbors"] = initial_neighbors
    elif coarsener == "axisymmetric_rm_v1":
        kwargs.update(
            balance_tolerance=balance_tolerance,
            max_neighbors=v1_max_neighbors,
            max_runtime_seconds=v1_max_runtime_seconds,
            solver_max_iterations=v1_solver_max_iterations,
            polish_max_iterations=v1_polish_max_iterations,
        )
        if initial_neighbors is not None:
            kwargs["initial_neighbors"] = initial_neighbors
    return kwargs


def validate_coarsening_options(
    *,
    coarsener: str,
    coarse_shape: tuple[int, int],
    mass_filter: float | None,
    kernel: str | None,
    initial_neighbors: int | None,
    balance_tolerance: float,
    balance_max_iterations: int,
    v1_max_neighbors: int,
    v1_max_runtime_seconds: float,
    v1_solver_max_iterations: int,
    v1_polish_max_iterations: int,
) -> None:
    """统一验证命令行和 Python API 的粗粒化参数。"""

    if coarsener not in SUPPORTED_COARSENERS:
        raise ValueError(f"不支持的 coarsener：{coarsener!r}")
    if len(coarse_shape) != 2 or any(value <= 0 for value in coarse_shape):
        raise ValueError("coarse_shape 的两个值都必须为正整数。")
    if coarsener != "spatial_block" and (mass_filter is None or mass_filter <= 0):
        raise ValueError("axisymmetric R_m 算子要求正的 mass_filter。")
    if coarsener == "spatial_block" and kernel is not None:
        raise ValueError("kernel 只适用于 axisymmetric R_m 算子。")
    if coarsener == "spatial_block" and mass_filter is not None:
        raise ValueError("mass_filter 只适用于 axisymmetric R_m 算子。")
    if coarsener == "axisymmetric_rm_v0" and kernel == "wendland_c2":
        raise ValueError("R_m-v0 只支持 top_hat 或 cubic_spline。")
    if coarsener == "axisymmetric_rm_v1" and kernel not in (None, "wendland_c2"):
        raise ValueError("R_m-v1 固定使用 wendland_c2。")
    if initial_neighbors is not None and initial_neighbors < 2:
        raise ValueError("initial_neighbors 至少为 2。")
    positive_values = {
        "balance_tolerance": balance_tolerance,
        "balance_max_iterations": balance_max_iterations,
        "v1_max_neighbors": v1_max_neighbors,
        "v1_max_runtime_seconds": v1_max_runtime_seconds,
        "v1_solver_max_iterations": v1_solver_max_iterations,
        "v1_polish_max_iterations": v1_polish_max_iterations,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"以下参数必须为正：{invalid}")


def build_coarse_from_fine(
    *,
    config_dir: str | os.PathLike[str] = ".config",
    galaxies: Iterable[str] = (),
    coarsener: str = "spatial_block",
    coarse_shape: tuple[int, ...] | None = None,
    mass_filter: float | None = None,
    kernel: str | None = None,
    coarse_dirname: str | None = None,
    fine_dirname: str = "fine",
    snapshot_indices: Iterable[int] = (),
    max_snapshots: int | None = None,
    initial_neighbors: int | None = None,
    balance_tolerance: float = 1e-6,
    balance_max_iterations: int = 30000,
    v1_max_neighbors: int = 4096,
    v1_max_runtime_seconds: float = 60.0,
    v1_solver_max_iterations: int = 300,
    v1_polish_max_iterations: int = 500,
    prediction_mode: str = "direct",
) -> list[Path]:
    """从 fine 目录构建 coarse 数据，并返回本次写出的文件路径。"""

    if coarsener == "spatial_3d":
        if mass_filter is not None or kernel is not None or initial_neighbors is not None:
            raise ValueError("三维空间块算子不接受 R_m 参数。")
        from raw_data_processing.build_snapshots_3d import build_coarse_3d
        return build_coarse_3d(config_dir=config_dir,galaxies=galaxies,
            shape=(32,16,32) if coarse_shape is None else coarse_shape,
            mode=prediction_mode,fine_dirname=fine_dirname,coarse_dirname=coarse_dirname,
            indices=snapshot_indices,max_snapshots=max_snapshots)
    coarse_shape = tuple(int(value) for value in ((8,8) if coarse_shape is None else coarse_shape))
    validate_coarsening_options(
        coarsener=coarsener,
        coarse_shape=coarse_shape,
        mass_filter=mass_filter,
        kernel=kernel,
        initial_neighbors=initial_neighbors,
        balance_tolerance=balance_tolerance,
        balance_max_iterations=balance_max_iterations,
        v1_max_neighbors=v1_max_neighbors,
        v1_max_runtime_seconds=v1_max_runtime_seconds,
        v1_solver_max_iterations=v1_solver_max_iterations,
        v1_polish_max_iterations=v1_polish_max_iterations,
    )
    if max_snapshots is not None and max_snapshots <= 0:
        raise ValueError("max_snapshots 必须为正整数或 None。")
    for name, value in (("fine_dirname", fine_dirname), ("coarse_dirname", coarse_dirname)):
        if value is not None and (not value or Path(value).name != value):
            raise ValueError(f"{name} 必须是单层目录名，不能包含路径分隔符。")

    configs = load_config(str(config_dir))
    required_sections = {"BaseConfig", "RawDataConfig"}
    missing_sections = required_sections.difference(configs)
    if missing_sections:
        raise KeyError(f"配置目录缺少 JSON 配置：{sorted(missing_sections)}")
    data_dir = Path(configs["BaseConfig"]["data_dir"])
    configured = _configured_galaxy_names(configs)
    selected = set(galaxies)
    unknown = sorted(selected.difference(configured))
    if unknown:
        raise ValueError(f"--galaxy 与配置中的生成名不匹配：{unknown}")
    galaxy_names = [name for name in configured if not selected or name in selected]
    requested_indices = {int(value) for value in snapshot_indices}
    if any(value < 0 for value in requested_indices):
        raise ValueError("snapshot_indices 不能包含负数。")
    output_name = coarse_dirname or (
        "coarse" if coarsener == "spatial_block" else f"coarse_{coarsener}"
    )
    kwargs = _operator_kwargs(
        coarsener=coarsener,
        mass_filter=mass_filter,
        kernel=kernel,
        initial_neighbors=initial_neighbors,
        balance_tolerance=balance_tolerance,
        balance_max_iterations=balance_max_iterations,
        v1_max_neighbors=v1_max_neighbors,
        v1_max_runtime_seconds=v1_max_runtime_seconds,
        v1_solver_max_iterations=v1_solver_max_iterations,
        v1_polish_max_iterations=v1_polish_max_iterations,
    )
    written: list[Path] = []

    for galaxy_name in galaxy_names:
        if galaxy_name.startswith("dwarf_galaxy_"):
            if selected:
                raise ValueError("矮星系请指定 --coarsener spatial_3d。")
            continue
        input_dir = data_dir / galaxy_name / fine_dirname
        if not input_dir.exists():
            print(f"[跳过] {galaxy_name}：找不到 fine 目录 {input_dir}")
            continue
        input_paths = sorted(input_dir.glob("*.h5"), key=_processed_sort_key)
        if requested_indices:
            available = {int(path.stem) for path in input_paths if path.stem.isdigit()}
            missing = sorted(requested_indices.difference(available))
            if missing:
                raise ValueError(f"{galaxy_name} 的 fine 目录缺少 snapshot index：{missing}")
            input_paths = [
                path for path in input_paths
                if path.stem.isdigit() and int(path.stem) in requested_indices
            ]
        if max_snapshots is not None:
            input_paths = input_paths[:max_snapshots]
        if not input_paths:
            print(f"[跳过] {galaxy_name}：没有待处理 fine HDF5")
            continue

        output_dir = data_dir / galaxy_name / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{galaxy_name}] fine→coarse ({coarsener})：{len(input_paths)} 个 snapshot")
        for source_path in input_paths:
            fine = GalData().load_h5(str(source_path))
            _validate_fine_input(fine, source_path)
            coarse = fine.coarsen(coarsener, new_shape=coarse_shape, **kwargs)
            output_path = output_dir / source_path.name
            coarse.save_h5(str(output_path))
            written.append(output_path)

    return written


def build_argument_parser() -> argparse.ArgumentParser:
    """构造 CLI；旧入口已有的粗粒化参数名和默认值保持不变。"""

    parser = argparse.ArgumentParser(
        description="Build coarse MACER HDF5 files from processed fine HDF5 files."
    )
    parser.add_argument("--galaxy", action="append", default=[])
    parser.add_argument("--config-dir", default=".config")
    parser.add_argument("--fine-dirname", default="fine")
    parser.add_argument("--prediction-mode", choices=("direct", "residual"), default="direct", help="三维 Direct/Residual；二维行为不变。")
    parser.add_argument("--snapshot-index", action="append", default=[], type=int)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--coarsener", choices=SUPPORTED_COARSENERS, default="spatial_block")
    parser.add_argument(
        "--coarse-shape", nargs="+", type=int, default=None,
        metavar="N", help="二维默认 8 8；spatial_3d 默认 32 16 32，顺序 r theta phi。",
    )
    parser.add_argument("--mass-filter", type=float, default=None)
    parser.add_argument(
        "--kernel", choices=("top_hat", "cubic_spline", "wendland_c2"), default=None
    )
    parser.add_argument("--coarse-dirname", default=None)
    parser.add_argument("--initial-neighbors", type=int, default=None)
    parser.add_argument("--balance-tolerance", type=float, default=1e-6)
    parser.add_argument("--balance-max-iterations", type=int, default=30000)
    parser.add_argument("--v1-max-neighbors", type=int, default=4096)
    parser.add_argument("--v1-max-runtime-seconds", type=float, default=60.0)
    parser.add_argument("--v1-solver-max-iterations", type=int, default=300)
    parser.add_argument("--v1-polish-max-iterations", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.max_snapshots is not None and args.max_snapshots <= 0:
        parser.error("--max-snapshots 必须为正整数。")
    if any(index < 0 for index in args.snapshot_index):
        parser.error("--snapshot-index 不能为负数。")
    try:
        written = build_coarse_from_fine(
            config_dir=args.config_dir,
            galaxies=args.galaxy,
            coarsener=args.coarsener,
            coarse_shape=None if args.coarse_shape is None else tuple(args.coarse_shape),
            prediction_mode=args.prediction_mode,
            mass_filter=args.mass_filter,
            kernel=args.kernel,
            coarse_dirname=args.coarse_dirname,
            fine_dirname=args.fine_dirname,
            snapshot_indices=args.snapshot_index,
            max_snapshots=args.max_snapshots,
            initial_neighbors=args.initial_neighbors,
            balance_tolerance=args.balance_tolerance,
            balance_max_iterations=args.balance_max_iterations,
            v1_max_neighbors=args.v1_max_neighbors,
            v1_max_runtime_seconds=args.v1_max_runtime_seconds,
            v1_solver_max_iterations=args.v1_solver_max_iterations,
            v1_polish_max_iterations=args.v1_polish_max_iterations,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"fine→coarse 完成：写出 {len(written)} 个文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
