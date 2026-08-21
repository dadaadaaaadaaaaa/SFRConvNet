"""所需空间粗粒化后输出形状的 R_h 块所包含的气体质量的中位数，估算有用的 R_m-v0 质量滤波器值。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from raw_data_processing.galaxy_data import GalData
except ModuleNotFoundError:
    from galaxy_data import GalData


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate R_m-v0 mass-filter values by matching an R_h information budget."
    )
    parser.add_argument("--input", required=True, type=Path, help="Fine HDF5 file or directory.")
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        default=(8, 8),
        metavar=("N_THETA", "N_R"),
        help="Reference R_h output shape (default: 8 8).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N files in directory mode.",
    )
    args = parser.parse_args()
    if any(value <= 0 for value in args.shape):
        parser.error("--shape values must be positive.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    return args


def _files(input_path: Path, limit: int | None):
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    paths = sorted(input_path.glob("*.h5"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No .h5 files found in {input_path}")
    return paths


def main():
    args = _parse_args()
    paths = _files(args.input, args.limit)
    block_masses = []
    fine_masses = []
    for path in paths:
        fine = GalData().load_h5(path)
        mass = np.asarray(fine.gas_mass, dtype=np.float64)
        if any(old % new != 0 for old, new in zip(mass.shape, args.shape)):
            raise ValueError(
                f"{path}: source shape {mass.shape} is not divisible by {tuple(args.shape)}"
            )
        block_masses.append(GalData._block_reduce_sum(mass, tuple(args.shape)).ravel())
        fine_masses.append(mass.ravel())

    block_masses = np.concatenate(block_masses)
    fine_masses = np.concatenate(fine_masses)
    quantiles = (0.05, 0.25, 0.50, 0.75, 0.95)
    print(f"snapshots={len(paths)}, reference_shape={tuple(args.shape)}")
    print("quantile       5%          25%          50%          75%          95%")
    print(
        "fine_cell "
        + " ".join(f"{value:12.5g}" for value in np.quantile(fine_masses, quantiles))
    )
    print(
        "R_h_block "
        + " ".join(f"{value:12.5g}" for value in np.quantile(block_masses, quantiles))
    )
    reference = float(np.median(block_masses))
    print(f"\nM0 = median(R_h block mass) = {reference:.8g}")
    print("Suggested exploratory mass_filter values (same code units):")
    for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
        print(f"  {factor:4.2g} M0 = {factor * reference:.8g}")


if __name__ == "__main__":
    main()
