"""Build axisymmetric R_m-v0 datasets from processed MACER fine snapshots.

Example (run from the repository root):

    python raw_data_processing/build_rm_v0.py \
        --input data/disk_galaxy_fiducial/fine/00000.h5 \
        --output data/disk_galaxy_fiducial/coarse_rm_v0_m1e6/00000.h5 \
        --shape 8 8 --mass-filter 1e6

Directory mode preserves input filenames and refuses to overwrite by default.
The mass value must use the same code units as ``density * cell_volume``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from raw_data_processing.galaxy_data import GalData
except ModuleNotFoundError:
    from galaxy_data import GalData


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Apply the conservative axisymmetric R_m-v0 operator to processed MACER HDF5 data."
    )
    parser.add_argument("--input", required=True, type=Path, help="Fine HDF5 file or directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output HDF5 file or directory.")
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        default=(8, 8),
        metavar=("N_THETA", "N_R"),
        help="Regular CNN output shape in snapshot array order (default: 8 8).",
    )
    parser.add_argument(
        "--mass-filter",
        required=True,
        type=float,
        help="Axisymmetric enclosed ring-mass budget in density*volume code units.",
    )
    parser.add_argument(
        "--kernel",
        choices=("top_hat", "cubic_spline"),
        default="top_hat",
        help="Compact support profile (default: top_hat for the v0 experiment).",
    )
    parser.add_argument(
        "--initial-neighbors",
        type=int,
        default=32,
        help="Initial cKDTree query size before adaptive doubling.",
    )
    parser.add_argument(
        "--balance-tolerance",
        type=float,
        default=1e-6,
        help="Maximum relative row/column marginal error (default: 1e-6).",
    )
    parser.add_argument(
        "--balance-max-iterations",
        type=int,
        default=5000,
        help="Maximum sparse transport balancing iterations (default: 5000).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files in directory mode (useful for smoke tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    args = parser.parse_args()
    if any(value <= 0 for value in args.shape):
        parser.error("--shape values must be positive.")
    if args.mass_filter <= 0:
        parser.error("--mass-filter must be positive.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if args.balance_tolerance <= 0:
        parser.error("--balance-tolerance must be positive.")
    if args.balance_max_iterations <= 0:
        parser.error("--balance-max-iterations must be positive.")
    return args


def _resolve_jobs(input_path: Path, output_path: Path, limit: int | None):
    if input_path.is_file():
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / input_path.name
        return [(input_path, output_path)]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    files = sorted(input_path.glob("*.h5"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {input_path}")
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("Directory input requires a directory output.")
    output_path.mkdir(parents=True, exist_ok=True)
    return [(path, output_path / path.name) for path in files]


def main():
    args = _parse_args()
    jobs = _resolve_jobs(args.input, args.output, args.limit)
    print(
        f"R_m-v0: files={len(jobs)}, shape={tuple(args.shape)}, "
        f"mass_filter={args.mass_filter:.8g}, kernel={args.kernel}"
    )
    for index, (input_file, output_file) in enumerate(jobs, start=1):
        if output_file.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_file}; pass --overwrite explicitly."
            )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fine = GalData().load_h5(input_file)
        coarse = fine.rescale_mass_adaptive(
            tuple(args.shape),
            mass_filter=args.mass_filter,
            kernel=args.kernel,
            initial_neighbors=args.initial_neighbors,
            balance_tolerance=args.balance_tolerance,
            balance_max_iterations=args.balance_max_iterations,
        )
        coarse.save_h5(output_file)
        metadata = coarse.coarsening_metadata
        print(
            f"[{index}/{len(jobs)}] {input_file.name} -> {output_file} "
            f"(balance_error={metadata['transport_balance_error']:.3e}, "
            f"iterations={metadata['transport_balance_iterations']})"
        )


if __name__ == "__main__":
    main()
