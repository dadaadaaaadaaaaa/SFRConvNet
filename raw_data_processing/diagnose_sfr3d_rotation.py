"""只诊断旋转主导格点的残差，不将由监督值反推的引力用于正式算子。"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_data_processing.athdf_reader import read_athdf
from raw_data_processing.macer_sfr_operator_3d import calculate_macer_sfr_3d, validate_gravity_geometry
from raw_data_processing.validate_macer_sfr_3d import input_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gravity", type=Path, default=Path("Fid_yuan18.out1.00000.athdf"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    params = input_parameters(args.input)
    gravity = read_athdf(args.gravity, fields=("grav",), dtype=np.float64)
    # hst 无表头：下列列号仅作为数值对照，不赋予已确认的物理语义。
    hst = np.loadtxt(args.raw_dir / "yuan18.hst", usecols=(0, 19, 26, 29))
    rows = []
    for index in (51, 200, 299):
        snap = read_athdf(args.raw_dir / f"yuan18.out1.{index:05d}.athdf",
                         fields=("rho", "press", "metallicity", "cool_rate", "rhodotsf"), dtype=np.float64)
        validate_gravity_geometry(snap, gravity)
        f = snap.fields
        y, v = f['rhodotsf'], snap.geometry.cell_volume
        radius = snap.geometry.centers[0][:, None, None]
        result = calculate_macer_sfr_3d(density=f['rho'], pressure=f['press'], metallicity=f['metallicity'],
            cooling_rate=f['cool_rate'], cell_volume=v, radius=radius,
            radial_acceleration=gravity.fields['grav'], parameters=params)
        relative = np.divide(abs(result.sfr_density-y), y, out=np.zeros_like(y), where=y > 0)
        # 避开时标分支边界，反推旋转率要求的加速度，仅用于检验残差空间形状。
        mask = ((y > 0) & (result.tau_rotation < .8*result.tau_free_fall)
                & (result.tau_cool < .1*result.tau_rotation))
        effective_g = radius*(params.rotation_factor*y/(params.efficiency*f['rho']))**2
        delta_mass = (effective_g-abs(gravity.fields['grav']))*radius**2/params.gravitational_constant
        log = hst[np.argmin(abs(hst[:, 0]-snap.metadata.time))]
        rows.append(dict(snapshot=index, time=snap.metadata.time,
            local_relative_error_quantiles=dict(zip(("p90", "p99", "max"), map(float, np.quantile(relative[y>0], [.9,.99,1])))),
            local_error_gt_1pct_cells=int(np.count_nonzero((y>0)&(relative>.01))),
            local_error_gt_1pct_sfr_fraction=float(np.sum((y*v)[relative>.01])/np.sum(y*v)),
            rotation_true_sfr_fraction=float(np.sum((y*v)[result.limiter_code==3])/np.sum(y*v)),
            inferred_delta_mass_msun_quantiles=dict(zip(("p01", "p50", "p99"), map(float, np.quantile(delta_mass[mask]*2.5e7, [.01,.5,.99])))),
            hst_nearest_time=float(log[0]), hst_time_offset=float(log[0]-snap.metadata.time),
            hst_column20=float(log[1]), hst_column27=float(log[2]), hst_column30=float(log[3]),
            truth_integral_msun_per_yr=float(np.sum(y*v)*params.sfr_to_msun_per_yr)))
    payload = dict(warning="反推引力使用 rhodotsf，仅用于诊断，不能用作模型 baseline；hst 列号为 1-based 且语义未由作者确认。",
                   snapshots=rows)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
