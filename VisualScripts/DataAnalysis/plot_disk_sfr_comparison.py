"""复用已核验的原始 SFR，补读 Model0.25，绘制六组同轴演化图。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyhdf.SD import SD

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_data_processing.galaxy_data import (
    _build_snapshot_schedule,
    _validate_macer_dataset_labels,
    _validate_macer_snapshot_time,
)
from raw_data_processing.parse_log_file import load_config

LABELS = {
    'fiducial': 'Fiducial', 'fiducial_4': 'Fiducial4',
    'fiducial_7': 'Fiducial7', 'model_0_25': 'Model0.25',
    'model_0_75': 'Model0.75', 'model_1_1': 'Model1.1',
}
VERIFIED = PROJECT_ROOT / 'VisualScripts/DataAnalysis/outputs/ssfr_verified_20261023'
DEFAULT_OUTPUT = PROJECT_ROOT / 'VisualScripts/DataAnalysis/outputs/sfr_six_models_20261117'


def read_sfr(path, time):
    """仅读取 Data-Set-11；调用前进入原始根目录，规避中文路径问题。"""
    handle = SD(str(path))
    try:
        _validate_macer_dataset_labels(handle, {'Data-Set-11'})
        _validate_macer_snapshot_time(handle, float(time))
        values = np.asarray(handle.select('Data-Set-11')[:], dtype=np.float64)
    finally:
        handle.end()
    if values.squeeze().shape != (72, 280):
        raise ValueError(f'{path} 的原始网格 shape 异常：{values.shape}')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f'{path} 的原始 SFR 含负值或非有限值。')
    return float(values.sum(dtype=np.float64)) * 0.025


def load_sfr(configs):
    """逐帧核对缓存身份和当前时间映射；首、中、末帧再抽查原始 SFR。"""
    root = Path(configs['BaseConfig']['raw_data_dir']).resolve()
    target = configs['HdfraConfig']['disk_galaxy']['target']
    if (target['dataset'] != 'Data-Set-11' or target['reduction'] != 'sum'
            or float(target['to_msun_per_yr']) != 0.025):
        raise ValueError('当前配置与原始 SFR 归约/单位契约不一致。')
    cached = pd.read_parquet(VERIFIED / 'ssfr_verified.parquet')
    audit = json.loads((VERIFIED / 'stellar_mass_audit.json').read_text(encoding='utf-8'))
    if audit['sfr_definition'] != 'sum(Data-Set-11) * 0.025 Msun/yr':
        raise ValueError('已有核验表的 SFR 定义不匹配。')
    frames, reports = [], []
    previous = Path.cwd()
    try:
        os.chdir(root)
        for nickname, label in LABELS.items():
            galaxy = f'disk_galaxy_{nickname}'
            if nickname == 'model_0_25':
                # 独立分析配置：不把该运行自动加入训练配置。逐帧验证 HDF 时间。
                config = dict(folder_name='DiskGalaxy/Model0.25', mode='filename_index',
                              snapshot_dt=0.0125, range=[0, 1000], offset=0)
            else:
                config = configs['RawDataConfig']['disk_galaxy'][nickname]
            folder = root / config['folder_name']
            paths = sorted((p for p in folder.glob('hdfra.*')
                            if p.is_file() and p.suffix[1:].isdigit()),
                           key=lambda p: int(p.suffix[1:]))
            if not paths:
                raise FileNotFoundError(f'{folder} 没有原始 snapshot。')
            times, indices = _build_snapshot_schedule([p.name for p in paths], config)
            if nickname == 'model_0_25':
                rows = []
                for number, (path, time, index) in enumerate(zip(paths, times, indices), 1):
                    rows.append(dict(galaxy=galaxy, snapshot_file=path.name,
                                     raw_snapshot_index=int(index), time_gyr=float(time),
                                     sfr_msun_per_yr=read_sfr(path.relative_to(root), time),
                                     source_size_bytes=path.stat().st_size,
                                     source_mtime_ns=path.stat().st_mtime_ns))
                    if number % 200 == 0:
                        print(f'{label}: read {number}/{len(paths)}', flush=True)
                frame = pd.DataFrame(rows)
                source = 'raw_hdf4_all_frames'
            else:
                frame = cached.loc[cached.galaxy.eq(galaxy)].copy()
                if frame.snapshot_file.duplicated().any():
                    raise ValueError(f'{label} 核验表含重复 snapshot。')
                if set(frame.snapshot_file) != {p.name for p in paths}:
                    raise ValueError(f'{label} 当前原始文件清单与核验表不同。')
                frame = frame.set_index('snapshot_file').loc[[p.name for p in paths]].reset_index()
                if not np.array_equal(frame.raw_snapshot_index, indices):
                    raise ValueError(f'{label} 原始索引不匹配。')
                if not np.allclose(frame.time_gyr, times, rtol=0, atol=1e-12):
                    raise ValueError(f'{label} 时间映射已改变。')
                for path, row in zip(paths, frame.itertuples()):
                    stat = path.stat()
                    if (stat.st_size != row.source_size_bytes
                            or stat.st_mtime_ns != row.source_mtime_ns):
                        raise ValueError(f'{path} 已改变；拒绝复用旧 SFR。')
                for i in sorted({0, len(paths) // 2, len(paths) - 1}):
                    raw = read_sfr(paths[i].relative_to(root), times[i])
                    if not np.isclose(raw, frame.iloc[i].sfr_msun_per_yr, rtol=1e-12, atol=1e-14):
                        raise ValueError(f'{paths[i]} 原始 SFR 与核验表不一致。')
                source = 'verified_raw_sfr_table_with_file_identity_checks_and_3_raw_spot_checks'
            values = frame.sfr_msun_per_yr.to_numpy()
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'{label} 的 SFR 非法。')
            frame['label'] = label
            frame['snapshot_dt_gyr'] = config['snapshot_dt']
            columns = ['galaxy', 'label', 'snapshot_file', 'raw_snapshot_index', 'time_gyr',
                       'sfr_msun_per_yr', 'source_size_bytes', 'source_mtime_ns', 'snapshot_dt_gyr']
            frames.append(frame[columns])
            reports.append(dict(label=label, source=source, n_snapshots=len(frame),
                                time_min_gyr=float(times.min()), time_max_gyr=float(times.max()),
                                zero_sfr_count=int((values == 0).sum()),
                                sfr_min=float(values.min()), sfr_max=float(values.max())))
            print(f'{label}: validated {len(frame)} snapshots ({source})', flush=True)
    finally:
        os.chdir(previous)
    return pd.concat(frames, ignore_index=True), reports


def plot_comparison(frame):
    """保留零 SFR，并在缺帧或初始化至正式运行的时间跳跃处断线。"""
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': True, 'axes.spines.right': True})
    fig, axis = plt.subplots(figsize=(12, 6), layout='constrained')
    colors = ('#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#ff0000', '#9467bd')
    for label, color in zip(LABELS.values(), colors):
        group = frame.loc[frame.label.eq(label)].sort_values('time_gyr')
        x, y = [], []
        previous = None
        for row in group.itertuples():
            if previous is not None and row.time_gyr - previous > 1.5 * row.snapshot_dt_gyr:
                x.append(np.nan)
                y.append(np.nan)
            x.append(row.time_gyr)
            y.append(row.sfr_msun_per_yr)
            previous = row.time_gyr
        axis.plot(x, y, color=color, lw=0.8, alpha=0.9, label=label,
                  linestyle='-', marker='None')
    axis.set_yscale('symlog', linthresh=1e-4, linscale=0.6)
    axis.set(xlabel='Simulation time [Gyr]', ylabel=r'SFR [$M_\odot\,\mathrm{yr}^{-1}$]',
             title='SFR evolution: six disk galaxies', xlim=(-0.05, 12.6))
    axis.grid(True, which='major', ls='--', alpha=0.25)
    axis.legend(ncol=2, loc='upper right', framealpha=0.8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'拒绝覆盖已有产品，请选择空目录：{output}')
    configs = load_config(str(PROJECT_ROOT / '.config'))
    frame, reports = load_sfr(configs)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / 'sfr_six_models.parquet', index=False)
    frame.to_csv(output / 'sfr_six_models.csv', index=False)
    report = dict(sfr_definition='sum(Data-Set-11) * 0.025 Msun/yr',
                  time_definition='absolute simulation time, Gyr',
                  smoothing='none', missing_snapshots='line breaks; no interpolation', runs=reports)
    (output / 'sfr_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    fig = plot_comparison(frame)
    fig.savefig(output / 'SFR_six_models.png', dpi=220)
    fig.savefig(output / 'SFR_six_models.pdf', dpi=220)
    plt.close(fig)
    print(f'Saved {len(frame)} snapshots and comparison plot to {output}')


if __name__ == '__main__':
    main()
