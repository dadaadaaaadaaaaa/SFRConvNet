"""核验五组 Zou 盘星系的恒星质量，并绘制 sSFR 演化。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import h5py
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

GALAXIES = ('fiducial', 'fiducial_4', 'fiducial_7', 'model_0_75', 'model_1_1')
LABELS = dict(zip(GALAXIES, ('Fiducial', 'Fiducial4', 'Fiducial7', 'Model0.75', 'Model1.1')))
DEFAULT_OUTPUT = PROJECT_ROOT / 'VisualScripts/DataAnalysis/outputs/ssfr_verified_20261023'
MASS_UNIT_MSUN = 2.5e7
SFR_TO_MSUN_PER_YR = 0.025
MASS_RTOL = 1e-3


@contextmanager
def raw_directory(root):
    """避免 Windows HDF4 库接收中文绝对路径。"""
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def read_field(handle, index, token, diagnostics=None):
    """校验字段标签和数值；质量字段已乘体积，不再次乘体积。"""
    dataset = handle.select(f'Data-Set-{index}')
    label = str(dataset.attributes().get('long_name', ''))
    if token.upper() not in label.upper():
        raise ValueError(f'Data-Set-{index} 标签不匹配：{label!r}')
    field = np.asarray(dataset[:], dtype=np.float64)
    if not np.isfinite(field).all():
        raise ValueError(f'Data-Set-{index} 含非有限值。')
    negative = field < 0
    if index != 12 and negative.any():
        raise ValueError(f'Data-Set-{index} 含负值。')
    total = float(np.sum(field, dtype=np.float64))
    if total < 0:
        raise ValueError(f'Data-Set-{index} 总质量为负值。')
    # MNEWSTAR 存在局部负数值残差；忠实保留原始求和，不裁剪或重分布。
    if diagnostics is not None:
        diagnostics['new_star_negative_cell_count'] = int(negative.sum())
        diagnostics['new_star_negative_mass_code'] = float(-field[negative].sum())
    return total


def fine_background_mass(data_dir, galaxy):
    """复现上一版 notebook 使用的 fine 背景恒星质量口径。"""
    for dirname in ('fine_sfr_v7', 'fine_sfr_v6'):
        folder = data_dir / galaxy / dirname
        source = next(folder.glob('*.h5'), None) if folder.is_dir() else None
        if source is None:
            continue
        with h5py.File(source, 'r') as handle:
            density = np.asarray(handle['snapshot/initial_stellar_density'], dtype=np.float64)
            volume = np.asarray(handle['geometry/cell_volume'], dtype=np.float64)
        if density.shape != volume.shape or not np.isfinite(density).all() or not np.isfinite(volume).all():
            raise ValueError(f'{source} 的密度与体积无效。')
        if np.any(density < 0) or np.any(volume <= 0):
            raise ValueError(f'{source} 的密度或体积非物理。')
        return float(np.sum(density * volume, dtype=np.float64)) * MASS_UNIT_MSUN, str(source)
    raise FileNotFoundError(f'{galaxy} 缺少 fine 背景质量产品。')


def choose_mass_source(cached, raw, rtol=MASS_RTOL):
    """按整个运行的最大总质量相对误差决定是否复用缓存。"""
    if np.any(raw <= 0) or not np.isfinite(raw).all() or not np.isfinite(cached).all():
        raise ValueError('总恒星质量必须为正有限值。')
    error = np.abs(cached - raw) / raw
    return ('fine_plus_log' if error.max() <= rtol else 'raw_hdf4'), error


def audit_all(configs, output_dir):
    """逐帧比较缓存质量与原始质量，同时保存 SFR，供后续零原始 I/O 绘图。"""
    raw_root = Path(configs['BaseConfig']['raw_data_dir']).resolve()
    data_dir = Path(configs['BaseConfig']['data_dir']).resolve()
    frames, reports = [], []
    part_dir = output_dir / '_audit_parts'
    part_dir.mkdir(parents=True, exist_ok=True)
    with raw_directory(raw_root):
        for nickname in GALAXIES:
            galaxy = f'disk_galaxy_{nickname}'
            part_table = part_dir / f'{galaxy}.parquet'
            part_report = part_dir / f'{galaxy}.json'
            # 仅在显式指定的本次输出目录恢复已完成运行，避免异常后重复全量读盘。
            if part_table.exists() and part_report.exists():
                frames.append(pd.read_parquet(part_table))
                reports.append(json.loads(part_report.read_text(encoding='utf-8')))
                print(f'Resume completed run: {galaxy}', flush=True)
                continue
            config = configs['RawDataConfig']['disk_galaxy'][nickname]
            folder = raw_root / config['folder_name']
            paths = sorted(
                (p for p in folder.glob('hdfra.*') if p.is_file() and p.suffix[1:].isdigit()),
                key=lambda p: int(p.suffix[1:]),
            )
            if not paths:
                raise FileNotFoundError(f'{folder} 中没有科学 snapshot。')
            times, indices = _build_snapshot_schedule([p.name for p in paths], config)
            background, fine_source = fine_background_mass(data_dir, galaxy)
            log_path = data_dir / f'{galaxy}.parquet'
            log = pd.read_parquet(log_path, columns=['time', 'NEWstar'])
            if log.empty or not np.isfinite(log.to_numpy()).all() or (log.NEWstar < 0).any():
                raise ValueError(f'{log_path} 日志质量列无效。')
            log = log.sort_values('time', kind='mergesort').drop_duplicates('time', keep='last')
            if times.min() < log.time.min() - 5e-4 or times.max() > log.time.max() + 5e-4:
                raise ValueError(f'{galaxy} 的时间超出日志舍入容差。')
            cached_new = np.interp(times, log.time, log.NEWstar) * MASS_UNIT_MSUN
            rows = []
            for number, (path, time, index) in enumerate(zip(paths, times, indices), 1):
                handle = SD(str(path.relative_to(raw_root)))
                diagnostics = {}
                try:
                    _validate_macer_dataset_labels(handle, {'Data-Set-11'})
                    _validate_macer_snapshot_time(handle, float(time))
                    sfr = read_field(handle, 11, 'SFR') * SFR_TO_MSUN_PER_YR
                    new = read_field(handle, 12, 'MNEWSTAR', diagnostics) * MASS_UNIT_MSUN
                    initial = read_field(handle, 14, 'Mstar_i') * MASS_UNIT_MSUN
                finally:
                    handle.end()
                rows.append({
                    'galaxy': galaxy, 'snapshot_file': path.name, 'raw_snapshot_index': int(index),
                    'time_gyr': float(time), 'sfr_msun_per_yr': sfr,
                    'raw_new_stellar_mass_msun': new, 'raw_initial_stellar_mass_msun': initial,
                    'raw_stellar_mass_msun': initial + new,
                    'source_size_bytes': path.stat().st_size,
                    'source_mtime_ns': path.stat().st_mtime_ns,
                    **diagnostics,
                })
                if number % 500 == 0 or number == len(paths):
                    print(f'{galaxy}: {number}/{len(paths)}', flush=True)
            frame = pd.DataFrame(rows)
            frame['cached_stellar_mass_msun'] = background + cached_new
            frame['cached_new_stellar_mass_msun'] = cached_new
            source, errors = choose_mass_source(
                frame.cached_stellar_mass_msun.to_numpy(), frame.raw_stellar_mass_msun.to_numpy()
            )
            frame['stellar_mass_relative_error'] = errors
            frame['stellar_mass_msun'] = frame[
                'cached_stellar_mass_msun' if source == 'fine_plus_log' else 'raw_stellar_mass_msun'
            ]
            frame['mass_source'] = source
            frame['ssfr_per_yr'] = frame.sfr_msun_per_yr / frame.stellar_mass_msun
            report = {
                'galaxy': galaxy, 'n_snapshots': len(frame), 'mass_source': source,
                'mass_rtol': MASS_RTOL, 'max_total_mass_relative_error': float(errors.max()),
                'median_total_mass_relative_error': float(np.median(errors)),
                'fine_background_mass_msun': background,
                'max_background_relative_error': float(np.max(np.abs(
                    frame.raw_initial_stellar_mass_msun.to_numpy() - background
                ) / frame.raw_initial_stellar_mass_msun.to_numpy())),
                'fine_source': fine_source, 'log_source': str(log_path), 'raw_source': str(folder),
                'time_min_gyr': float(times.min()), 'time_max_gyr': float(times.max()),
                'endpoint_clamped_count': int(((times < log.time.min()) | (times > log.time.max())).sum()),
                'snapshots_with_negative_new_star_cells': int((frame.new_star_negative_cell_count > 0).sum()),
                'max_negative_new_star_fraction': float(np.max(
                    frame.new_star_negative_mass_code.to_numpy() * MASS_UNIT_MSUN
                    / np.maximum(frame.raw_new_stellar_mass_msun.to_numpy(), 1e-30)
                )),
            }
            frame.to_parquet(part_table, index=False)
            part_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            frames.append(frame)
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
    result = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_dir / 'ssfr_verified.parquet', index=False)
    (output_dir / 'stellar_mass_audit.json').write_text(json.dumps({
        'mass_definition': '(sum(Data-Set-14) + sum(Data-Set-12)) * 2.5e7 Msun',
        'cached_mass_definition': 'fine sum(initial_stellar_density*cell_volume) + interpolated log NEWstar',
        'sfr_definition': 'sum(Data-Set-11) * 0.025 Msun/yr',
        'relative_error_threshold': MASS_RTOL, 'runs': reports,
        'limitation': '采用模拟输出的背景质量加新恒星质量；不另行推定质量回收或观测孔径修正。',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def plot_arrays(frame, cadence, production=False):
    """缺失帧和段边界处断线；保留原始零值。"""
    frame = frame.sort_values('time_gyr')
    if production:
        frame = frame.loc[frame.time_gyr >= 0.5625 - 1e-9]
    xs, ys = [], []
    previous = None
    for row in frame.itertuples():
        if previous is not None and row.time_gyr - previous > cadence * 1.5:
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(row.time_gyr - (0.5625 if production else 0))
        ys.append(row.ssfr_per_yr)
        previous = row.time_gyr
    return xs, ys


def plot_evolution(frame, configs, output_dir, production=False):
    """导出分面图和同轴对比图，并返回两张图供 notebook 显示。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True, sharey=True)
    comparison, axis = plt.subplots(figsize=(12, 6))
    for number, nickname in enumerate(GALAXIES):
        group = frame.loc[frame.galaxy == f'disk_galaxy_{nickname}']
        if group.empty:
            raise ValueError(f'绘图数据缺少 {nickname}。')
        cadence = configs['RawDataConfig']['disk_galaxy'][nickname]['snapshot_dt']
        x, y = plot_arrays(group, cadence, production)
        for target in (axes[number], axis):
            target.plot(x, y, color=colors[number], linewidth=0.8, label=LABELS[nickname])
            target.set_yscale('symlog', linthresh=1e-14, linscale=0.6)
            target.grid(True, which='both', linestyle='--', alpha=0.3)
        axes[number].set_title(LABELS[nickname], loc='left')
        axes[number].set_ylabel(r'sSFR [$\mathrm{yr}^{-1}$]')
    xlabel = 'Production time [Gyr]' if production else 'Simulation time [Gyr]'
    axes[-1].set_xlabel(xlabel)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r'sSFR [$\mathrm{yr}^{-1}$]')
    axis.legend(loc='best', ncol=2)
    axis.set_title('sSFR evolution: five disk galaxies')
    fig.suptitle('Specific star formation rate evolution', y=0.998)
    fig.tight_layout()
    comparison.tight_layout()
    for figure, name in ((fig, 'sSFREvolution'), (comparison, 'sSFR_comparison')):
        figure.savefig(output_dir / f'{name}.png', dpi=180, bbox_inches='tight')
        figure.savefig(output_dir / f'{name}.pdf', bbox_inches='tight')
    return fig, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--production-time', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    # 核验结果默认只生成一次；复用由调用方显式读取，避免意外覆盖已有审计。
    if (output / 'ssfr_verified.parquet').exists():
        raise FileExistsError(f'{output} 已有审计；请复用该文件或选择新的 --output-dir。')
    configs = load_config(str(PROJECT_ROOT / '.config'))
    frame = audit_all(configs, output)
    plot_evolution(frame, configs, output, args.production_time)
    print(f'Output: {output}', flush=True)


if __name__ == '__main__':
    main()
