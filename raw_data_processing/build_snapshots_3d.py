"""统一 raw→fine、fine→coarse 命令行入口的三维实现分支。"""

from pathlib import Path
from dataclasses import asdict, replace
import json
import h5py
from raw_data_processing.galaxy_data import GalData
from raw_data_processing.galaxy_data_3d import GalData3D
from raw_data_processing.athdf_reader import read_athdf
from raw_data_processing.metadata_schema import sha256_file
from raw_data_processing.parse_log_file import load_config
from raw_data_processing.validate_macer_sfr_3d import input_parameters


def directory_name(name):
    if not name or name in ('.','..') or Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('输出子目录必须为单层目录名。')
    return name


def select_paths(paths, indices, max_snapshots, *, raw=False):
    if max_snapshots is not None and (isinstance(max_snapshots,bool) or max_snapshots <= 0 or int(max_snapshots) != max_snapshots):
        raise ValueError('max_snapshots 必须为正整数。')
    indices = set(indices)
    if any(isinstance(n,bool) or int(n) != n or n < 0 for n in indices):
        raise ValueError('snapshot index 必须为非负整数。')
    mapping = {}
    for p in paths:
        index = int(p.name.split('.')[-2] if raw else p.stem)
        if index in mapping: raise ValueError(f'重复 snapshot index: {index}')
        mapping[index] = p
    if indices-set(mapping): raise ValueError(f'缺少 snapshot: {indices-set(mapping)}')
    chosen = [(i,mapping[i]) for i in sorted(mapping) if not indices or i in indices]
    return chosen if max_snapshots is None else chosen[:max_snapshots]


def file_identity(path, mode='stat'):
    """返回源文件身份；默认不额外通读大型 ATHDF/HDF5。"""
    if mode not in ('stat','sha256'):
        raise ValueError("source_hash 必须为 'stat' 或 'sha256'。")
    path = Path(path)
    stat = path.stat()
    identity = dict(mode=mode,file_id=path.name,size_bytes=int(stat.st_size))
    if mode == 'sha256':
        identity['sha256'] = sha256_file(path)
    else:
        identity['mtime_ns'] = int(stat.st_mtime_ns)
    return identity


def _read_contract(path):
    """只读取 HDF5 契约，不载入三维数组。"""
    with h5py.File(path,'r') as h:
        if h.attrs.get('macnet_3d_format') != 1:
            raise ValueError('不是 macnet_3d_format=1 产品')
        required = ('metadata/contract_json','snapshot','geometry/cell_volume',
                    'sfr_supervision/sfr_density_true','sfr_supervision/canonical/Y_alpha')
        missing = [name for name in required if name not in h]
        if missing:
            raise ValueError(f'缺少数据集：{missing}')
        metadata = json.loads(h['metadata/contract_json'][()])
        if h.attrs.get('prediction_mode') != metadata.get('prediction_mode'):
            raise ValueError('HDF5 属性与 metadata 的 prediction_mode 不一致')
        return metadata


def _assert_fine_resume_match(target, *, name, configuration, index, raw_identity,
                              mode, parameters, provenance):
    """确认已有 fine 确实对应本次任务；不以同名文件作为充分条件。"""
    try:
        meta = _read_contract(target)
        source = meta['source']
        expected_source = dict(simulation_code='MACER3D',galaxy_type='dwarf_galaxy',galaxy_id=name,run_id=name,
            source_series_id=configuration['folder_name'],raw_snapshot_index=index,
            source_file_id=raw_identity['file_id'],source_file_size_bytes=raw_identity['size_bytes'])
        mismatches = [key for key,value in expected_source.items() if source.get(key) != value]
        if meta.get('prediction_mode') != mode:
            mismatches.append('prediction_mode')
        if meta.get('physics') != asdict(parameters):
            mismatches.append('physics')
        if meta.get('derivation',{}).get('operator_id') != 'identity_fine':
            mismatches.append('derivation.operator_id')
        if meta.get('provenance',{}).get('input_sha256') != provenance['input_sha256']:
            mismatches.append('provenance.input_sha256')
        if raw_identity['mode'] == 'sha256' and source.get('source_file_sha256') != raw_identity['sha256']:
            mismatches.append('source_file_sha256')
        old_raw_identity = source.get('source_file_identity')
        if old_raw_identity is not None and old_raw_identity != raw_identity:
            mismatches.append('source_file_identity')
        if mode == 'residual':
            old = meta.get('provenance',{})
            expected_gravity = provenance['gravity_identity']
            # 兼容修复前仅保存 gravity_sha256 的产品；默认 stat 模式下仍要求
            # 引力文件路径一致，避免把另一份静态势误当成可续跑产品。
            if old.get('gravity_file') != provenance['gravity_file']:
                mismatches.append('provenance.gravity_file')
            if expected_gravity['mode'] == 'sha256' and old.get('gravity_sha256') != expected_gravity['sha256']:
                mismatches.append('provenance.gravity_sha256')
            old_gravity_identity = old.get('gravity_identity')
            if old_gravity_identity is not None and old_gravity_identity != expected_gravity:
                mismatches.append('provenance.gravity_identity')
        if mismatches:
            raise ValueError(f'契约字段不匹配：{sorted(set(mismatches))}')
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f'已有输出不能安全续跑：{target}；{exc}') from exc


def build_fine_3d(configs,name,configuration,indices,max_snapshots,fine_dirname,mode,
                  source_hash='stat',resume=True):
    raw_root = Path(configs['BaseConfig']['raw_data_dir'])
    folder = raw_root/configuration['folder_name']
    if not folder.exists(): raise FileNotFoundError(folder)
    selected = select_paths(folder.glob('*.athdf'),indices,max_snapshots,raw=True)
    if not selected: raise ValueError(f'没有 ATHDF snapshot: {folder}')
    input_path = raw_root/configuration['input_file']
    parameters = input_parameters(input_path)
    baseline = configs['HdfraConfig']['dwarf_galaxy']['baseline']
    parameters = replace(parameters,rotation_factor=float(baseline.get('rotation_factor',parameters.rotation_factor)))
    provenance = dict(input_sha256=sha256_file(input_path),input_file=str(input_path.resolve()),
                      config_source='HdfraConfig.dwarf_galaxy',metallicity_semantics='absolute_mass_fraction_data_supported')
    gravity = None
    gravity_path = None
    if mode == 'residual':
        root = Path(configs['BaseConfig'].get('workspace_root',Path.cwd()))
        gravity_path = root/baseline['gravity_reference_file']
        gravity_identity = file_identity(gravity_path,source_hash)
        provenance.update(gravity_file=str(gravity_path.resolve()),gravity_identity=gravity_identity)
        if source_hash == 'sha256':
            provenance['gravity_sha256'] = gravity_identity['sha256']
    output = Path(configs['BaseConfig']['data_dir'])/name/directory_name(fine_dirname)
    written = []
    for index,path in selected:
        raw_identity = file_identity(path,source_hash)
        target = output/f'{index:05d}.h5'
        if target.exists() and resume:
            _assert_fine_resume_match(target,name=name,configuration=configuration,index=index,
                raw_identity=raw_identity,mode=mode,parameters=parameters,provenance=provenance)
            print(f'[跳过] [{name}] raw→fine {mode}: {target.name}（已有产品契约匹配）',flush=True)
            continue
        if mode == 'residual' and gravity is None:
            gravity = read_athdf(gravity_path,fields=('grav',),dtype='float64')
        source = dict(simulation_code='MACER3D',galaxy_type='dwarf_galaxy',
            galaxy_id=name,run_id=name,source_series_id=configuration['folder_name'],raw_snapshot_index=index,
            source_file_id=raw_identity['file_id'],source_file_size_bytes=raw_identity['size_bytes'],
            source_file_identity=raw_identity)
        if source_hash == 'sha256':
            source['source_file_sha256'] = raw_identity['sha256']
        else:
            source['source_file_mtime_ns'] = raw_identity['mtime_ns']
        fine = GalData().read_athdf(path,source=source,
            parameters=parameters,mode=mode,gravity=gravity,provenance=provenance)
        fine.save_h5(target)
        print(f'[{name}] raw→fine {mode}: {target.name}',flush=True)
        written.append(target)
    return written


def build_coarse_3d(*,config_dir,galaxies,shape,mode,fine_dirname,coarse_dirname,indices,max_snapshots):
    if mode not in ('direct','residual'): raise ValueError('未知 prediction mode。')
    if len(shape) != 3 or any(isinstance(n,bool) or int(n) != n or n <= 0 for n in shape):
        raise ValueError('三维 coarse shape 必须为三个正整数。')
    shape = tuple(int(n) for n in shape)
    directory_name(fine_dirname)
    out_name = directory_name(coarse_dirname or f'coarse_spatial3d_{"x".join(map(str,shape))}_{mode}')
    if out_name == fine_dirname: raise ValueError('coarse 输出不能覆盖 fine 目录。')
    configs = load_config(str(config_dir))
    names = [f'dwarf_galaxy_{n}' for n in configs['RawDataConfig'].get('dwarf_galaxy',{})]
    selected = set(galaxies)
    if selected-set(names): raise ValueError(f'spatial_3d 仅支持已配置的矮星系：{selected-set(names)}')
    data_dir = Path(configs['BaseConfig']['data_dir'])
    written = []
    for name in names:
        if selected and name not in selected: continue
        folder = data_dir/name/fine_dirname
        if not folder.exists(): raise FileNotFoundError(folder)
        paths = select_paths(folder.glob('*.h5'),indices,max_snapshots)
        if not paths: raise ValueError(f'没有 fine HDF5: {folder}')
        for index,path in paths:
            fine = GalData().load_h5(path)
            if not isinstance(fine,GalData3D): raise ValueError(f'非三维 fine: {path}')
            coarse = fine.coarsen('spatial_3d',shape,prediction_mode=mode)
            # fine 已由 GalData 完整读取，不再为 provenance 额外通读一次。
            coarse.metadata['provenance']['fine_file_identity'] = file_identity(path,'stat')
            target = data_dir/name/out_name/f'{index:05d}.h5'
            coarse.save_h5(target)
            print(f'[{name}] fine→coarse {mode} {shape}: {target.name}',flush=True)
            written.append(target)
    return written
