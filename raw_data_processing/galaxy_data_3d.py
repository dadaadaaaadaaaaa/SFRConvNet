"""三维 processed HDF5 数据契约；由 galaxy_data.GalData 的读入接口分派。

空间轴一律 (r,theta,phi)。监督场与 snapshot 分组分离；三维物理量绝不套用二维 T=93e/rho。
"""

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import tempfile
import h5py
import numpy as np

from raw_data_processing.athdf_reader import read_athdf
from raw_data_processing.coarsening_spatial_3D import derive_fields, validate_state, coarsen_spatial_3d
from raw_data_processing.macer_sfr_operator_3d import SFR3DParameters, calculate_macer_sfr_3d, validate_gravity_geometry
from raw_data_processing.metadata_schema import (build_fine_metadata, build_derived_metadata,
    build_sample_uid, validate_complete_metadata, geometry_fingerprint, canonical_json)


def field_schema(fields):
    """字段角色和单位显式写入文件，避免冗余量、baseline 依赖混入输入。"""
    units = dict(density='code_mass/code_length^3',pressure='code_energy/code_volume',
        gas_energy='code_internal_energy/code_volume',gas_mass='code_mass',temperature='K',
        metallicity='absolute_mass_fraction',cooling_rate_density='code_energy/code_volume/code_time',
        cooling_time='code_time',radial_acceleration='code_length/code_time^2')
    schema = {}
    for name in fields:
        item = dict(units=units.get(name,'code_length/code_time'), role='auxiliary')
        if name in ('density','pressure','vel1','vel2','vel3'):
            item['role'] = 'default_model_input'
        if name.startswith('momentum_'):
            item.update(units='code_mass*code_length/code_time', definition='cell_integrated_M_times_velocity',basis='cartesian',
                        description='单元积分动量，不是动量密度；分组直接求和。')
        elif name.startswith('velocity_'):
            item.update(basis='cartesian',description='固定笛卡尔基底；coarse 总动量除以总质量。')
        elif name in ('vel1','vel2','vel3'):
            item.update(basis='spherical_at_cell_center',description='同一笛卡尔速度投影到单元中心的 r/theta/phi 基底，非直接平均 fine 球坐标分量。')
        elif name in ('cooling_time','cooling_rate_density','radial_acceleration'):
            item['role'] = 'baseline_only'
        if name == 'gas_energy': item['description'] = '气体内能密度 p/(gamma-1)，不是总内能或总能量。'
        if name == 'radial_acceleration': item['description'] = 'coarse 为质量加权的有效径向标量，不代表完整矢量力守恒。'
        schema[name] = item
    return schema


@dataclass
class GalData3D:
    snapshot: dict
    grid_volume: np.ndarray
    axes: tuple
    axis_edges: tuple
    sfr_density_true: np.ndarray
    metadata: dict
    parameters: SFR3DParameters
    mode: str = 'direct'
    diagnostics: dict = None
    ndim: int = 3

    @property
    def time(self): return self.metadata['source']['time']

    @property
    def sfr_target(self): return float(np.sum(self.sfr_density_true*self.grid_volume))

    @property
    def target_metadata(self): return dict(self.metadata['target'])

    @property
    def gas_mass(self): return self.snapshot['gas_mass']

    @property
    def sfr_macer_baseline(self): return self.baseline().total_sfr

    @property
    def sfr_closure_true(self): return self.sfr_target-self.sfr_macer_baseline

    def baseline(self):
        if self.mode != 'residual': raise ValueError('Direct 产品没有 MACER baseline。')
        f = self.snapshot
        return calculate_macer_sfr_3d(density=f['density'],pressure=f['pressure'],metallicity=f['metallicity'],
            cooling_rate=f['cooling_rate_density'],cell_volume=self.grid_volume,radius=self.axes[0][:,None,None],
            radial_acceleration=f['radial_acceleration'],parameters=self.parameters)

    @classmethod
    def from_raw(cls, path, *, source, parameters=None, mode='direct', gravity=None, provenance=None):
        if mode not in ('direct','residual'): raise ValueError('未知 prediction mode。')
        parameters = parameters or SFR3DParameters()
        requested = ['rho','press','metallicity','vel1','vel2','vel3','rhodotsf']
        if mode == 'residual': requested.append('cool_rate')
        s = read_athdf(path,fields=requested,dtype=np.float64)
        f = dict(density=s.fields['rho'],pressure=s.fields['press'],metallicity=s.fields['metallicity'],
                 **{k:s.fields[k] for k in ('vel1','vel2','vel3')})
        if mode == 'residual':
            if gravity is None: raise ValueError('Residual fine 需要静态 gravity reference。')
            validate_gravity_geometry(s,gravity)
            f.update(cooling_rate_density=s.fields['cool_rate'],radial_acceleration=gravity.fields['grav'])
        v, axes, edges = s.geometry.cell_volume, s.geometry.centers, s.geometry.edges
        f = derive_fields(f,v,axes,parameters)
        truth = s.fields['rhodotsf']
        validate_state(f,v,axes,edges,truth,parameters)
        source_path = Path(path)
        source_stat = source_path.stat()
        source = {**source, 'time':s.metadata.time,'time_units':'code_time_approximately_Gyr',
                  'source_file_id':source_path.name}
        # 大型 ATHDF 已由 read_athdf 完整读取；此处不得再隐式通读一次来计算
        # SHA-256。调用方可显式传入强哈希，否则采用文件状态指纹。
        source.setdefault('source_file_size_bytes',int(source_stat.st_size))
        source.setdefault('source_file_mtime_ns',int(source_stat.st_mtime_ns))
        if 'source_file_identity' not in source:
            if source.get('source_file_sha256'):
                source['source_file_identity'] = {
                    'mode':'sha256','sha256':source['source_file_sha256'],
                    'size_bytes':source['source_file_size_bytes'],
                }
            else:
                source['source_file_identity'] = {
                    'mode':'stat','size_bytes':source['source_file_size_bytes'],
                    'mtime_ns':source['source_file_mtime_ns'],
                }
        target = dict(quantity='instantaneous_sfr',units='MACER_mass_code/code_time',
                      field='rhodotsf',reduction='volume_integral',to_msun_per_yr=parameters.sfr_to_msun_per_yr)
        meta = build_fine_metadata(source=source,snapshot_fields=list(f),axes=axes,axis_edges=edges,
            coordinate_mode='spherical_polar',ndim=3,target_metadata=target,volume_source=s.geometry.volume_source)
        meta['geometry'].update(coordinate_axis_order=['r','theta','phi'],spatial_axis_order=['r','theta','phi'],vector_basis='cartesian_and_spherical_at_center')
        meta['physics'] = asdict(parameters)
        meta['provenance'] = provenance or {}
        meta['prediction_mode'] = mode
        meta['fields'] = field_schema(f)
        # 模式、源内容与参数均进入产品身份，避免旧缓存混用。
        op = meta['derivation']['operator_parameters']
        op.update(mode=mode,physics=asdict(parameters),source_identity=source['source_file_identity'],provenance=provenance or {})
        meta['identity']['sample_uid'] = build_sample_uid(meta['identity']['parent_hr_uid'],operator_id='identity_fine',
            operator_version=1,operator_parameters=op,output_geometry_fingerprint=meta['geometry']['geometry_fingerprint'])
        meta['initialization_candidate'] = bool(not np.any(truth) and mode == 'residual' and not np.any(s.fields['cool_rate']))
        return cls(f,v,axes,edges,truth,meta,parameters,mode,{})

    def coarsen(self, operator='spatial_3d', new_shape=(32,16,32), *, prediction_mode=None, **kwargs):
        if operator not in ('spatial_3d','spatial_block','spatial','rh','r_h'):
            raise ValueError('三维目前只支持空间块降采样。')
        if self.metadata['derivation']['operator_id'] != 'identity_fine':
            raise ValueError('必须由 fine 构建 coarse，不允许重复降采样。')
        mode = prediction_mode or self.mode
        if 'conservation_rtol' in kwargs: kwargs['rtol'] = kwargs.pop('conservation_rtol')
        if 'conservation_atol' in kwargs: kwargs['atol'] = kwargs.pop('conservation_atol')
        r = coarsen_spatial_3d(self.snapshot,self.grid_volume,self.axes,self.axis_edges,self.sfr_density_true,
                              self.parameters,new_shape,mode=mode,**kwargs)
        op = dict(operator='spatial_3d',operator_version=1,operator_parameters=dict(
            shape=list(new_shape),mode=mode,source_sample_uid=self.metadata['identity']['sample_uid'],
            cooling='volume_average_power_then_recompute_time',gravity='mass_average_radial_scalar',
            velocity='cartesian_momentum_conservation_then_center_spherical_projection',
            physics=asdict(self.parameters)))
        meta = build_derived_metadata(self.metadata,operator_metadata=op,axes=r.axes,axis_edges=r.edges,
                                     coordinate_mode='spherical_polar',volume_source='athdf_faces:block_volume_sum')
        meta['fields'] = field_schema(r.fields)
        meta['prediction_mode'] = mode
        meta['baseline_scope'] = 'data_compatible_coarse_cooling_not_cooling_table_recomputed'
        return GalData3D(r.fields,r.volume,r.axes,r.edges,r.truth,meta,self.parameters,mode,r.diagnostics)

    def save_h5(self, path):
        validate_state(self.snapshot,self.grid_volume,self.axes,self.axis_edges,self.sfr_density_true,self.parameters)
        validate_complete_metadata(self.metadata)
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            with h5py.File(path,'r') as old:
                if old.attrs.get('macnet_3d_format') != 1 or json.loads(old['metadata/contract_json'][()])['identity']['sample_uid'] != self.metadata['identity']['sample_uid']:
                    raise ValueError(f'输出已存在且产品签名不同，请指定独立目录：{path}')
        # 原子替换：出错时不会留下看似可用的半份训练数据。
        fd, temp = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
        os.close(fd)
        try:
            with h5py.File(temp,'w') as h:
                h.attrs.update(macnet_3d_format=1,time=self.time,prediction_mode=self.mode)
                group = h.create_group('snapshot')
                for name, values in self.snapshot.items():
                    d = group.create_dataset(name,data=values,compression='lzf',shuffle=True)
                    for k,value in self.metadata['fields'][name].items(): d.attrs[k] = value
                geometry = h.create_group('geometry')
                geometry.create_dataset('cell_volume',data=self.grid_volume,compression='lzf')
                geometry.attrs['volume_source'] = self.metadata['geometry']['volume_source']
                coords = h.create_group('coordinates')
                coords.attrs.update(ndim=3,coordinate_mode='spherical_polar',axis_order='r,theta,phi')
                for i,(a,e) in enumerate(zip(self.axes,self.axis_edges)):
                    coords.create_dataset(f'axis_{i}',data=a)
                    coords.create_dataset(f'axis_{i}_edges',data=e)
                h.create_dataset('metadata/contract_json',data=canonical_json(self.metadata))
                h.create_dataset('coarsening/diagnostics_json',data=canonical_json(self.diagnostics or {}))
                sup = h.create_group('sfr_supervision')
                sup.attrs.update(units='MACER_mass_code/code_time',sfr_to_msun_per_yr=self.parameters.sfr_to_msun_per_yr)
                sup.create_dataset('sfr_density_true',data=self.sfr_density_true,compression='lzf')
                sup.create_dataset('canonical/Y_alpha',data=self.sfr_target)
                if self.mode == 'residual':
                    b = self.baseline()
                    sup.create_dataset('sfr_density_baseline',data=b.sfr_density,compression='lzf')
                    sup.create_dataset('sfr_density_closure_true',data=self.sfr_density_true-b.sfr_density,compression='lzf')
                    sup.create_dataset('canonical/B_alpha',data=b.total_sfr)
                    sup.create_dataset('canonical/C_alpha_true',data=self.sfr_target-b.total_sfr)
                    sup.create_dataset('baseline_parameters_json',data=canonical_json(b.parameters))
            os.replace(temp,path)
        finally:
            if os.path.exists(temp): os.unlink(temp)

    @classmethod
    def load_h5(cls,path):
        with h5py.File(path,'r') as h:
            if h.attrs.get('macnet_3d_format') != 1: raise ValueError('不是三维 processed HDF5。')
            meta = json.loads(h['metadata/contract_json'][()])
            validate_complete_metadata(meta)
            obj = cls({k:h['snapshot'][k][()] for k in h['snapshot']},h['geometry/cell_volume'][()],
                tuple(h[f'coordinates/axis_{i}'][()] for i in range(3)),
                tuple(h[f'coordinates/axis_{i}_edges'][()] for i in range(3)),h['sfr_supervision/sfr_density_true'][()],
                meta,SFR3DParameters(**meta['physics']),meta['prediction_mode'],json.loads(h['coarsening/diagnostics_json'][()]))
            if obj.mode not in ('direct','residual') or h.attrs['prediction_mode'] != obj.mode:
                raise ValueError('prediction mode 元数据不一致。')
            if obj.mode == 'residual' and not {'cooling_time','cooling_rate_density','radial_acceleration'}.issubset(obj.snapshot):
                raise ValueError('Residual 文件缺少 baseline 输入。')
            validate_state(obj.snapshot,obj.grid_volume,obj.axes,obj.axis_edges,obj.sfr_density_true,obj.parameters)
            if geometry_fingerprint(obj.axes,obj.axis_edges,coordinate_mode='spherical_polar') != meta['geometry']['geometry_fingerprint']:
                raise ValueError('几何指纹不一致。')
            y = float(h['sfr_supervision/canonical/Y_alpha'][()])
            if not np.isclose(y,obj.sfr_target,rtol=1e-12,atol=1e-12): raise ValueError('SFR 场与总量不一致。')
            if obj.mode == 'residual':
                b = h['sfr_supervision/sfr_density_baseline'][()]
                c = h['sfr_supervision/sfr_density_closure_true'][()]
                bt = float(h['sfr_supervision/canonical/B_alpha'][()])
                ct = float(h['sfr_supervision/canonical/C_alpha_true'][()])
                if not (b.shape == obj.grid_volume.shape and c.shape == obj.grid_volume.shape
                    and np.isfinite(b).all() and np.isfinite(c).all() and np.all(b >= 0)
                    and np.allclose(b+c,obj.sfr_density_true,rtol=1e-10,atol=1e-12)
                    and np.isclose(np.sum(b*obj.grid_volume),bt,rtol=1e-12,atol=1e-12)
                    and np.isclose(bt+ct,y,rtol=1e-12,atol=1e-12)):
                    raise ValueError('Residual Y/B/C 契约不一致。')
        return obj
