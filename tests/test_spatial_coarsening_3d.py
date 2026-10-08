"""三维守恒降采样、坐标基底及 HDF5 标签隔离的测试。"""

from dataclasses import asdict
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import h5py

from raw_data_processing.coarsening_spatial_3D import (
    spherical_volume, cartesian_to_spherical, derive_fields, coarsen_spatial_3d,
)
from raw_data_processing.galaxy_data_3d import GalData3D
from raw_data_processing.galaxy_data import GalData
from raw_data_processing.macer_sfr_operator_3d import SFR3DParameters
from raw_data_processing.build_coarse_from_fine import build_coarse_from_fine
from raw_data_processing.build_snapshots_3d import file_identity


class Spatial3DTests(unittest.TestCase):
    def test_stat_identity_does_not_hash_large_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source.athdf'
            source.write_bytes(b'synthetic')
            with patch('raw_data_processing.build_snapshots_3d.sha256_file') as digest:
                identity = file_identity(source)
            digest.assert_not_called()
            self.assertEqual(identity['mode'],'stat')
            self.assertEqual(identity['size_bytes'],9)
            self.assertIn('mtime_ns',identity)

    def test_sha256_identity_is_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source.athdf'
            source.write_bytes(b'synthetic')
            identity = file_identity(source,'sha256')
            self.assertEqual(identity['mode'],'sha256')
            self.assertEqual(len(identity['sha256']),64)

    def state(self, residual=True):
        edges = (np.geomspace(.1,10,9),np.linspace(.1,3.,5),np.linspace(0,2*np.pi,9))
        axes = tuple((e[:-1]+e[1:])/2 for e in edges)
        v = spherical_volume(edges)
        rng = np.random.default_rng(101)
        rho = rng.uniform(.5,10,v.shape)
        xyz = tuple(rng.normal(size=v.shape) for _ in range(3))
        sph = cartesian_to_spherical(*xyz,axes)
        f = dict(density=rho,pressure=rho*200,metallicity=rng.uniform(.001,.1,v.shape),
                 **dict(zip(('vel1','vel2','vel3'),sph)))
        if residual: f.update(cooling_rate_density=np.full_like(v,1e6),radial_acceleration=-np.ones_like(v))
        p = SFR3DParameters()
        return derive_fields(f,v,axes,p),v,axes,edges,rng.uniform(0,1,v.shape),p

    def test_conservation_and_target_volume_average(self):
        f,v,a,e,y,p = self.state()
        r = coarsen_spatial_3d(f,v,a,e,y,p,(4,2,4),mode='residual')
        self.assertTrue(r.diagnostics['conservation_passed'])
        for name, d in r.diagnostics.items():
            if isinstance(d,dict): self.assertLess(d['scaled_relative_error'],1e-12,name)
        # 显式遍历一个块作为独立几何/目标参考。
        block = np.s_[:2,:2,:2]
        self.assertAlmostEqual(r.truth[0,0,0],np.sum(y[block]*v[block])/np.sum(v[block]))
        self.assertAlmostEqual(r.fields['metallicity'][0,0,0],np.sum(f['metallicity'][block]*f['gas_mass'][block])/np.sum(f['gas_mass'][block]))
        self.assertGreaterEqual(r.diagnostics['unresolved_kinetic_energy'],-1e-10)

    def test_constant_cartesian_flow_stays_constant(self):
        f,v,a,e,y,p = self.state(False)
        xyz = tuple(np.full_like(v,x) for x in (2.,-3.,4.))
        f.update(dict(zip(('vel1','vel2','vel3'),cartesian_to_spherical(*xyz,a))))
        f = derive_fields(f,v,a,p)
        r = coarsen_spatial_3d(f,v,a,e,y,p,(2,1,2))
        for name,value in zip('xyz',(2.,-3.,4.)):
            np.testing.assert_allclose(r.fields['velocity_'+name],value,atol=1e-13)
        self.assertAlmostEqual(r.diagnostics['unresolved_kinetic_energy'],0,delta=1e-8)

    def test_identity_resolution_preserves_cartesian_state(self):
        f,v,a,e,y,p = self.state()
        r = coarsen_spatial_3d(f,v,a,e,y,p,v.shape,mode='residual')
        for name in ('density','pressure','metallicity','velocity_x','momentum_y'):
            np.testing.assert_allclose(r.fields[name],f[name],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(r.truth,y)

    def test_rejects_bad_shape_geometry_and_inconsistent_momentum(self):
        for shape in ((3,2,4),(4,2),(4.5,2,4),(0,2,4)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError): coarsen_spatial_3d(*self.state(),new_shape=shape)
        f,v,a,e,y,p = self.state()
        with self.assertRaises(ValueError): coarsen_spatial_3d(f,v*2,a,e,y,p,(4,2,4))
        f['momentum_x'] = f['momentum_x']+1
        with self.assertRaises(ValueError): coarsen_spatial_3d(f,v,a,e,y,p,(4,2,4))

    def test_direct_omits_baseline_and_residual_requires_fields(self):
        f,v,a,e,y,p = self.state(False)
        r = coarsen_spatial_3d(f,v,a,e,y,p,(4,2,4))
        self.assertNotIn('cooling_time',r.fields)
        self.assertNotIn('sfr_density_true',r.fields)
        with self.assertRaises(ValueError): coarsen_spatial_3d(f,v,a,e,y,p,(4,2,4),mode='residual')

    def fine_object(self, path, mode='residual'):
        f,v,a,e,y,p = self.state(mode=='residual')
        fields = dict(rho=f['density'],press=f['pressure'],metallicity=f['metallicity'],rhodotsf=y,
                      **{k:f[k] for k in ('vel1','vel2','vel3')})
        if mode=='residual': fields['cool_rate']=f['cooling_rate_density']
        geometry = SimpleNamespace(cell_volume=v,centers=a,edges=e,volume_source='athdf_cell_faces:spherical_exact')
        raw = SimpleNamespace(fields=fields,geometry=geometry,metadata=SimpleNamespace(time=2.5))
        gravity = SimpleNamespace(fields={'grav':-np.ones_like(v)},geometry=geometry)
        source = dict(simulation_code='MACER3D',galaxy_id='dwarf_galaxy_test',run_id='test',source_series_id='test',raw_snapshot_index=51)
        with patch('raw_data_processing.galaxy_data_3d.read_athdf',return_value=raw):
            obj=GalData3D.from_raw(path,source=source,mode=mode,gravity=gravity,parameters=p)
        return obj

    def test_hdf5_roundtrip_and_no_label_in_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'raw';source.write_bytes(b'synthetic')
            fine=self.fine_object(source)
            file=Path(folder)/'fine.h5'; fine.save_h5(file)
            loaded=GalData().load_h5(file)
            coarse=loaded.coarsen(new_shape=(4,2,4),prediction_mode='residual')
            output=Path(folder)/'coarse.h5';coarse.save_h5(output)
            result=GalData().load_h5(output)
            self.assertAlmostEqual(result.sfr_target,fine.sfr_target)
            with h5py.File(output,'r') as h:
                self.assertNotIn('sfr_density_true',h['snapshot'])
                self.assertIn('笛卡尔',h['snapshot/velocity_x'].attrs['description'])
                yb=h['sfr_supervision/canonical']
                self.assertAlmostEqual(yb['Y_alpha'][()],yb['B_alpha'][()]+yb['C_alpha_true'][()])
            with self.assertRaises(ValueError): result.coarsen(new_shape=(2,1,2))
            with h5py.File(output,'r+') as h: h['sfr_supervision/canonical/Y_alpha'][...]=12345.
            with self.assertRaises(ValueError): GalData().load_h5(output)

    def test_baseline_independent_of_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'raw';source.write_bytes(b'synthetic')
            fine=self.fine_object(source)
            first=fine.coarsen(new_shape=(4,2,4))
            fine.sfr_density_true=fine.sfr_density_true*5
            second=fine.coarsen(new_shape=(4,2,4))
            np.testing.assert_array_equal(first.baseline().sfr_density,second.baseline().sfr_density)
            self.assertAlmostEqual(second.sfr_target,first.sfr_target*5)

    def test_zero_cooling_roundtrip_and_mode_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'raw';source.write_bytes(b'synthetic')
            fine=self.fine_object(source)
            fine.snapshot['cooling_rate_density'][:]=0
            fine.snapshot=derive_fields(fine.snapshot,fine.grid_volume,fine.axes,fine.parameters)
            file=Path(folder)/'out.h5';fine.save_h5(file)
            result=GalData3D.load_h5(file)
            self.assertEqual(result.baseline().total_sfr,0)
            self.assertTrue(np.isposinf(result.snapshot['cooling_time']).all())
            direct=self.fine_object(source,mode='direct')
            with self.assertRaises(ValueError): direct.save_h5(file)

    def test_coarse_builder_dispatch_and_reject_fine_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'raw'; source.write_bytes(b'synthetic')
            cfg=root/'config';cfg.mkdir()
            (cfg/'BaseConfig.json').write_text(json.dumps({'data_dir':str(root/'data')}))
            (cfg/'RawDataConfig.json').write_text(json.dumps({'dwarf_galaxy':{'test':{}}}))
            fine=self.fine_object(source)
            fine.save_h5(root/'data/dwarf_galaxy_test/fine/00051.h5')
            kwargs=dict(config_dir=cfg,galaxies=['dwarf_galaxy_test'],coarsener='spatial_3d',
                        coarse_shape=(4,2,4),snapshot_indices=[51],prediction_mode='residual')
            paths=build_coarse_from_fine(**kwargs)
            self.assertEqual(len(paths),1)
            self.assertEqual(GalData3D.load_h5(paths[0]).grid_volume.shape,(4,2,4))
            with self.assertRaises(ValueError): build_coarse_from_fine(**kwargs,coarse_dirname='fine')


if __name__ == '__main__': unittest.main()
