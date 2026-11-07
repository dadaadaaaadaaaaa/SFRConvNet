"""异常输入检查：邻帧边界、经验分位、局部匹配和真实cache接口。"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from VisualScripts.ResultAnalysis.inspect_sfr_unet_inputs import select_neighbors,distribution,nearest_patches,inspect_inputs
from tests.test_sfr_unet_training import make_cache
from Experiment.train_sfr_unet import train

class InputInspectionTests(unittest.TestCase):
    def test_neighbors_and_invalid_time(self):
        center,indices=select_neighbors([3.,1.,2.],1.,5,1e-8)
        self.assertEqual(center,1);self.assertEqual(indices.tolist(),[1,2,0])
        with self.assertRaises(ValueError):select_neighbors([1,2],1.5,1,.01)
        with self.assertRaises(ValueError):select_neighbors([1,1],1,1,.01)

    def test_distribution_and_patch(self):
        s=distribution([0,1,2,np.nan],1)
        self.assertEqual(s['p50'],1);self.assertEqual(s['nonfinite_count'],1)
        a=np.zeros((3,2,4,4));a[1]=1;a[2]=np.nan
        self.assertEqual(nearest_patches(a,np.ones((2,4,4)),0,0)[0],(1,0.))

    def test_training_cache_and_wrong_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            config=make_cache(directory,'exp2');result=train(config)
            r=inspect_inputs(result['output_dir'],directory,times=[9.],neighbors=1,
                             channels=['density'],plots=True)
            self.assertEqual(len(r['frames']),2)
            self.assertTrue(list(Path(r['output_dir']).glob('event0/*maps.png')))
            with np.load(Path(r['output_dir'])/'input_windows.npz',allow_pickle=False) as saved:
                self.assertEqual(saved['x_standardized'].shape[0],2)
                self.assertEqual(saved['sample_uid'].tolist(),r['frames'].sample_uid.tolist())
                self.assertEqual(saved['x_standardized'].shape[1],len(saved['channels']))
            self.assertTrue((Path(r['output_dir'])/'artifacts.json').is_file())
            # 修改cache字节应在读取前拒绝，不能静默用其他实验输入。
            path=Path(directory)/f'{config.data_exp}out.pt'
            with path.open('ab') as f:f.write(b'bad')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                inspect_inputs(result['output_dir'],directory,times=[9.],plots=False)

if __name__=='__main__':unittest.main()
