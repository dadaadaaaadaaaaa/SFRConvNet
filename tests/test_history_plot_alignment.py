"""旧CNN时间重启边界和跨版本真实标签对齐验证。"""
import unittest
import numpy as np
import pandas as pd
from VisualScripts.ResultAnalysis.plot_cnn_unet_histories import legacy_times,align_predictions


class HistoryAlignmentTests(unittest.TestCase):
    def test_restart_boundary_and_special_initial_frame(self):
        mapping=dict(index_zero_time=0.,positive_index_offset=.5625,
                     segment_cycle_size=999,extra_time_steps_per_segment=1)
        times,indices=legacy_times(np.array([0,46,999,1000])*.00125,mapping)
        np.testing.assert_allclose(times,[0.,.62,1.81125,1.81375])
        with self.assertRaises(ValueError):legacy_times([.0005],mapping)

    def test_legacy_alignment_checks_truth_and_shared_frames(self):
        cnn=pd.DataFrame(dict(time=[0.,.62,.62125],Y=[0.,2.,3.],P=[1.,2.,4.]))
        unet=pd.DataFrame(dict(time=[.62125,.62],Y=[3.,2.],P=[3.,2.]))
        paired,audit=align_predictions(cnn,unet,legacy=True)
        self.assertEqual(audit['cnn_extra_N'],1)
        self.assertEqual(paired.CNN.tolist(),[2.,4.])
        cnn.loc[1,'Y']=99.
        with self.assertRaises(ValueError):align_predictions(cnn,unet,legacy=True)


if __name__=='__main__':unittest.main()
