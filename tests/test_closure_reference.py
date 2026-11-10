"""闭合统计的负值语义、输入匹配隔离与缺失参考回归检查。"""
import unittest
import numpy as np
import pandas as pd
from VisualScripts.ResultAnalysis.compare_closure_reference import (
    describe, relation, nearest_reference, validate_manifest, block_summary,
)


class ClosureReferenceTests(unittest.TestCase):
    def test_signed_distribution_preserves_extreme(self):
        result=describe([-3,0,1,1000])
        self.assertEqual(result['min'],-3)
        self.assertEqual(result['max'],1000)
        self.assertEqual(result['negative_fraction'],.25)
        with self.assertRaises(ValueError):
            describe([0,np.nan])

    def test_error_decomposition_and_reference_pull(self):
        a=pd.DataFrame(dict(C=[4.,8.],Chat=[2.,4.],Cref=[2.,4.]))
        r=relation(a,.1)
        self.assertEqual(r['Alpha_median'],0)
        self.assertEqual(r['Closer_to_reference'],1)
        self.assertEqual(r['Through_origin_slope'],0)
        a.Chat=a.C
        r=relation(a,.1)
        self.assertEqual(r['Alpha_median'],1)
        self.assertEqual(r['MAE'],0)

    def test_zero_gap_and_zero_total_not_ratio(self):
        r=relation(pd.DataFrame(dict(C=[-1.,1.],Chat=[0.,0.],Cref=[-1.,1.])),.1)
        self.assertTrue(np.isnan(r['Recovery']))
        self.assertTrue(np.isnan(r['Alpha_median']))
        self.assertEqual(r['ratio_N'],0)

    def test_missing_reference_does_not_drop_prediction_error(self):
        r=relation(pd.DataFrame(dict(C=[5.,5.],Chat=[1.,3.],Cref=[np.nan,4.])),.1)
        self.assertEqual(r['N'],2)
        self.assertEqual(r['matched_N'],1)
        self.assertEqual(r['MAE'],3)
        self.assertEqual(r['Chat_matched_mean'],3)
        self.assertEqual(r['Bias_matched'],-2)

    def test_matching_uses_inputs_and_baseline_not_truth(self):
        train=pd.DataFrame(dict(uid=['a','b'],group=['known','known'],time=[1.,2.],B=[0.,10.],C=[100.,1.]))
        x=np.array([[[[0.]]],[[[1.]]]],dtype=np.float32)
        query=pd.DataFrame(dict(uid=['q'],group=['held'],time=[1.],B=[10.],C=[100000.]))
        matched,_=nearest_reference(train,x,query,x[:1],methods={'x':None,'b10':(.1,.25)})
        self.assertEqual(matched.loc[matched.method=='x','ref_uid'].iloc[0],'a')
        self.assertEqual(matched.loc[matched.method=='b10','ref_uid'].iloc[0],'b')
        query.C=-500
        again,_=nearest_reference(train,x,query,x[:1],methods={'x':None,'b10':(.1,.25)})
        pd.testing.assert_frame_equal(matched,again)

    def test_matching_excludes_near_time_and_has_no_fake_reference(self):
        train=pd.DataFrame(dict(uid=['a','b'],group=['g','g'],time=[1.,2.],B=[0.,0.],C=[1.,2.]))
        x=np.array([[[[0.]]],[[[1.]]]],dtype=np.float32)
        query=pd.DataFrame(dict(uid=['q'],group=['g'],time=[1.1],B=[10.]))
        m,_=nearest_reference(train,x,query,x[:1],methods={'x':None,'b10':(.1,.25)})
        self.assertEqual(m.loc[m.method=='x','ref_uid'].iloc[0],'b')
        bad=m[m.method=='b10'].iloc[0]
        self.assertEqual(bad.ref_uid,'')
        self.assertTrue(np.isnan(bad.Cref))
        self.assertEqual(bad.k_valid,0)

    def test_parent_overlap_rejected(self):
        f={'train':pd.DataFrame(dict(uid=['a'],parent=['p'])), 'out':pd.DataFrame(dict(uid=['b'],parent=['p']))}
        manifest={s:dict(sample_uid=list(a.uid),parent_hr_uid=list(a.parent)) for s,a in f.items()}
        with self.assertRaises(ValueError):
            validate_manifest(manifest,f)

    def test_block_bootstrap_keeps_error_identity(self):
        f=pd.DataFrame(dict(C=np.arange(10.)+2,Chat=np.arange(10.)+1,Cref=np.arange(10.),
                            group=['g']*10,time=np.arange(10.)))
        r=block_summary(f,.5,100)
        self.assertAlmostEqual(r['Bias_low'],-1)
        self.assertAlmostEqual(r['Bias_high'],-1)
        self.assertAlmostEqual(r['slope_low'],.5)


if __name__=='__main__':
    unittest.main()
