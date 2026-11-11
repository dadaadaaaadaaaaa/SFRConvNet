"""跨折相同帧对照必须按UID对齐，且拒绝不同物理标签。"""
import unittest
import pandas as pd
from VisualScripts.DataAnalysis.compare_cold_dense_folds import matched_role_metrics


class FoldComparisonTests(unittest.TestCase):
    def test_same_frames_are_aligned_before_comparison(self):
        held=pd.DataFrame(dict(uid=['a','b'],parent=['pa','pb'],time=[1.,2.],
                               Y=[1.,3.],B=[0.,1.],P=[0.,1.]))
        known=held.iloc[::-1].copy();known.P=known.Y
        result=matched_role_metrics(held,known)
        self.assertEqual(result['N'],2)
        self.assertEqual(result['held_MAE'],1.5)
        self.assertEqual(result['known_MAE'],0.)
        known.loc[0,'Y']=99.
        with self.assertRaises(ValueError):matched_role_metrics(held,known)
        with self.assertRaises(ValueError):matched_role_metrics(held,known.iloc[:0])


if __name__=='__main__':unittest.main()
