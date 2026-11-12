"""逐格零值分类、幅度收支和正标签分箱的物理语义验证。"""
import unittest
import numpy as np
from VisualScripts.ResultAnalysis.analyze_unet_all_cells import classification,budget,bins


class AllCellTests(unittest.TestCase):
    def test_conditional_rates_and_error_budget(self):
        y=np.array([0.,0.,1.,2.]);p=np.array([.5,0.,0.,1.])
        row=classification(y,p,0.)
        self.assertEqual([row[k] for k in ['TN','FP','FN','TP']],[1,1,1,1])
        self.assertEqual(row['false_positive_rate'],.5)
        self.assertEqual(row['false_negative_rate'],.5)
        np.testing.assert_allclose(budget(y,p),[100*.5/3,0.,-200/3])
        self.assertAlmostEqual(budget(y,p).sum(),-50.)

    def test_epsilon_false_positive_is_excess_over_nonzero_truth(self):
        row=classification(np.array([.05,.2]),np.array([.2,.05]),.1)
        self.assertEqual(row['FP'],1);self.assertEqual(row['FN'],1)
        self.assertAlmostEqual(row['fp_excess_pct_true_sum'],60.)
        self.assertAlmostEqual(row['fn_deficit_pct_true_sum'],60.)

    def test_positive_bins_preserve_last_edge_and_exclude_exact_zero(self):
        result=bins(np.array([0.,.5,1.,2.]),np.array([99.,1.,2.,4.]),np.array([0.,1.,2.]))
        self.assertEqual(result.N.sum(),3)
        self.assertAlmostEqual(result.true_sum.sum(),3.5)
        np.testing.assert_allclose(result.ratio,2.)


if __name__=='__main__':unittest.main()
