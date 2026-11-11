"""冷致密质量比例与非线性密度矩的边界及代数验证。"""
import unittest
import numpy as np
from VisualScripts.DataAnalysis.analyze_cold_dense_closure import moments,shapley,K


class ColdDenseTests(unittest.TestCase):
    def test_mass_fraction_not_cell_fraction(self):
        rho=np.array([[1.,9.]])
        e=rho*np.array([[1e5,1e4]])/93
        d=moments(rho,e,np.ones_like(rho),np.ones_like(rho,dtype=bool))
        self.assertAlmostEqual(d['f'],.9)
        self.assertAlmostEqual(d['cell_fraction'],.5)
        self.assertAlmostEqual(d['q'],3)
        self.assertAlmostEqual(d['Sff'],27*K)

    def test_density_tail_matters_at_equal_mass_and_fraction(self):
        def calc(r):
            return moments(r,r*1e4/93,np.ones_like(r),np.ones_like(r,dtype=bool))
        a=calc(np.array([[2.,2.]]));b=calc(np.array([[1.,3.]]))
        self.assertEqual(a['M'],b['M']);self.assertEqual(a['f'],b['f'])
        self.assertGreater(b['Sff'],a['Sff'])

    def test_empty_eligible_has_zero_sfr(self):
        r=np.ones((2,2));d=moments(r,r*4e4/93,r,r.astype(bool))
        self.assertEqual(d['f'],0);self.assertEqual(d['Sff'],0)

    def test_symmetric_product_decomposition(self):
        a=[4.,.4,3.];b=[2.,.2,2.]
        parts=shapley(a,b)
        self.assertAlmostEqual(sum(parts),K*(np.prod(a)-np.prod(b)))
        np.testing.assert_allclose(parts,-np.asarray(shapley(b,a)))
        a=[2.,.4,2.]
        self.assertEqual(shapley(a,b)[0],0);self.assertEqual(shapley(a,b)[2],0)


if __name__=='__main__':unittest.main()
