"""真实零标签L1：手算归约、梯度、单位、默认兼容及完整训练/分析契约。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch
from Experiment.cell_sfr_objective import CellSFRLoss
from Experiment.train_sfr_unet import TrainConfig, DataBundle, train, evaluate
from Experiment.SFRUNet import SFRUNet
from Experiment.run_optuna_sfr_unet import parse_search, CATEGORICAL, study_contract
from VisualScripts.ResultAnalysis.analyze_sfr_unet import analyze_result
from tests.test_sfr_unet_training import make_cache


class ZeroL1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def objective(self, weight=0, **kw):
        return CellSFRLoss(closure_scale=1,cell_scale=2,total_scale=3,
                           output_constraint='relu',candidate_weight=1,zero_l1_weight=weight,**kw)

    def test_hand_reduction_and_gradient(self):
        # 首帧两个零标签，次帧一个，第三帧无零；不得改为batch所有零格一起平均。
        y=torch.tensor([[[[0.,0.]]],[[[0.,1.]]],[[[1.,1.]]]],dtype=torch.float64)
        b=torch.ones_like(y)*3
        q=torch.tensor([[[[2.,4.]]],[[[6.,8.]]],[[[10.,12.]]]],dtype=torch.float64)
        z=torch.asinh(q-b).requires_grad_()
        old,detail0=self.objective()(z,b,y)
        new,detail=self.objective(.3)(z,b,y)
        self.assertAlmostEqual(detail['zero_l1_loss'].item(),(1.5+3+0)/3)
        torch.testing.assert_close(new-old,.3*detail['zero_l1_loss'])
        grad=torch.autograd.grad(new-old,z)[0]
        expected=torch.tensor([[[[.025,.025]]],[[[.05,0.]]],[[[0.,0.]]]],dtype=torch.float64)*z.cosh()
        torch.testing.assert_close(grad,expected)
        # 默认关闭时总损失严格保留原有算式。
        err=detail0['sfr_map_physical']-y
        expected_loss=((err/2).square().mean((1,2,3))+(err.sum((1,2,3))/3).square()
                       +torch.zeros(3,dtype=torch.float64)
                       +((detail0['candidate_sfr_map_physical']-y)/2).square().mean((1,2,3))).mean()
        torch.testing.assert_close(old,expected_loss,rtol=0,atol=0)

    def test_exact_zero_final_not_closure_and_units(self):
        y=torch.tensor([[[[0.,1e-20,0.]]]],dtype=torch.float64)
        b=torch.ones_like(y)*2
        z=torch.asinh(torch.tensor([[[[-2.,1.,-3.]]]],dtype=torch.float64)).requires_grad_()
        loss,d=self.objective(1)(z,b,y)
        self.assertLess(d['zero_l1_loss'].abs().item(),1e-15)
        # 极小正标签不能被阈值误判成零；负候选由candidate监督，零L1不额外惩罚。
        grad=torch.autograd.grad(d['zero_l1_loss'],z)[0]
        torch.testing.assert_close(grad,torch.zeros_like(grad))
        z=z.detach()+.2
        loss,_=self.objective(.3)(z,b,y)
        scaled=CellSFRLoss(closure_scale=.025,cell_scale=.05,total_scale=.075,
                          output_constraint='relu',candidate_weight=1,zero_l1_weight=.3)
        other,_=scaled(z,b*.025,y*.025)
        torch.testing.assert_close(loss,other)

    def test_invalid_weights_and_search(self):
        for w in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):self.objective(w)
            with self.assertRaises(ValueError):TrainConfig(data_exp='x',zero_l1_weight=w).validate()
        self.assertEqual(parse_search(['zero_l1_weight']),('zero_l1_weight',))
        self.assertIn(0.,CATEGORICAL['zero_l1_weight'])

    def test_exp2_train_resume_analysis_and_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=replace(make_cache(folder,'exp2'),output_constraint='relu',candidate_weight=1,zero_l1_weight=.1)
            bundle=DataBundle(cfg)
            full=train(cfg,bundle=bundle)
            paused=train(replace(cfg,stop_after_epochs=1),bundle=bundle)
            resume=replace(cfg,resume=str(Path(paused['output_dir'])/'last.pt'))
            resumed=train(resume,bundle=bundle)
            a=torch.load(Path(full['output_dir'])/'last.pt',weights_only=False)
            b=torch.load(Path(resumed['output_dir'])/'last.pt',weights_only=False)
            for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
            self.assertEqual(a['loss_contract']['zero_l1_weight'],.1)
            self.assertEqual(a['loss_contract']['schema_version'],3)
            self.assertIn('train_zero_l1_loss',a['history'][0])
            with self.assertRaisesRegex(ValueError,'恢复配置'):train(replace(resume,zero_l1_weight=0),bundle=bundle)
            result=analyze_result(full['output_dir'],plots=False)
            self.assertIn('ZeroCellFalseSFR_mean',result['metrics'])
            self.assertTrue((result['metrics'].ZeroCellFalseSFR_mean>=0).all())
            model=SFRUNet(len(bundle.datasets['train'].input_channels),base_ch=4)
            scores=[]
            for weight in (0,1):
                obj=CellSFRLoss.from_scale_contract(bundle.scales,output_constraint='relu',candidate_weight=1,zero_l1_weight=weight)
                metrics,_=evaluate(model,bundle.datasets['val'],obj,bundle.scales,device=torch.device('cpu'),batch_size=2)
                scores.append(metrics['model']['macro_joint_score'])
            self.assertEqual(*scores)
            first=study_contract(cfg,bundle,('lr',),'tpe',42,False)
            second=study_contract(replace(cfg,zero_l1_weight=0),bundle,('lr',),'tpe',42,False)
            self.assertNotEqual(first,second)

    def test_hpo_new_parameter(self):
        from Experiment.run_optuna_sfr_unet import optimize
        import json
        with tempfile.TemporaryDirectory() as folder:
            cfg=replace(make_cache(folder),epochs=1,output_constraint='relu',candidate_weight=1)
            result=optimize(cfg,search=['zero_l1_weight'],n_trials=1,timeout=60,pruning=False)[0]
            best=json.loads(Path(result['best_config']).read_text())
            self.assertIn(best['zero_l1_weight'],CATEGORICAL['zero_l1_weight'])

    @unittest.skipUnless(torch.cuda.is_available(),'需要CUDA')
    def test_cuda_active_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=replace(make_cache(folder),epochs=1,device='cuda',output_constraint='relu',candidate_weight=1,zero_l1_weight=.1)
            bundle=DataBundle(cfg)
            for amp in ['none']+(['bfloat16'] if torch.cuda.is_bf16_supported() else []):
                result=train(replace(cfg,amp=amp),bundle=bundle)
                saved=torch.load(Path(result['output_dir'])/'last.pt',weights_only=False)
                self.assertTrue(torch.isfinite(torch.tensor(saved['history'][0]['train_zero_l1_loss'])))


if __name__=='__main__':unittest.main()
