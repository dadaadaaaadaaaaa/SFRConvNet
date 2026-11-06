"""非负逐格 SFR 的数值/梯度、训练恢复、HPO 及新旧分析契约回归。"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from Experiment.cell_sfr_objective import CellSFRLoss, compose_cell_sfr
from Experiment.train_sfr_unet import DataBundle, train, TrainConfig
from Experiment.run_optuna_sfr_unet import optimize
from VisualScripts.ResultAnalysis.analyze_sfr_unet import analyze_result, load_predictions
from tests.test_sfr_unet_training import make_cache


class NonnegativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_relu_zero_and_auxiliary_gradient(self):
        # 正标签而候选值为负：主分支梯度为零，辅助项提供正确方向的梯度。
        b = torch.tensor([[[[.2]]]], dtype=torch.float64)
        y = torch.full_like(b, .5)
        z = torch.asinh(torch.full_like(b, -.5)).requires_grad_()
        composed = compose_cell_sfr(z, b, closure_scale=1, output_constraint="relu")
        main = (composed['sfr_map_physical']-y).square().mean()
        self.assertEqual(torch.autograd.grad(main, z)[0].item(), 0)
        objective = CellSFRLoss(closure_scale=1, cell_scale=1, total_scale=1,
                               output_constraint="relu", candidate_weight=.3)
        loss, detail = objective(z, b, y)
        self.assertEqual(detail['sfr_map_physical'].item(), 0)
        self.assertAlmostEqual(detail['closure_map_physical'].item(), -.2)
        self.assertAlmostEqual(detail['candidate_loss'].item(), .64)
        self.assertAlmostEqual(loss.item(), .25+.25+.3*.64)
        gradient = torch.autograd.grad(loss, z)[0]
        torch.testing.assert_close(gradient, 2*.3*(-.8)*torch.cosh(z))
        self.assertLess(gradient.item(), 0)
        # 零真值/零候选精确输出零，没有 Softplus 背景。
        zero = torch.zeros_like(z)
        loss, _ = objective(zero, zero, zero)
        self.assertEqual(loss.item(), 0)

    def test_positive_negative_closure_and_units(self):
        y = torch.tensor([[[[0., .5, 2.]]]], dtype=torch.float64)
        b = torch.tensor([[[[1., 2., .1]]]], dtype=torch.float64)
        z = torch.asinh((y-b)/2)
        objective = CellSFRLoss(closure_scale=2, cell_scale=3, total_scale=5,
                               output_constraint="relu", candidate_weight=1)
        loss, detail = objective(z, b, y)
        self.assertLess(loss.item(), 1e-28)
        torch.testing.assert_close(detail['closure_map_physical']+b, detail['sfr_map_physical'])
        self.assertTrue((detail['closure_map_physical'] >= -b).all())
        self.assertLess(detail['closure_map_physical'][0,0,0,1].item(), 0)
        z = z + .2
        first, _ = objective(z,b,y)
        other = CellSFRLoss(closure_scale=2*.025, cell_scale=3*.025, total_scale=5*.025,
                           output_constraint="relu", candidate_weight=1)
        second, _ = other(z,b*.025,y*.025)
        torch.testing.assert_close(first,second)

    def test_invalid_modes_and_hidden_overflow(self):
        for mode, weight in [('relu',0), ('none',1), ('unknown',1), ('relu',float('nan'))]:
            with self.assertRaises(ValueError):
                TrainConfig(data_exp='test',output_constraint=mode,candidate_weight=weight).validate()
            with self.assertRaises(ValueError):
                CellSFRLoss(closure_scale=1,cell_scale=1,total_scale=1,
                            output_constraint=mode,candidate_weight=weight)
        objective=CellSFRLoss(closure_scale=1,cell_scale=1,total_scale=1,
                              output_constraint='relu',candidate_weight=1)
        zeros=torch.zeros(1,1,2,2)
        # 即使最终 ReLU 能隐藏负无穷，也必须拒绝非法候选和损失。
        with self.assertRaises(FloatingPointError):
            objective(zeros-1000,zeros,zeros)

    def test_exp2_train_resume_export_and_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            config=replace(make_cache(directory,'exp2'),output_constraint='relu',candidate_weight=.3)
            bundle=DataBundle(config)
            full=train(config,bundle=bundle)
            paused=train(replace(config,stop_after_epochs=1),bundle=bundle)
            resume=replace(config,resume=str(Path(paused['output_dir'])/'last.pt'))
            resumed=train(resume,bundle=bundle)
            a=torch.load(Path(full['output_dir'])/'last.pt',weights_only=False)
            b=torch.load(Path(resumed['output_dir'])/'last.pt',weights_only=False)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'恢复配置'):
                train(replace(resume,output_constraint='none',candidate_weight=0),bundle=bundle)
            root=Path(full['output_dir'])
            result=analyze_result(root,plots=False)
            self.assertTrue((result['metrics'].NegativePredFrac==0).all())
            payload=torch.load(root/'out_predictions.pt',weights_only=False)
            self.assertEqual(payload['schema'],'macnet_sfr_unet_predictions_v2')
            torch.testing.assert_close(payload['prediction_map'],torch.relu(payload['candidate_map']))
            torch.testing.assert_close(payload['closure_map']+payload['baseline_map'],payload['prediction_map'])
            self.assertEqual(a['loss_contract']['negative_weight'],0)
            self.assertEqual(len(a['history'][0]['train_loss']),5)
            # 故意篡改候选值，分析必须检测，而不重新裁剪/覆盖结果。
            payload['candidate_map'] += 1
            torch.save(payload,root/'bad.pt')
            identity=json.loads((root/'split_manifest.json').read_text())['out']
            with self.assertRaisesRegex(ValueError,'候选'):
                load_predictions(root/'bad.pt',identity=identity,expected_groups=['run_c'])

    def test_hpo_candidate_search_and_fixed_score(self):
        with tempfile.TemporaryDirectory() as directory:
            config=replace(make_cache(directory),epochs=1,output_constraint='relu',candidate_weight=1)
            with self.assertRaisesRegex(ValueError,'停用'):
                optimize(config,search=['negative_weight'],n_trials=1,timeout=60)
            with self.assertRaisesRegex(ValueError,'仅用于'):
                optimize(replace(config,output_constraint='none',candidate_weight=0),
                         search=['candidate_weight'],n_trials=1,timeout=60)
            result=optimize(config,search=['candidate_weight'],n_trials=1,timeout=60,pruning=False)[0]
            best=json.loads(Path(result['best_config']).read_text())
            self.assertEqual(best['output_constraint'],'relu')
            self.assertIn(best['candidate_weight'],[.1,.3,1.])
            # 权重变化不能把训练辅助项直接加进选模指标。
            from Experiment.train_sfr_unet import evaluate
            from Experiment.SFRUNet import SFRUNet
            bundle=DataBundle(config)
            model=SFRUNet(len(bundle.datasets['train'].input_channels),base_ch=4)
            scores=[]
            for weight in (.1,1.):
                objective=CellSFRLoss.from_scale_contract(bundle.scales,output_constraint='relu',candidate_weight=weight)
                metrics,_=evaluate(model,bundle.datasets['val'],objective,bundle.scales,
                                   device=torch.device('cpu'),batch_size=2)
                scores.append(metrics['model']['macro_joint_score'])
            self.assertEqual(*scores)

    @unittest.skipUnless(torch.cuda.is_available(), '需要 CUDA 设备')
    def test_cuda_nonnegative_training(self):
        # 验证混合精度仅作用于网络，候选组成与标签仍保留物理精度。
        with tempfile.TemporaryDirectory() as directory:
            config=replace(make_cache(directory),epochs=1,device='cuda',
                           output_constraint='relu',candidate_weight=1)
            bundle=DataBundle(config)
            modes=['none']+(['bfloat16'] if torch.cuda.is_bf16_supported() else [])
            for amp in modes:
                result=train(replace(config,amp=amp),bundle=bundle)
                payload=torch.load(Path(result['output_dir'])/'test_predictions.pt',weights_only=False)
                self.assertTrue(torch.isfinite(payload['prediction_map']).all())
                self.assertTrue((payload['prediction_map']>=0).all())
                torch.testing.assert_close(payload['prediction_map'],torch.relu(payload['candidate_map']))


if __name__ == '__main__':
    unittest.main()
