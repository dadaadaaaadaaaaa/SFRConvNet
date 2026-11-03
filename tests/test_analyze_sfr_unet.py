"""空间预测分析的单位、负值、积分、缺口、事件删失及正式结果兼容检查。"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from VisualScripts.ResultAnalysis.analyze_sfr_unet import (load_predictions, regression,
    integrate_series, threshold_event, burst_metrics, analyze_result)


def fixture(root):
    """构造与正式导出相同schema，真值每帧总量40code=1Msun/yr。"""
    root=Path(root); manifest={"train":{"sample_uid":["train"],"parent_hr_uid":["parent_train"]},
        "val":{"sample_uid":["val"],"parent_hr_uid":["parent_val"]}}
    for split,group in (("test","run_a"),("out","run_b")):
        y=torch.ones(3,1,4,4,dtype=torch.float64)*2.5
        b=y*.5; p=y*.8
        # 总量不变但局部出现负预测，分析不能裁剪。
        p[1,0,0,0]=-1.; p[1,0,0,1]+=3.
        ids={"sample_uid":[f"{split}{i}" for i in range(3)],"parent_hr_uid":[f"parent_{split}{i}" for i in range(3)]}
        manifest[split]=ids
        torch.save(dict(schema="macnet_sfr_unet_predictions_v1",units="MACER_mass_code/Gyr",to_msun_per_yr=.025,
                        prediction_map=p,target_map=y,baseline_map=b,closure_map=p-b,
                        prediction_total=p.sum((1,2,3)),target_total=y.sum((1,2,3)),baseline_total=b.sum((1,2,3)),
                        groups=[group]*3,time=torch.tensor([0.,1.,2.]),**ids),root/f"{split}_predictions.pt")
    (root/"split_manifest.json").write_text(json.dumps(manifest))
    (root/"summary.json").write_text(json.dumps(dict(status="complete",hpo=False,best_epoch=1)))
    (root/"contract.json").write_text(json.dumps(dict(model=dict(architecture="SFRUNet"),cache_metadata=dict(
        experiment="exp2",train_groups=["run_a"],held_out_groups=["run_b"]))))
    return manifest


class AnalysisTests(unittest.TestCase):
    def test_integral_units_gap_and_duplicate_time(self):
        mass,valid=integrate_series([0,1,2],np.ones(3))
        self.assertEqual(mass[-1],2e9)
        mass,valid=integrate_series([0,1,2,20,21],np.ones(5))
        self.assertEqual(mass[-1],3e9); self.assertEqual(valid.tolist(),[True,True,False,True])
        with self.assertRaises(ValueError):
            integrate_series([0,0],[1,1])

    def test_zero_truth_and_threshold_censoring(self):
        result=regression(np.zeros(3),np.array([-1.,0.,1.]))
        self.assertIsNone(result["R2"]); self.assertIsNone(result["MedianAbsRelativeError_nonzero"])
        self.assertEqual(result["NegativePredFrac"],1/3)
        event=threshold_event(np.arange(4.),np.array([0.,2.,2.,0.]),1.)
        self.assertEqual(event["start_gyr"],.5); self.assertEqual(event["end_gyr"],2.5)
        self.assertTrue(threshold_event(np.arange(3.),np.ones(3)*2,1)["right_censored"])

    def test_burst_edges_and_missing_event(self):
        frame=pd.DataFrame(dict(time_gyr=[0.,1.,2.,3.],type=["run"]*4,
              true_sfr_msun_per_yr=[0.,2.,2.,0.],model_sfr_msun_per_yr=[0.]*4,baseline_sfr_msun_per_yr=[0.]*4))
        row=burst_metrics(frame,(.25,2.75),threshold=1)[0]
        self.assertAlmostEqual(row["true_formed_mass_msun"],3.875e9)
        self.assertEqual(row["mass_relative_error"],-1)
        self.assertIsNone(row["start_error_gyr"])
        with self.assertRaises(ValueError):
            burst_metrics(frame,(-1,3))

    def test_export_analysis_maps_and_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); ids=fixture(root)
            result=analyze_result(root,plots=True,burst_window=(0.,2.),burst_threshold=.9)
            table=result["integrated_metrics"]
            self.assertAlmostEqual(table.iloc[0]["Model Error all time"],-20)
            self.assertAlmostEqual(table.iloc[0]["Baseline Error all time"],-50)
            output=Path(result["output_dir"])
            self.assertTrue((output/"out_run_b"/"cumulative_sfr.png").is_file())
            self.assertFalse((output/"test_run_a"/"cumulative_sfr.png").exists())
            self.assertTrue((output/"burst_metrics.csv").exists())
            with self.assertRaises(FileExistsError):
                analyze_result(root,output_dir=output,plots=False)
            path=root/"out_predictions.pt"
            payload=torch.load(path,weights_only=False); payload["prediction_total"][0]+=1; torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,"求和"):
                load_predictions(path,identity=ids["out"],expected_groups=["run_b"])

    def test_notebook_syntax_and_all_code_cells_documented(self):
        path=Path(__file__).resolve().parents[1]/"VisualScripts/ResultAnalysis/Results_stat_Exp2_SFRUNet.ipynb"
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            if cell["cell_type"]=="code":
                source="".join(cell["source"])
                self.assertTrue(source.startswith("#")); compile(source,str(path),"exec")


if __name__=="__main__":
    unittest.main()
