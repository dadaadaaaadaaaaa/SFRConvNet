"""将四折UNet残差与预测散点上下排列，共用真实SFR横轴。"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from VisualScripts.ResultAnalysis.analyze_sfr_unet import file_hash

SOURCES={
    'fiducial':('Exp2_LOO_Fiducial_SFRUNet_ReLU_Candidate1_seed43_20261107_041950_b08b68f2','analysis_sfr_unet_20261110_140345_b48a71','disk_galaxy_fiducial'),
    'fiducial4':('Exp2_LOO_Fiducial4_SFRUNet_ReLU_Candidate1_Final_20261111_175216_a6a6b5c3','analysis_sfr_unet_20261111_182713_92d23b','disk_galaxy_fiducial_4'),
    'fiducial7':('Exp2_LOO_Fiducial7_SFRUNet_ReLU_Candidate1_Final_20261110_031212_9b4f1351','analysis_sfr_unet_20261110_033819_005bb4','disk_galaxy_fiducial_7'),
    'model075':('Exp2_LOO_Model075_SFRUNet_ReLU_Candidate1_Final_20261109_142102_519eb27d','analysis_sfr_unet_20261109_170220_dc6eab','disk_galaxy_model_0_75'),
}


def draw(axes,frame,title):
    """沿用原scatter样式，残差使用相同散点参数，避免多值关系的折线连接。"""
    top,bottom=axes
    truth=frame.true_sfr_msun_per_yr.to_numpy()
    for label in ['model','baseline']:
        prediction=frame[label+'_sfr_msun_per_yr'].to_numpy()
        error=frame[label+'_error'].to_numpy()
        if not np.allclose(error,prediction-truth,rtol=1e-10,atol=1e-10):raise ValueError('残差与预测真值不一致')
        top.scatter(truth,error,s=5,label=label,alpha=.5)
        bottom.scatter(truth,prediction,s=5,label=label,alpha=.5)
    limits=[min(0.,float(frame.model_sfr_msun_per_yr.min())),
            float(frame[['true_sfr_msun_per_yr','model_sfr_msun_per_yr','baseline_sfr_msun_per_yr']].max().max())]
    bottom.plot(limits,limits,color='gray',ls='--')
    top.axhline(0,color='gray',lw=.6)
    top.set_ylabel('Signed error [Msun/yr]');top.legend()
    bottom.set_xlabel('True SFR [Msun/yr]');bottom.set_ylabel('Predicted SFR [Msun/yr]');bottom.legend()
    top.set_title(title)
    return dict(N=len(frame),xlim=list(bottom.get_xlim()),signed_error_ylim=list(top.get_ylim()),
                predicted_sfr_ylim=list(bottom.get_ylim()))


def save(fig,path):
    for suffix in ['png','pdf','svg']:
        target=path.with_suffix('.'+suffix)
        fig.savefig(target,dpi=150,bbox_inches='tight')
        if suffix=='svg':
            target.write_text('\n'.join(s.rstrip() for s in target.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    hashes={};audit={};frames={}
    plt.rcdefaults()
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none'})
    for case,(folder,analysis,group) in SOURCES.items():
        root=ROOT/'Results'/folder;source=root/analysis
        manifest=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'))['out']
        metadata=json.loads((source/'analysis.json').read_text(encoding='utf-8'))
        for name in ['out_predictions.pt','contract.json','summary.json','split_manifest.json']:
            path=root/name;digest=file_hash(path)
            if digest!=metadata['sources'][name]:raise ValueError('分析来源文件摘要不匹配')
            hashes[str(path.relative_to(ROOT))]=digest
        frame=pd.read_csv(source/'out_predictions.csv')
        if frame.sample_uid.duplicated().any() or set(frame.sample_uid)!=set(manifest['sample_uid']):raise ValueError('UID不一致')
        parent=dict(zip(manifest['sample_uid'],manifest['parent_hr_uid']))
        if not np.array_equal(frame.parent_hr_uid.to_numpy(),frame.sample_uid.map(parent).to_numpy()):raise ValueError('parent不一致')
        if set(frame['type'])!={group} or set(frame.split)!={'out'}:raise ValueError('留出运行不一致')
        cols=['true_sfr_msun_per_yr','model_sfr_msun_per_yr','baseline_sfr_msun_per_yr','model_error','baseline_error']
        if not np.isfinite(frame[cols].to_numpy()).all():raise ValueError('非有限数据')
        for path in [source/'out_predictions.csv',source/'analysis.json',source/f'out_{group}'/'scatter_and_cell_error.png',source/f'out_{group}'/'sfr_history.png']:
            hashes[str(path.relative_to(ROOT))]=file_hash(path)
        frames[case]=frame
        fig,axes=plt.subplots(2,1,figsize=(6.5,8),sharex=True,constrained_layout=True,
                              gridspec_kw={'height_ratios':[1,2],'hspace':.03})
        audit[case]=dict(result_dir=folder,analysis_dir=analysis,**draw(axes,frame,group+' / out'))
        if not np.array_equal(axes[0].get_xlim(),axes[1].get_xlim()):raise ValueError('上下横轴范围不一致')
        save(fig,output/f'{case}_signed_error_and_scatter')
        print(case,len(frame),'verified',flush=True)
    fig=plt.figure(figsize=(13,16),constrained_layout=True)
    grid=fig.add_gridspec(2,2)
    for index,(case,frame) in enumerate(frames.items()):
        sub=grid[index//2,index%2].subgridspec(2,1,height_ratios=[1,2],hspace=.03)
        top=fig.add_subplot(sub[0]);bottom=fig.add_subplot(sub[1],sharex=top)
        plt.setp(top.get_xticklabels(),visible=False)
        draw([top,bottom],frame,SOURCES[case][2]+' / out')
    save(fig,output/'all_four_signed_error_and_scatter')
    hashes[str(Path(__file__).relative_to(ROOT))]=file_hash(Path(__file__))
    (output/'sources.json').write_text(json.dumps(dict(sha256=hashes,experiments=audit,
        parameters=dict(scatter_size=5,scatter_alpha=.5,colors=['tab:blue','tab:orange'],scale='linear',zero_line_width=.6,
            identity_line='gray dashed',dpi=150,layout='signed error above; predicted SFR below; shared True SFR x-axis',
            signed_error='prediction - true SFR',single_figure_inches=[6.5,8],height_ratio=[1,2]),
        handling='all out frames; no binning/smoothing/clipping; original physical units Msun/yr; residual scatter uses original prediction-scatter style'),
        ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)
