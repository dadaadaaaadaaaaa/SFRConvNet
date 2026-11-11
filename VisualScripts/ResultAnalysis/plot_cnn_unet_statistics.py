"""从已对齐的四折历史指标绘制组会统计图，保留baseline与来源审计。"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from VisualScripts.ResultAnalysis.analyze_sfr_unet import file_hash

SOURCE=ROOT/'VisualScripts/ResultAnalysis/outputs/cnn_unet_histories_baseline_20261111'
CASES=['fiducial','fiducial4','fiducial7','model075']
LABELS=['Fiducial','Fiducial4','Fiducial7','Model0.75']
COLORS={'CNN':'#d87935','UNet':'#087f8c','Baseline':'#b7bec7'}


def save(fig,path):
    for ext in ['png','pdf','svg']:
        target=path.with_suffix('.'+ext)
        fig.savefig(target,dpi=230,facecolor='white')
        if ext=='svg':
            target.write_text('\n'.join(v.rstrip() for v in target.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    data=pd.read_csv(SOURCE/'metrics.csv')
    sources=json.loads((SOURCE/'sources.json').read_text(encoding='utf-8'))
    if data[['case','model']].duplicated().any():raise ValueError('重复指标行')
    for case in CASES:
        rows=data[data.case==case]
        if set(rows.model)!={'CNN','UNet','Baseline'} or rows.N.nunique()!=1:raise ValueError('模型集合或共同帧数不一致')
        if rows.N.iloc[0]!=sources['sources'][case]['matched_N']:raise ValueError('帧数与来源审计不一致')
    fields=['MAE','RMSE','Bias','R2','MedianAbsRelativeError_nonzero']
    if not np.isfinite(data[fields].to_numpy()).all():raise ValueError('指标存在非有限值')
    if (data[['MAE','RMSE','MedianAbsRelativeError_nonzero']]<0).any().any():raise ValueError('误差指标为负')
    records=[]
    indexed=data.set_index(['case','model'])
    for case in CASES:
        for metric in fields:
            c,u,b=[float(indexed.loc[(case,m),metric]) for m in ['CNN','UNet','Baseline']]
            records.append(dict(case=case,metric=metric,N=int(indexed.loc[(case,'CNN'),'N']),
                CNN=c,UNet=u,Baseline=b,UNet_minus_CNN=u-c,
                error_reduction_pct=100*(1-u/c) if metric in ['MAE','RMSE','MedianAbsRelativeError_nonzero'] and c>0 else np.nan))
    summary=pd.DataFrame(records)
    summary.to_csv(output/'comparison.csv',index=False)
    data.to_csv(output/'metrics.csv',index=False)
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(19,10.4))
    fig.subplots_adjust(left=.065,right=.985,bottom=.13,top=.83,hspace=.48,wspace=.27)
    fig.text(.065,.965,'CNN 与 UNet：四个留出星系的预测统计',fontsize=23,weight='bold',color='#19384b')
    fig.text(.065,.922,'同一批 snapshot 逐帧对齐 · 物理空间指标 · 每个星系独立统计',fontsize=12,color='#59636b')
    fig.legend(handles=[Patch(facecolor=COLORS[m],label={'Baseline':'MACER-LR baseline'}.get(m,m))
                        for m in ['CNN','UNet','Baseline']],loc='upper right',bbox_to_anchor=(.985,.945),ncol=3,frameon=False)
    definitions=[('MAE','平均绝对误差 MAE ↓','SFR [M☉/yr]',1),
        ('RMSE','均方根误差 RMSE ↓','SFR [M☉/yr]',1),
        ('MedianAbsRelativeError_nonzero','中位绝对相对误差 ↓','相对误差 [%]',100),
        ('R2','决定系数 R² ↑','R²',1),('Bias','平均偏差 Bias → 0','预测 − 真值 [M☉/yr]',1)]
    x=np.arange(4);width=.24
    for ax,(metric,title,ylabel,factor) in zip(axes.flat,definitions):
        for j,model in enumerate(['CNN','UNet','Baseline']):
            values=np.array([indexed.loc[(case,model),metric] for case in CASES])*factor
            bars=ax.bar(x+(j-1)*width,values,width=.21,color=COLORS[model],zorder=3)
            labels=[f'{v:.1f}' if factor==100 else f'{v:.2f}' for v in values]
            ax.bar_label(bars,labels=labels,padding=4,fontsize=9,color='#253740')
        ax.set_title(title,loc='left',fontsize=14,pad=14,weight='bold')
        ax.set_xticks(x,LABELS);ax.set_ylabel(ylabel,fontsize=10)
        ax.grid(axis='y',alpha=.15,zorder=0);ax.set_axisbelow(True)
        ax.margins(y=.22)
        if metric in ['R2','Bias']:ax.axhline(0,color='#66717a',lw=.8,zorder=2)
        if metric=='R2':ax.set_ylim(min(-2.,data.R2.min()-.25),1.3)
    ax=axes[1,2]
    gain=summary[summary.metric=='MAE'].set_index('case').loc[CASES,'error_reduction_pct'].to_numpy()
    bars=ax.bar(x,gain,width=.56,color=COLORS['UNet'],zorder=3)
    ax.bar_label(bars,labels=[f'{v:.1f}%' for v in gain],padding=6,fontsize=13,weight='bold',color=COLORS['UNet'])
    ax.set(xticks=x,xticklabels=LABELS,ylim=(0,105),ylabel='MAE 降低 [%]')
    ax.set_title('UNet 相对 CNN 的 MAE 改善 ↑',loc='left',fontsize=14,pad=14,weight='bold')
    ax.grid(axis='y',alpha=.15,zorder=0);ax.set_axisbelow(True)
    fig.text(.065,.065,'共同帧数：Fiducial 8429 · Fiducial4 8770 · Fiducial7 7025 · Model0.75 954',fontsize=10,color='#59636b')
    fig.text(.065,.035,'Fiducial4 / Model0.75 的 CNN 为历史实验；模型间训练设置、监督或输入存在差异。统计描述当前实验效果。',fontsize=10,color='#59636b')
    save(fig,output/'statistics_overview')
    # 归一化改善图突出各折进步幅度，三个误差指标沿用同一百分比坐标。
    fig,ax=plt.subplots(figsize=(12.8,6.5),layout='constrained')
    metrics=['MAE','RMSE','MedianAbsRelativeError_nonzero']
    for j,(metric,label,color) in enumerate(zip(metrics,['MAE','RMSE','中位绝对相对误差'],['#087f8c','#4477aa','#9467bd'])):
        values=summary[summary.metric==metric].set_index('case').loc[CASES,'error_reduction_pct'].to_numpy()
        bars=ax.bar(x+(j-1)*.24,values,width=.21,color=color,label=label)
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=4,fontsize=11)
    ax.set(xticks=x,xticklabels=LABELS,ylim=(0,110),ylabel='相对 CNN 的误差降低 [%]')
    ax.set_title('UNet 的误差改善幅度：按留出星系比较',loc='left',fontsize=18,pad=24)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper center',ncol=3,frameon=False)
    fig.supxlabel('误差降低 = 100 × (1 − UNet误差 / CNN误差)；数值越高，改善幅度越大',fontsize=11)
    save(fig,output/'error_reduction')
    (output/'sources.json').write_text(json.dumps(dict(
        sha256={str(p.relative_to(ROOT)):file_hash(p) for p in [SOURCE/'metrics.csv',SOURCE/'sources.json',Path(__file__)]},
        model_sources=sources['sources'],
        units='MAE/RMSE/Bias: Msun/yr; relative error: nonzero true Y only; plot percentages multiply by 100',
        comparison='Exact same models and matched snapshots as the preceding SFR history plots',
        uncertainty='Descriptive point estimates for fixed models; adjacent frames are correlated; no significance or seed uncertainty estimated',
        definitions='Bias=mean(prediction-truth); R2 relative change is not used; error_reduction only for positive error metrics'),
        ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)
