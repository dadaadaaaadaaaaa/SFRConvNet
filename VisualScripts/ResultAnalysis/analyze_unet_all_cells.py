"""汇总四折所有时间与coarse cells，诊断逐格成星识别和SFR幅度。"""
from pathlib import Path
import argparse
import json
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from VisualScripts.ResultAnalysis.plot_unet_residual_scatter import SOURCES
from VisualScripts.ResultAnalysis.analyze_sfr_unet import load_predictions,file_hash,regression

EPS=1e-4
THRESHOLDS=[0.,1e-6,EPS,1e-3]
LABELS={'fiducial':'Fiducial','fiducial4':'Fiducial4','fiducial7':'Fiducial7','model075':'Model0.75'}


def classification(y,p,eps):
    """行是真实类别，列是预测类别；报告计数、条件比例和SFR贡献。"""
    actual=y>eps;predicted=p>eps
    tn=(~actual)&(~predicted);fp=(~actual)&predicted;fn=actual&(~predicted);tp=actual&predicted
    counts=np.array([[tn.sum(),fp.sum()],[fn.sum(),tp.sum()]])
    total=y.sum()
    return dict(epsilon=eps,TN=int(counts[0,0]),FP=int(counts[0,1]),FN=int(counts[1,0]),TP=int(counts[1,1]),
        true_inactive_N=int((~actual).sum()),true_active_N=int(actual.sum()),
        false_positive_rate=float(fp.sum()/(~actual).sum()) if (~actual).any() else np.nan,
        false_negative_rate=float(fn.sum()/actual.sum()) if actual.any() else np.nan,
        precision=float(tp.sum()/predicted.sum()) if predicted.any() else np.nan,
        recall=float(tp.sum()/actual.sum()) if actual.any() else np.nan,
        fp_predicted_sum=float(p[fp].sum()),fp_true_sum=float(y[fp].sum()),
        fn_true_sum=float(y[fn].sum()),fn_predicted_sum=float(p[fn].sum()),
        fp_excess_pct_true_sum=float(100*(p[fp]-y[fp]).sum()/total),
        fn_deficit_pct_true_sum=float(100*(y[fn]-p[fn]).sum()/total))


def budget(y,p):
    """精确零值虚报、正标签高估、正标签低估三项之和等于总偏差。"""
    e=p-y;positive=y>0
    parts=np.array([p[y==0].sum(),np.maximum(e[positive],0).sum(),-np.maximum(-e[positive],0).sum()])
    if not np.isclose(parts.sum(),e.sum(),rtol=1e-10,atol=1e-9):raise ValueError('误差收支恒等式失败')
    return parts/y.sum()*100


def bins(y,p,edges):
    """只对正真值分箱；保留低样本量箱并记录N。"""
    positive=y>0;y=y[positive];p=p[positive]
    ids=np.searchsorted(edges,y,side='right')-1;ids[y==edges[-1]]=len(edges)-2
    if (ids<0).any() or (ids>=len(edges)-1).any():raise ValueError('分箱范围未覆盖全部正标签')
    rows=[]
    for i in range(len(edges)-1):
        chosen=ids==i
        if not chosen.any():continue
        a,b=y[chosen],p[chosen]
        rows.append(dict(bin=i,lower=edges[i],upper=edges[i+1],N=len(a),true_sum=a.sum(),predicted_sum=b.sum(),
            true_mean=a.mean(),ratio=b.sum()/a.sum(),median_ratio=np.median(b/a),
            median_prediction=np.median(b),under_fraction=(b<a).mean()))
    return pd.DataFrame(rows)


def confusion(ax,row,title):
    counts=np.array([[row['TN'],row['FP']],[row['FN'],row['TP']]])
    percent=np.divide(counts,counts.sum(1,keepdims=True),out=np.zeros_like(counts,dtype=float),where=counts.sum(1,keepdims=True)>0)*100
    ax.imshow(percent,cmap='Blues',vmin=0,vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f"{counts[i,j]:,}\n{percent[i,j]:.2f}% of row",ha='center',va='center',fontsize=10,
                    color='white' if percent[i,j]>55 else '#172c3b')
    ax.set(xticks=[0,1],xticklabels=['Inactive','Active'],yticks=[0,1],yticklabels=['Inactive','Active'],
           xlabel='Predicted class',ylabel='True class',title=title)


def density(ax,y,p,edges,norm,title):
    h,_,_=np.histogram2d(y,p,bins=(edges,edges))
    if int(h.sum())!=len(y):raise ValueError('密度图丢失样本')
    m=ax.pcolormesh(edges,edges,np.ma.masked_equal(h.T,0),norm=norm,cmap='viridis',rasterized=True)
    ax.plot([0,edges[-1]],[0,edges[-1]],ls='--',color='#d9534f',lw=1)
    ax.set_xscale('symlog',linthresh=EPS);ax.set_yscale('symlog',linthresh=EPS)
    ax.set(xlim=(0,edges[-1]),ylim=(0,edges[-1]),xlabel='True cell SFR [Msun/yr]',ylabel='Predicted cell SFR [Msun/yr]',title=title)
    return m


def save(fig,path):
    for suffix in ['png','pdf','svg']:
        target=path.with_suffix('.'+suffix);fig.savefig(target,dpi=180,bbox_inches='tight')
        if suffix=='svg':target.write_text('\n'.join(s.rstrip() for s in target.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    plt.rcdefaults();plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.size':10})
    hashes={};metrics=[];classes=[];binned=[];high=[];budgets=[];all_data={};audits={}
    for case,(folder,analysis,group) in SOURCES.items():
        root=ROOT/'Results'/folder
        contract=json.loads((root/'contract.json').read_text(encoding='utf-8'))
        manifest=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'))
        prior=json.loads((root/analysis/'analysis.json').read_text(encoding='utf-8'))
        for name in ['out_predictions.pt','contract.json','split_manifest.json','summary.json']:
            digest=file_hash(root/name)
            if digest!=prior['sources'][name]:raise ValueError('来源摘要变化')
            hashes[str((root/name).relative_to(ROOT))]=digest
        data=load_predictions(root/'out_predictions.pt',identity=manifest['out'],expected_groups={group},expected_loss=contract['loss'])
        if data['target_map'].shape[1:]!=(1,12,20):raise ValueError('逐格地图shape错误')
        arrays=[data[name+'_map'].reshape(-1)*.025 for name in ['target','prediction','baseline']]
        y,p,b=arrays
        if min(p.min(),b.min())<0:raise ValueError('本诊断要求非负逐格预测')
        audits[case]=dict(result_dir=folder,N_snapshots=len(data['time']),N_cells=len(y),shape=list(data['target_map'].shape),
                          true_zero_N=int((y==0).sum()),true_zero_fraction=float((y==0).mean()))
        all_data[case]=(y,p,b)
        del data;gc.collect()
        print(case,len(y),'cells verified',flush=True)
    peak=max(float(a.max()) for values in all_data.values() for a in values)
    upper=10**np.ceil(np.log10(max(peak,1.)))
    density_edges=np.r_[np.linspace(0,EPS,17),np.geomspace(EPS,upper,120)[1:]]
    positive_edges=np.r_[0.,np.geomspace(1e-8,EPS,13)[:-1],np.geomspace(EPS,upper,25)]
    norm=LogNorm(vmin=1,vmax=max(len(v[0]) for v in all_data.values()))
    for case,(y,p,b) in all_data.items():
        rows={};curves={};parts={}
        for model,pred in [('UNet',p),('Baseline',b)]:
            metrics.append(dict(case=case,model=model,N=len(y),**regression(y,pred),sum_recovery=pred.sum()/y.sum()))
            for eps in THRESHOLDS:
                row=dict(case=case,model=model,**classification(y,pred,eps));classes.append(row)
                if eps==EPS:rows[model]=row
            curves[model]=bins(y,pred,positive_edges)
            binned.extend(curves[model].assign(case=case,model=model).to_dict('records'))
            parts[model]=budget(y,pred)
            budgets.append(dict(case=case,model=model,zero_false_positive_pct=parts[model][0],positive_over_pct=parts[model][1],
                                positive_under_pct=parts[model][2],net_bias_pct=parts[model].sum()))
            for threshold in [.1,1.,10.]:
                chosen=y>=threshold
                if chosen.any():high.append(dict(case=case,model=model,threshold=threshold,N=int(chosen.sum()),
                    true_sum_fraction=y[chosen].sum()/y.sum(),ratio=pred[chosen].sum()/y[chosen].sum(),under_fraction=(pred[chosen]<y[chosen]).mean()))
        fig,axes=plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
        for ax,pred,model in [(axes[0,0],p,'UNet'),(axes[0,1],b,'Baseline')]:
            mesh=density(ax,y,pred,density_edges,norm,model+': all snapshots and cells')
            fig.colorbar(mesh,ax=ax,label='Number of time-cell samples')
        confusion(axes[0,2],rows['UNet'],'UNet classification (epsilon = 1e-4)')
        confusion(axes[1,0],rows['Baseline'],'Baseline classification (epsilon = 1e-4)')
        ax=axes[1,1]
        for model,color in [('UNet','#087f8c'),('Baseline','#999999')]:
            curve=curves[model]
            curve=curve[curve.lower>=EPS]
            ax.plot(curve.true_mean,curve.ratio,'o-',ms=3,lw=1,label=model,color=color)
        ax.set_xscale('log');ax.set_yscale('symlog',linthresh=.1)
        ax.axhline(1,ls='--',color='#d9534f',lw=1)
        ax.set(xlim=(EPS,upper),xlabel='Mean true cell SFR in bin [Msun/yr]',ylabel='Sum predicted / sum true',title='SFR recovery (true cell SFR >= 1e-4)')
        ax.legend();ax.grid(alpha=.2)
        ax=axes[1,2];x=np.arange(3)
        for j,(model,color) in enumerate([('UNet','#087f8c'),('Baseline','#999999')]):
            bars=ax.bar(x+(j-.5)*.34,parts[model],width=.3,color=color,label=model)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.axhline(0,color='gray',lw=.7);ax.margins(y=.2)
        ax.set(xticks=x,xticklabels=['True-zero\nfalse SFR','Positive-cell\noverestimate','Positive-cell\nunderestimate'],
               ylabel='% of snapshot-summed true SFR',title='Exact-zero error budget (epsilon = 0)')
        ax.legend(fontsize=9)
        fig.suptitle(f"Held-out {LABELS[case]} | {audits[case]['N_snapshots']:,} snapshots x 240 cells\n"
                     'Cell-integrated instantaneous SFR; classification uses a fixed diagnostic threshold',fontsize=14)
        save(fig,output/f'{case}_all_cells')
    pd.DataFrame(metrics).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(output/'classification_thresholds.csv',index=False)
    pd.DataFrame(binned).to_csv(output/'positive_sfr_bins.csv',index=False)
    pd.DataFrame(high).to_csv(output/'high_sfr_cells.csv',index=False)
    pd.DataFrame(budgets).to_csv(output/'error_budget.csv',index=False)
    # 四个留出星系的模型密度使用统一坐标和颜色标尺。
    fig,axes=plt.subplots(2,2,figsize=(11,10),constrained_layout=True)
    for ax,(case,(y,p,b)) in zip(axes.flat,all_data.items()):
        mesh=density(ax,y,p,density_edges,norm,LABELS[case]+' | UNet')
    fig.colorbar(mesh,ax=axes.ravel().tolist(),label='Number of time-cell samples')
    save(fig,output/'all_four_unet_density')
    hashes[str(Path(__file__).relative_to(ROOT))]=file_hash(Path(__file__))
    (output/'sources.json').write_text(json.dumps(dict(sha256=hashes,experiments=audits,epsilon=EPS,thresholds=THRESHOLDS,
        density_edges=density_edges.tolist(),positive_bin_edges=positive_edges.tolist(),
        units='Msun/yr per coarse cell; no additional volume multiplication',
        weighting='Each time-cell sample equal weight; sums are over snapshots, not time integrals or formed masses',
        limitations='Unequal cell volumes; correlated adjacent times/cells; no confidence intervals; no HR subcell recovery claimed',
        threshold_definition='Fixed diagnostic choices, independent of held-out error; epsilon=0 separately preserves exact-zero semantics'),
        ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args().output_dir)
