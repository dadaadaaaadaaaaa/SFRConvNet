"""审计已完成模型的闭合项时间变化、幅值分层误差和跨 cache 共性。

仅写入独立分析目录；不修改训练、原始数据、cache 或既有结果。
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from VisualScripts.ResultAnalysis import analysis_utils as utils

MAIN = 'Exp2_Rh12x20_FiveRuns_Closure_LOO_Model075_Trial53_Final_20260913_004906'
MODELS = {
    'Exp2_LOO': (MAIN, ['out', 'test']),
    'Exp1_random': ('Exp1_Rh12x20_Closure_BestTrial55_seed42_20260910_111655', ['test']),
    'Exp3.2_random': ('Exp3.2_Closure_Trial23_seed42_20261013_142007', ['val']),
}
CACHES = {
    'Exp1_8x8_v6': 'Exp1_Rh8x8_Closure_auto_',
    'Exp1_12x20_v6': 'Exp1_Rh12x20_Closure_',
    'Exp2_12x20_v6': 'Exp2_Rh12x20_FiveRuns_Closure_LOO_disk_galaxy_model_0_75_',
    'Exp2_12x20_v7': 'Exp2_Rh12x20_FiveRuns_Closure_v7_LOO_disk_galaxy_fiducial_',
    'Exp3.2_12x20_v7': 'Exp3_Rh12x20_FiveRuns_Closure_Exp3.2_',
}
FACTOR = .025
BINS = [0., 1., 3., 10., 20., np.inf]
LABELS = ['[0,1)', '[1,3)', '[3,10)', '[10,20)', '[20,inf)']


def fingerprint(path):
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def read_cache(path):
    """只保留小型物理量表，并核对重复 UID 对应同一个物理样本。"""
    d = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    t = d['tensors']
    f = pd.DataFrame({
        'Y': np.asarray(t['target_sfr_physical']).reshape(-1)*FACTOR,
        'B': np.asarray(t['macer_sfr_baseline']).reshape(-1)*FACTOR,
        'C': np.asarray(t['closure_true_physical']).reshape(-1)*FACTOR,
        'time': np.asarray(t['time']).reshape(-1),
        'group': np.asarray(t['groups']).reshape(-1),
        'uid': d['sample_identity']['sample_uid'],
        'parent_uid': d['sample_identity']['parent_hr_uid'],
    })
    assert np.isfinite(f[['Y','B','C','time']]).all().all()
    assert np.allclose(f.Y, f.B+f.C, rtol=2e-6, atol=1e-6)
    duplicates = f[f.duplicated('uid', keep=False)]
    if len(duplicates):
        assert duplicates.groupby('uid')[['Y','B','C','time','group','parent_uid']].nunique().max().max() == 1
    n_stored = len(f)
    f = f.drop_duplicates('uid').copy()
    assert not f.parent_uid.duplicated().any()
    return f, n_stored


def physical_predictions(f):
    result = pd.DataFrame({
        'Y': f.sfr_true*FACTOR, 'B': f.sfr_macer_baseline*FACTOR,
        'C': f.closure_true_physical*FACTOR,
        'Yhat': f.sfr_pred*FACTOR, 'Chat': f.closure_pred_physical*FACTOR,
        'time': f.time, 'group': f.type,
        'uid': f.sample_uid, 'parent_uid': f.parent_hr_uid,
    })
    assert np.isfinite(result[['Y','B','C','Yhat','Chat','time']]).all().all()
    assert np.allclose(result.Y, result.B+result.C, rtol=2e-6, atol=1e-6)
    assert np.allclose(result.Yhat, result.B+result.Chat, rtol=2e-6, atol=1e-6)
    assert np.allclose(result.Yhat-result.Y, result.Chat-result.C, rtol=2e-5, atol=1e-5)
    assert not result.uid.duplicated().any()
    return result


def metrics(f):
    y, c = f.Y.to_numpy(), f.C.to_numpy()
    e = (f.Chat-f.C).to_numpy()
    sst_y, sst_c = np.sum((y-y.mean())**2), np.sum((c-c.mean())**2)
    nz_y, nz_c = y > 1e-4, abs(c) > 1e-3
    return dict(n=len(f), C_mean=float(c.mean()), C_median=float(np.median(c)),
        mae=float(np.mean(abs(e))), rmse=float(np.sqrt(np.mean(e**2))), bias=float(e.mean()),
        r2_Y=float(1-np.sum(e**2)/sst_y) if sst_y>0 else None,
        r2_C=float(1-np.sum(e**2)/sst_c) if sst_c>0 else None,
        baseline_mae=float(np.mean(abs(c))), baseline_rmse=float(np.sqrt(np.mean(c**2))),
        mae_skill=float(1-np.sum(abs(e))/np.sum(abs(c))) if np.sum(abs(c)) else None,
        mse_skill=float(1-np.sum(e**2)/np.sum(c**2)) if np.sum(c**2) else None,
        median_abs_relative_C=float(np.median(abs(e[nz_c])/abs(c[nz_c]))) if nz_c.any() else None,
        median_abs_relative_Y=float(np.median(abs(e[nz_y])/y[nz_y])) if nz_y.any() else None,
        median_Chat_over_C=float(np.median(f.Chat.to_numpy()[nz_c]/c[nz_c])) if nz_c.any() else None,
        median_C_over_Y=float(np.median(c[nz_y]/y[nz_y])) if nz_y.any() else None,
        negative_Yhat=int((f.Yhat<0).sum()), relative_C_n=int(nz_c.sum()),relative_Y_n=int(nz_y.sum()))


def distribution(f):
    c, y = f.C.to_numpy(), f.Y.to_numpy()
    nz = y>1e-4
    q = np.quantile(abs(c), [.1,.5,.9,.99,1])
    return dict(n=len(f), zero_C=int((c==0).sum()),negative_C=int((c<0).sum()),
        C_abs_p10=q[0], C_abs_p50=q[1], C_abs_p90=q[2], C_abs_p99=q[3], C_abs_max=q[4],
        median_C_over_Y=np.median(c[nz]/y[nz]),
        frac_C_small_vs_Y=np.mean(abs(c[nz])/y[nz]<.1),
        relative_denominator_n=int(nz.sum()))


def integral(f):
    """仅用于单个、连续轨迹；有缺口时拒绝跨缺口计算。"""
    f = f.sort_values('time')
    dt = np.diff(f.time)
    assert (dt>0).all() and dt.max()<1.5*np.median(dt)
    true = np.trapezoid(f.Y, f.time)
    return dict(formed_mass_true_msun=true*1e9,
        mass_bias_fraction=np.trapezoid(f.Yhat-f.Y,f.time)/true,
        baseline_mass_bias_fraction=np.trapezoid(f.B-f.Y,f.time)/true,
        integrated_absolute_error_fraction=np.trapezoid(abs(f.Yhat-f.Y),f.time)/true)


def stratify(f, name):
    rows=[]
    for lo, hi, label in zip(BINS[:-1], BINS[1:], LABELS):
        subset=f[(abs(f.C)>=lo)&(abs(f.C)<hi)]
        if len(subset):
            rows.append(dict(dataset=name,bin=label,**metrics(subset)))
    return rows


def block_bootstrap(f, width, repetitions=600):
    """固定物理时间块重采样；保留块内相关性，区间仅表示本轨迹的敏感性。"""
    f=f.sort_values('time').reset_index(drop=True)
    labels=np.floor((f.time-f.time.min())/width).astype(int)
    blocks=[np.flatnonzero(labels==k) for k in np.unique(labels)]
    rng=np.random.default_rng(20261025)
    estimates=[]
    for _ in range(repetitions):
        ix=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        s=f.iloc[ix]
        middle=s[(abs(s.C)>=1)&(abs(s.C)<3)]
        high=s[abs(s.C)>=10]
        if len(middle) and len(high):
            estimates.append((np.mean(abs(high.Chat-high.C))-np.mean(abs(middle.Chat-middle.C)),
                              np.mean(high.Chat-high.C),
                              np.median(abs(high.Chat-high.C)/abs(high.C))-np.median(abs(middle.Chat-middle.C)/abs(middle.C))))
    a=np.asarray(estimates)
    return dict(block_width_gyr=width, repetitions=len(a),
        mae_high_minus_middle_ci95=np.quantile(a[:,0],[.025,.975]).tolist(),
        bias_high_ci95=np.quantile(a[:,1],[.025,.975]).tolist(),
        median_relative_C_high_minus_middle_ci95=np.quantile(a[:,2],[.025,.975]).tolist())


def savefig(fig, out, name):
    fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(out/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def make_plots(out, held, bins, distributions, train_frames):
    f=held.sort_values('time');t=f.time-.5625
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(3,1,figsize=(12,10),sharex=True,layout='constrained')
    ax[0].plot(t,f.C,label=r'True $C=Y-B$',color='black',lw=1)
    ax[0].plot(t,f.Chat,label=r'Predicted $\hat C$',color='#d55e00',lw=.9,alpha=.85)
    ax[0].set_ylabel(r'Closure [$M_\odot$/yr]');ax[0].legend(loc='upper left')
    ax[1].plot(t,f.Y,label='True SFR Y',color='black',lw=1)
    ax[1].plot(t,f.Yhat,label='Predicted SFR',color='#d55e00',lw=.9,alpha=.85)
    ax[1].plot(t,f.B,label='MACER-LR baseline B',color='#0072b2',lw=.9)
    ax[1].set_ylabel(r'SFR [$M_\odot$/yr]');ax[1].legend(loc='upper left')
    ax[2].plot(t,f.Chat-f.C,color='#7b3294',lw=.9,label=r'$\hat C-C=\hat Y-Y$')
    ax[2].axhline(0,color='black',lw=.8);ax[2].set_ylabel(r'Signed error [$M_\odot$/yr]')
    ax[2].legend(loc='lower left');ax[2].set_xlabel('Production time [Gyr] = absolute time - 0.5625')
    for a in ax:
        a.axvspan(6,t.max(),color='#f0c36e',alpha=.15)
        a.axvline(6,color='#a46b00',ls='--',lw=.8)
        a.grid(alpha=.2)
    fig.suptitle('Exp2: fully held-out Model0.75 | shaded: existing burst window (t >= 6 Gyr)')
    savefig(fig,out,'heldout_closure_history')

    fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    sc=ax[0].scatter(f.C,f.Chat,c=t,cmap='viridis',s=12,alpha=.7)
    ax[0].plot([0,36],[0,36],color='black',ls='--',lw=1)
    ax[0].set(xlabel=r'True closure C [$M_\odot$/yr]',ylabel=r'Predicted closure [$M_\odot$/yr]',title='Amplitude and evolutionary phase')
    fig.colorbar(sc,ax=ax[0],label='Production time [Gyr]')
    sub=bins[bins.dataset=='Exp2_LOO_out'].set_index('bin').reindex(LABELS)
    x=np.arange(len(sub));ax[1].bar(x-.18,sub.mae,.36,label='CNN MAE',color='#d55e00')
    ax[1].bar(x+.18,sub.baseline_mae,.36,label='MACER-LR MAE',color='#0072b2')
    ax[1].set_xticks(x,[f'{s}\nn={int(n)}' for s,n in zip(LABELS,sub.n)])
    ax[1].set(xlabel=r'True |C| bin [$M_\odot$/yr]',ylabel=r'MAE [$M_\odot$/yr]',title='Correction helps most bins, but not all')
    ax[1].legend();ax[1].grid(axis='y',alpha=.2)
    savefig(fig,out,'heldout_closure_regimes')

    fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for name,frame in train_frames.items():
        c=np.sort(abs(frame.C.to_numpy()))
        ax[0].plot(c,np.arange(1,len(c)+1)/len(c),label=name,lw=1.4)
    ax[0].set_xscale('symlog',linthresh=1e-3)
    ax[0].set(xlabel=r'Train |C| [$M_\odot$/yr]',ylabel='Fraction of unique training snapshots',title='Training distributions (UID deduplicated)')
    ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
    for name,g in bins.groupby('dataset',sort=False):
        g=g.set_index('bin').reindex(LABELS)
        ax[1].plot(np.arange(len(g)),g.mae,marker='o',label=name)
    ax[1].set_yscale('log');ax[1].set_xticks(range(len(LABELS)),LABELS)
    ax[1].set(xlabel=r'True |C| bin [$M_\odot$/yr]',ylabel=r'MAE [$M_\odot$/yr]',title='Different models / evaluation domains')
    ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
    savefig(fig,out,'cross_cache_comparison')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'VisualScripts/ResultAnalysis/outputs/closure_regimes_20261025')
    parser.add_argument('--reuse-predictions',action='store_true',help='仅复用本脚本生成且来源指纹一致的推理结果')
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists() and not args.reuse_predictions:
        raise FileExistsError(f'拒绝覆盖已有目录：{out}')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audit={'units':'Msun/yr','device':str(device),'split_checks':{},'sources':[]}
    train_frames={};distribution_rows=[]
    for name,prefix in CACHES.items():
        path=ROOT/'.cache'/f'{prefix}train.pt'
        frame,stored=read_cache(path);train_frames[name]=frame
        distribution_rows.append(dict(cache=name,group='ALL',stored_n=stored,**distribution(frame)))
        for group,g in frame.groupby('group'):
            distribution_rows.append(dict(cache=name,group=group,stored_n=None,**distribution(g)))
        audit['sources'].append(fingerprint(path))
        print(f'[cache] {name}: stored={stored}, unique={len(frame)}',flush=True)
    evaluations={};all_bins=[];group_bins=[];overall=[];quantile_rows=[]
    for name,(result_name,splits) in MODELS.items():
        result=ROOT/'Results'/result_name
        params=json.loads((result/'hyperparameters.json').read_text(encoding='utf-8'))
        prefix=params['data_exp'];checkpoint=next(result.glob('*best_model.pth'))
        train,stored=read_cache(ROOT/'.cache'/f'{prefix}train.pt')
        identities={};parent_ids={}
        for split in ['train','val','test']+(['out'] if name=='Exp2_LOO' else []):
            frame,_=read_cache(ROOT/'.cache'/f'{prefix}{split}.pt')
            identities[split]=set(frame.uid);parent_ids[split]=set(frame.parent_uid)
        overlaps={}
        for a in identities:
            for b in identities:
                if a<b:
                    overlap=len(identities[a]&identities[b])+len(parent_ids[a]&parent_ids[b])
                    assert overlap==0,(name,a,b,overlap)
                    overlaps[a+'__'+b]=overlap
        audit['split_checks'][name]=overlaps
        model=None
        for split in splits:
            key=name+'_'+split;path=ROOT/'.cache'/f'{prefix}{split}.pt'
            source=[fingerprint(p) for p in [path,checkpoint,result/'hyperparameters.json',ROOT/'.cache'/f'{prefix}metadata.json']]
            csv_path=out/f'{key}_predictions.csv';manifest=out/f'{key}_source.json'
            if args.reuse_predictions and csv_path.exists():
                assert json.loads(manifest.read_text())==source
                frame=pd.read_csv(csv_path)
            else:
                ds=utils.CachedPTDataset(path)
                model,kind=utils.load_analysis_model(checkpoint,device,{split:ds})
                frame=physical_predictions(utils.predict_dataset(ds,model,device,kind,batch_size=128))
                frame.to_csv(csv_path,index=False)
                manifest.write_text(json.dumps(source,indent=2),encoding='utf-8')
                del ds,model;model=None;gc.collect()
                if device.type=='cuda':torch.cuda.empty_cache()
            evaluations[key]=frame
            overall.append(dict(dataset=key,group='ALL',**metrics(frame)))
            all_bins.extend(stratify(frame,key))
            for group,g in frame.groupby('group'):
                overall.append(dict(dataset=key,group=group,**metrics(g)))
                for row in stratify(g,key):
                    group_bins.append(dict(group=group,**row))
            # 阈值只取各模型去重训练集；不同模型的分位箱物理范围不同。
            q=np.quantile(abs(train.C),[.5,.9,.99])
            edges=[-np.inf,*q,np.inf]
            for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                g=frame[(abs(frame.C)>lo)&(abs(frame.C)<=hi)]
                if len(g):quantile_rows.append(dict(dataset=key,train_quantile_bin=['0-50','50-90','90-99','99-100'][k],lower=lo,upper=hi,**metrics(g)))
            print(f'[inference] {key}: n={len(frame)} MAE={metrics(frame)["mae"]:.5g}',flush=True)

    # 重现已保存预测；raw CSV 的 double 时间用于连续轨迹绘图。
    old=pd.read_csv(ROOT/'Results'/MAIN/'raw_held_out_predictions.csv')
    saved=pd.read_csv(ROOT/'Results'/MAIN/'held_out_test_predictions.csv')
    # 历史 raw CSV 没有 UID；按组和时间排序后逐项核验，再关联 cache 身份。
    old=old.sort_values(['type','time']).reset_index(drop=True)
    saved=saved.sort_values(['type','time']).reset_index(drop=True)
    assert len(old)==len(saved) and np.array_equal(old.type,saved.type)
    assert not old.duplicated(['type','time']).any()
    assert np.allclose(old.time,saved.time,rtol=0,atol=1e-5)
    for field in ['sfr_true','sfr_pred','sfr_macer_baseline','closure_true_physical']:
        assert np.allclose(old[field],saved[field],rtol=2e-6,atol=1e-5)
    for field in ['sample_uid','parent_hr_uid']:
        old[field]=saved[field].to_numpy()
    expected=physical_predictions(saved).set_index('uid')
    actual=evaluations['Exp2_LOO_out'].set_index('uid').loc[expected.index]
    assert set(expected.index)==set(actual.index)
    max_error=float(abs(expected.Yhat-actual.Yhat).max())
    # 历史运行环境未完整归档；记录逐点数值差异，并要求远小于模型科学误差。
    assert np.allclose(expected.Yhat,actual.Yhat,rtol=2.5e-3,atol=2e-4),max_error
    old_mae=float(np.mean(abs(expected.Yhat-expected.Y)))
    new_mae=float(np.mean(abs(actual.Yhat-actual.Y)))
    assert abs(new_mae-old_mae)/old_mae<1e-3
    held=physical_predictions(old).sort_values('time').reset_index(drop=True)
    assert set(held.uid)==set(actual.index)
    assert np.allclose(held.set_index('uid').Y,actual.Y,rtol=2e-6,atol=1e-5)
    audit['max_prediction_reproduction_error_msunyr']=max_error
    audit['max_prediction_relative_difference']=float(np.max(abs(expected.Yhat-actual.Yhat)/(abs(expected.Yhat)+1e-5)))
    audit['mae_relative_reproduction_difference']=abs(new_mae-old_mae)/old_mae
    audit['numerical_reproduction_note']='不是逐位复现；当前 checkpoint 推理与旧 CSV 的微小差异已量化，旧 held-out 图和阶段统计保留原 CSV，跨模型表使用新推理。'
    audit['heldout_used_for_plots']='archived raw_held_out_predictions.csv, verified against checkpoint inference'
    paper_time=held.time-.5625
    phase_rows=[]
    for label,mask in [('all',np.ones(len(held),dtype=bool)),('pre_t_lt_6',paper_time<6),('burst_t_ge_6',paper_time>=6),('early_t_lt_5.5',paper_time<5.5),('transition_5.5_6.5',(paper_time>=5.5)&(paper_time<6.5)),('sustained_t_ge_6.5',paper_time>=6.5)]:
        f=held[mask]
        phase_rows.append(dict(phase=label,**metrics(f),**integral(f)))
    audit['heldout_distribution']=distribution(held)
    audit['heldout_closure_peak']=dict(time_production_gyr=float(paper_time.iloc[held.C.argmax()]),C_max=float(held.C.max()),Chat_at_true_C_peak=float(held.Chat.iloc[held.C.argmax()]),Chat_max=float(held.Chat.max()),Chat_peak_time_production_gyr=float(paper_time.iloc[held.Chat.argmax()]),C_min=float(held.C.min()),largest_adjacent_C_change=float(abs(np.diff(held.C)).max()),cadence_gyr=float(np.median(np.diff(held.time))))
    audit['bootstrap']=[block_bootstrap(held,w) for w in [.25,.5,1.0]]
    # 重现旧 burst 指标，并检查后处理阈值与平滑尺度的敏感性。
    from VisualScripts.ResultAnalysis.analyze_exp2_burst import _timing_analysis
    timing_input=held.rename(columns={'Y':'sfr_true_msun_per_yr','Yhat':'sfr_pred_msun_per_yr'}).copy()
    timing_input['paper_time_gyr']=paper_time
    timing_rows=[]
    for smooth in [.25,.5,1.0]:
        for multiplier in [2.,3.,5.]:
            timing,_=_timing_analysis(timing_input,smooth,multiplier,.25)
            timing_rows.append(dict(smooth_width_gyr=smooth,threshold_multiplier=multiplier,
                threshold=timing['definition']['threshold_sfr_msun_per_yr'],
                true_start=timing['true_primary_start_gyr'],predicted_start=timing['predicted_primary_start_gyr'],
                start_error=timing['start_time_error_pred_minus_true_gyr'],
                true_primary_end=timing['true_primary_end_gyr'],predicted_primary_end=timing['predicted_primary_end_gyr'],
                true_right_censored=timing['true_end_is_right_censored'],
                recall=timing['paper_window_burst_recall']))
            if smooth==.5 and multiplier==3:
                old_timing=json.loads((ROOT/'Results'/MAIN/'burst_analysis/burst_summary.json').read_text())['timing_metrics']
                for key in ['true_primary_start_gyr','predicted_primary_start_gyr','predicted_primary_end_gyr','paper_window_burst_recall']:
                    assert np.isclose(timing[key],old_timing[key])
    audit['existing_burst_timing_reproduced']=True
    old_burst=pd.read_csv(ROOT/'Results'/MAIN/'burst_analysis/burst_metrics.csv').iloc[0]
    burst=held[paper_time>=6]
    assert np.isclose(metrics(burst)['mae'],old_burst.mae_msun_per_yr,rtol=1e-6)
    assert np.isclose(integral(burst)['mass_bias_fraction'],old_burst.integrated_relative_error,rtol=1e-6)
    audit['existing_burst_metrics_reproduced']=True
    audit['heldout_tail_above_training_max_n']=int((abs(held.C)>abs(train_frames['Exp2_12x20_v6'].C).max()).sum())
    audit['heldout_tail_learnability_warning']='分层误差是描述性关联；幅值、时间阶段和输入域偏移相互混杂。'
    audit['historical_contract_warning']='Exp2/Exp1 v6 的三个 Fiducial 运行早于时间与日志上下文修复；Model0.75 时间无此问题。'
    audit['bootstrap_limitation']='单轨迹固定时间块重采样；不是多独立 burst 或多随机种子的置信区间。'
    pd.DataFrame(distribution_rows).to_csv(out/'cache_distributions.csv',index=False)
    bins=pd.DataFrame(all_bins);bins.to_csv(out/'closure_bins.csv',index=False)
    pd.DataFrame(group_bins).to_csv(out/'closure_bins_by_group.csv',index=False)
    pd.DataFrame(overall).to_csv(out/'overall_metrics.csv',index=False)
    pd.DataFrame(quantile_rows).to_csv(out/'train_quantile_bins.csv',index=False)
    pd.DataFrame(phase_rows).to_csv(out/'heldout_phases.csv',index=False)
    pd.DataFrame(timing_rows).to_csv(out/'burst_definition_sensitivity.csv',index=False)
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    make_plots(out,held,bins,pd.DataFrame(distribution_rows),train_frames)
    print('[done]',out,flush=True)


if __name__=='__main__':
    main()
