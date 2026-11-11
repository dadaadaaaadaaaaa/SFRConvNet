"""比较四折冷致密近邻诊断，保留上调/下调分层和时间变化。"""
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
from VisualScripts.ResultAnalysis.compare_closure_reference import relation

SOURCES=[
    ('VisualScripts/DataAnalysis/outputs/cold_dense_closure_20261111',
     'VisualScripts/ResultAnalysis/outputs/closure_comparison_20261111_verified'),
    ('VisualScripts/DataAnalysis/outputs/fiducial4_cold_dense_20261111',
     'VisualScripts/ResultAnalysis/outputs/fiducial4_closure_20261111'),
]


def summarize(a,floor):
    """逐帧先计算贡献再求均值，不用总体中位数代替物理乘积。"""
    row=relation(a,floor)
    row.update(median_relative=((a.Chat-a.C)/a.Y.replace(0,np.nan)).median(),
        f_median=a.inner30_f.median(),ref_f_median=a.ref_inner30_f.median(),
        median_f_difference=(a.inner30_f-a.ref_inner30_f).median(),
        fraction_f_larger=(a.inner30_f>a.ref_inner30_f).mean(),
        q_median=a.inner30_q.median(),ref_q_median=a.ref_inner30_q.median())
    for part in ['mass','fraction','density']:
        row[part+'_contribution']=a['inner30_part_'+part].mean()
    return row


def matched_role_metrics(held,known):
    """只比较相同UID在留出与已知运行测试两种角色下的预测。"""
    a=held.merge(known,on='uid',suffixes=('_held','_known'),validate='one_to_one')
    if not len(a):raise ValueError('没有共同测试帧')
    for col in ['Y','B','time']:
        if not np.allclose(a[col+'_held'],a[col+'_known'],rtol=1e-9,atol=1e-9):
            raise ValueError('相同UID的物理量不一致')
    if not np.array_equal(a.parent_held,a.parent_known):raise ValueError('相同UID的parent不一致')
    row=dict(N=len(a))
    for role in ['held','known']:
        e=a['P_'+role]-a['Y_'+role]
        row[role+'_MAE']=abs(e).mean();row[role+'_Bias']=e.mean()
    return row


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    frames=[];floors={};hashes={}
    for gas,closure in SOURCES:
        gp,cp=ROOT/gas,ROOT/closure
        method=json.loads((cp/'method.json').read_text(encoding='utf-8'))
        floors.update({k:v['ratio_floor'] for k,v in method['sources'].items()})
        f=pd.read_csv(gp/'pairs.csv.gz')
        assert set(f.method)=={'b10'}
        frames.append(f)
        for path in [gp/'pairs.csv.gz',gp/'method.json',cp/'method.json']:
            hashes[str(path.relative_to(ROOT))]=file_hash(path)
    frame=pd.concat(frames,ignore_index=True)
    assert not frame[['case','split','uid']].duplicated().any()
    frame['delta']=frame.C-frame.Cref
    frame['response']=frame.Chat-frame.Cref
    frame['error']=frame.Chat-frame.C
    frame['f_difference']=frame.inner30_f-frame.ref_inner30_f
    good=np.isfinite(frame.Cref)
    assert np.allclose(frame.loc[good,'error'],(frame.response-frame.delta)[good],rtol=1e-10,atol=1e-10)
    rows=[];times=[];refs=[]
    for case,case_frame in frame.groupby('case'):
        floor=floors[case]
        for split,b in case_frame.groupby('split'):
            for scope,mask in [('matched',np.isfinite(b.Cref)),('close',b.distance<=.5)]:
                a=b[mask]
                masks={'all':np.ones(len(a),bool),'C_up':a.delta>floor,'C_down':a.delta<-floor,
                       'f_up_5pp':a.f_difference>.05,'f_down_5pp':a.f_difference<-.05,
                       'f_within_5pp':abs(a.f_difference)<=.05}
                for name,chosen in masks.items():
                    z=a[chosen]
                    if len(z):rows.append(dict(case=case,split=split,scope=scope,subset=name,**summarize(z,floor)))
                for group,z in a.groupby('ref_group'):
                    refs.append(dict(case=case,split=split,scope=scope,reference_group=group,N=len(z)))
        a=case_frame[case_frame.split=='out'].copy()
        a['time_bin']=np.floor(a.time/.5).astype(int)
        for i,z in a.groupby('time_bin'):
            times.append(dict(case=case,start_gyr=i*.5,end_gyr=(i+1)*.5,
                actual_start=z.time.min(),actual_end=z.time.max(),**summarize(z,floor)))
    pd.DataFrame(rows).to_csv(output/'conditional_metrics.csv',index=False)
    pd.DataFrame(times).to_csv(output/'time_metrics.csv',index=False)
    pd.DataFrame(refs).to_csv(output/'reference_groups.csv',index=False)
    roles=[]
    for group,held_case,known_case in [('disk_galaxy_fiducial_7','fiducial7','fiducial4'),
                                      ('disk_galaxy_fiducial_4','fiducial4','fiducial7')]:
        held=frame[(frame.case==held_case)&(frame.split=='out')]
        known=frame[(frame.case==known_case)&(frame.split=='test')&(frame.group==group)]
        roles.append(dict(group=group,held_case=held_case,known_case=known_case,**matched_role_metrics(held,known)))
    pd.DataFrame(roles).to_csv(output/'same_frames_roles.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for row,case in enumerate(['fiducial4','fiducial7']):
        a=frame[(frame.case==case)&(frame.split=='out')&np.isfinite(frame.Cref)].copy()
        ax=axes[row,0]
        scatter=ax.scatter(a.ref_inner30_f,a.inner30_f,c=a.error/a.Y,s=4,alpha=.5,
                           cmap='coolwarm',vmin=-.5,vmax=.5,rasterized=True)
        ax.plot([0,1],[0,1],'k--',lw=1)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Train reference cold-dense mass fraction',
               ylabel='Held-out cold-dense mass fraction',title=case+': r < 30 kpc')
        fig.colorbar(scatter,ax=ax,label='(Prediction - truth) / truth')
        ax=axes[row,1]
        ax.scatter(a.delta,a.response,s=4,alpha=.25,rasterized=True)
        low=min(a.delta.min(),a.response.min());high=max(a.delta.max(),a.response.max())
        ax.plot([low,high],[low,high],'k--',lw=1)
        ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7)
        ax.set(xlabel='C - Cref [Msun/yr]',ylabel='Chat - Cref [Msun/yr]',title=case+': closure adjustment')
        a=a.sort_values('time');ax=axes[row,2]
        gaps=np.flatnonzero(np.diff(a.time)>5*np.median(np.diff(a.time)))+1
        t=np.insert(a.time.to_numpy(),gaps,np.nan)
        for label,col in [('Truth','Y'),('Prediction','P')]:
            ax.plot(t,np.insert(a[col].to_numpy(),gaps,np.nan),lw=.7,label=label)
        ax.set(xlabel='Snapshot absolute time [Gyr]',ylabel='SFR [Msun/yr]',title=case+': matched out trajectory')
        ax.legend()
    fig.savefig(output/'fiducial4_vs_fiducial7.png',dpi=170)
    plt.close(fig)
    hashes[str(Path(__file__).relative_to(ROOT))]=file_hash(Path(__file__))
    (output/'method.json').write_text(json.dumps(dict(sha256=hashes,reference_method='b10',
        units='SFR: Msun/yr; fractions: mass fractions',fraction_bins='inner30 query minus reference: <-0.05, [-0.05,0.05], >0.05',
        closure_direction='delta > ratio_floor or delta < -ratio_floor; training-only ratio_floor from reference method',
        time_bins='fixed 0.5 Gyr in snapshot absolute time; descriptive, not event boundaries',
        limitations='paired descriptive analysis, nonzero input distance, no causal intervention or seed uncertainty'),
        ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)
