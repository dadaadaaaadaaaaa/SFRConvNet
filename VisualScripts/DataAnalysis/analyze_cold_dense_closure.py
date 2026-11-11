"""从现有 HR 气体场分解冷致密比例、质量和密度矩，并配对 Exp2 闭合。"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from VisualScripts.DataAnalysis.analyze_disk_sf_timescales import RUNS, LABELS, block_view
from VisualScripts.ResultAnalysis.compare_fiducial_domains import CASES
from VisualScripts.ResultAnalysis.analyze_sfr_unet import file_hash

K=.025*.01*np.sqrt(32*112.3/(3*np.pi))
OLD=ROOT/'VisualScripts/ResultAnalysis/outputs/closure_comparison_20261111_verified'
TIMES=ROOT/'VisualScripts/DataAnalysis/outputs/sf_timescales_20261111'


def moments(rho,energy,volume,mask):
    """权威体积下计算质量比例；q是成星气体的质量加权sqrt(rho)。"""
    eligible=(1.02*rho>1)&(93*energy/np.maximum(rho,1e-300)<4e4)&mask
    mass=rho*volume
    m=mass[mask].sum(); cd=mass[eligible].sum()
    w=(mass[eligible]*np.sqrt(rho[eligible])).sum()
    return dict(M=m,f=cd/m if m else np.nan,q=w/cd if cd else 0.,
                Sff=K*w,Mcd=cd,cell_fraction=eligible.sum()/mask.sum())


def shapley(a,b):
    """三因子乘积的对称代数分解；不把交换因素视为可实现的物理实验。"""
    result=[]
    for i in range(3):
        j,k=[v for v in range(3) if v!=i]
        result.append(K*(a[i]-b[i])*(b[j]*b[k]/3+(a[j]*b[k]+b[j]*a[k])/6+a[j]*a[k]/3))
    return result


def scan(out,smoke):
    frames=[]
    for name in RUNS:
        prior=pd.read_parquet(TIMES/f'{name}_snapshots.parquet')
        prior=prior[prior['mode']=='data_compatible']
        if smoke:
            prior=prior.iloc[np.unique(np.linspace(0,len(prior)-1,3,dtype=int))]
        rows=[]
        for n,(_,old) in enumerate(prior.iterrows()):
            path=ROOT/'data'/f'disk_galaxy_{name}'/old.fine_dir/old.file
            with h5py.File(path,'r') as h:
                meta=json.loads(h['metadata/contract_json'][()])
                assert meta['identity']['parent_hr_uid']==old.parent_hr_uid
                assert np.isclose(float(h.attrs['time']),old.time_gyr,atol=1e-10,rtol=0)
                y=float(h['sfr_supervision/canonical/Y_alpha'][()])*.025
                assert np.isclose(y,old.true_sfr,atol=1e-9,rtol=1e-9)
                rho=np.asarray(h['snapshot/density']);energy=np.asarray(h['snapshot/gas_energy'])
                v=np.asarray(h['geometry/cell_volume']);r=np.asarray(h['coordinates/axis_0'])
                assert rho.shape==v.shape==(72,280) and (v>0).all()
                assert np.isfinite(rho).all() and (rho>=0).all() and np.isfinite(energy).all()
                row=dict(group=LABELS[name],parent=old.parent_hr_uid,time=old.time_gyr,Y=y,
                         source=str(path.relative_to(ROOT)))
                for tag,mask in [('all',np.ones(rho.shape,bool)),('inner30',np.broadcast_to(r[None,:]<30,rho.shape))]:
                    row.update({tag+'_'+k:value for k,value in moments(rho,energy,v,mask).items()})
                bv=block_view(v,(12,20)).sum((2,3))
                br=block_view(rho*v,(12,20)).sum((2,3))/bv
                be=block_view(energy*v,(12,20)).sum((2,3))/bv
                assert np.isclose((br*bv).sum(),(rho*v).sum(),rtol=1e-12)
                lr=moments(br,be,bv,np.ones(br.shape,bool))
                row.update({'lr_'+k:value for k,value in lr.items()})
                elig=(rho*1.02>1)&(93*energy/np.maximum(rho,1e-300)<4e4)
                lr_elig=(br*1.02>1)&(93*be/np.maximum(br,1e-300)<4e4)
                hr_cd_block=block_view(rho*v*elig,(12,20)).sum((2,3))
                row['hidden_cd_fraction']=hr_cd_block[~lr_elig].sum()/max(row['all_Mcd'],1e-300)
                assert np.isclose(row['all_Sff'],K*row['all_M']*row['all_f']*row['all_q'],rtol=1e-12,atol=1e-12)
                rows.append(row)
            if (n+1)%1000==0: print(name,n+1,len(prior),flush=True)
        f=pd.DataFrame(rows);f.to_parquet(out/f'{name}.parquet',index=False);frames.append(f)
        print('完成',name,len(f),flush=True)
    return pd.concat(frames,ignore_index=True)


def analyze(f,out,reference_dir=OLD,cases=None,moments_dir=None):
    reference_dir=reference_dir.resolve()
    if moments_dir:moments_dir=moments_dir.resolve()
    assert len(f)==26132 and not f.parent.duplicated().any()
    features=['all_f','inner30_f','all_M','all_Mcd','all_q','inner30_q','lr_f','hidden_cd_fraction']
    rows=[]
    for group,a in f.groupby('group'):
        for col in features:
            rows.append(dict(group=group,feature=col,N=len(a),median=a[col].median(),mean=a[col].mean(),
                             p01=a[col].quantile(.01),p99=a[col].quantile(.99),min=a[col].min(),max=a[col].max()))
    pd.DataFrame(rows).to_csv(out/'galaxy_summary.csv',index=False)
    errors=f.assign(relative=(f.all_Sff-f.Y)/f.Y.replace(0,np.nan),inner_share=f.inner30_Sff/f.all_Sff.replace(0,np.nan))
    errors.groupby('group').agg(N=('Y','size'),median_relative=('relative','median'),
        p99_absolute=('relative',lambda v:abs(v).quantile(.99)),median_inner_share=('inner_share','median')).to_csv(out/'ff_validation.csv')
    pred=pd.read_csv(reference_dir/'predictions_and_references.csv.gz')
    coverage=[];pairs=[];decomp=[];errors_by_fraction=[]
    models={'fiducial':'UNet_original42','model075':'UNet','fiducial7':'UNet','fiducial4':'UNet'}
    selected=cases or ['fiducial','model075','fiducial7']
    for case in selected:
        model=models[case]
        payload=torch.load(ROOT/'.cache'/f"{CASES[case]['prefix']}train.pt",map_location='cpu',weights_only=False,mmap=True)
        ident=payload['sample_identity']
        train=pd.DataFrame(dict(uid=ident['sample_uid'],parent=ident['parent_hr_uid'])).drop_duplicates('uid')
        train=train.merge(f,on='parent',validate='one_to_one')
        assert len(train)==len(set(ident['sample_uid']))
        a=pred[(pred.case==case)&(pred.model==model)&(pred.method=='b10')].copy()
        truth=f.set_index('parent').loc[a.parent]
        assert np.allclose(a.Y,truth.Y.to_numpy(),atol=1e-8,rtol=1e-8)
        assert np.allclose(a.time,truth.time.to_numpy(),atol=1e-9,rtol=0)
        a=a.merge(f.drop(columns=['group','time','Y']),on='parent',validate='many_to_one')
        assert len(a)==len(pred[(pred.case==case)&(pred.model==model)&(pred.method=='b10')])
        ref=train[['uid',*features,'inner30_M','inner30_Mcd','inner30_Sff','all_Sff']].rename(columns=lambda c:'ref_'+c)
        a=a.merge(ref,left_on='ref_uid',right_on='ref_uid',how='left',validate='many_to_one')
        for col in features:
            lo,hi=train[col].min(),train[col].max()
            lq,hq=train[col].quantile([.01,.99])
            z=a[a.split=='out']
            coverage.append(dict(case=case,feature=col,N=len(z),train_min=lo,train_max=hi,
                train_median=train[col].median(),out_median=z[col].median(),
                out_low=(z[col]<lo).mean(),out_high=(z[col]>hi).mean(),out_central_tail=((z[col]<lq)|(z[col]>hq)).mean()))
            if col in ['all_f','inner30_f']:
                for category,mask in [('inside',z[col].between(lo,hi)),('outside',~z[col].between(lo,hi)),
                                      ('high_SFR_inside',(z.Y>train.Y.max())&z[col].between(lo,hi)),
                                      ('high_SFR_outside',(z.Y>train.Y.max())&~z[col].between(lo,hi))]:
                    zz=z[mask]
                    if len(zz):errors_by_fraction.append(dict(case=case,feature=col,category=category,N=len(zz),
                        Bias=(zz.Chat-zz.C).mean(),MAE=abs(zz.Chat-zz.C).mean(),
                        median_relative=((zz.Chat-zz.C)/zz.Y.replace(0,np.nan)).median()))
        for region in ['all','inner30']:
            x=[a[region+'_'+c].to_numpy() for c in ['M','f','q']]
            y=[a['ref_'+region+'_'+c].to_numpy() for c in ['M','f','q']]
            parts=shapley(x,y)
            for name,part in zip(['mass','fraction','density'],parts):a[region+'_part_'+name]=part
            good=np.isfinite(a['ref_'+region+'_Sff'])
            assert np.allclose(sum(parts)[good],(a[region+'_Sff']-a['ref_'+region+'_Sff'])[good],rtol=1e-9,atol=1e-9)
        for split in ['test','out']:
            for scope,mask in [('matched',a.distance.notna()&np.isfinite(a.distance)),('close',a.distance<=.5)]:
                z=a[(a.split==split)&mask].copy()
                if not len(z):continue
                for region in ['all','inner30']:
                    row=dict(case=case,split=split,scope=scope,region=region,N=len(z),
                        C_gap=(z.C-z.Cref).mean(),prediction_bias=(z.Chat-z.C).mean(),
                        baseline_gap=(z.B-z.B_ref).mean(),
                        Y_gap=(z.Y-(z.Cref+z.B_ref)).mean(),
                        f_median=z[region+'_f'].median(),ref_f_median=z['ref_'+region+'_f'].median(),
                        f_larger_fraction=(z[region+'_f']>z['ref_'+region+'_f']).mean(),
                        q_median=z[region+'_q'].median(),ref_q_median=z['ref_'+region+'_q'].median())
                    for part in ['mass','fraction','density']:row[part+'_contribution']=z[region+'_part_'+part].mean()
                    row['ff_delta']=(z[region+'_Sff']-z['ref_'+region+'_Sff']).mean()
                    row['residual_Y']=row['Y_gap']-row['ff_delta']
                    decomp.append(row)
        pairs.append(a)
    pd.DataFrame(coverage).to_csv(out/'fold_coverage.csv',index=False)
    pd.DataFrame(decomp).to_csv(out/'paired_decomposition.csv',index=False)
    pd.DataFrame(errors_by_fraction).to_csv(out/'errors_by_fraction.csv',index=False)
    pd.concat(pairs).to_csv(out/'pairs.csv.gz',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for group,a in f.groupby('group'):
        for ax,col in zip(axes,['all_f','inner30_f','all_q']):
            x=np.sort(a[col]);ax.plot(x,np.arange(1,len(x)+1)/len(x),label=group)
            ax.set(xlabel=col,ylabel='CDF');ax.set_xscale('symlog',linthresh=.001)
    axes[-1].legend();fig.savefig(out/'cold_dense_distributions.png',dpi=170);plt.close(fig)
    source_paths=[Path(__file__),reference_dir/'predictions_and_references.csv.gz',reference_dir/'method.json']
    source_paths.extend(TIMES/f'{name}_snapshots.parquet' for name in RUNS)
    if moments_dir:source_paths.extend(moments_dir/f'{name}.parquet' for name in RUNS)
    previous=json.loads((reference_dir/'method.json').read_text(encoding='utf-8'))
    for case in selected:
        path=ROOT/'.cache'/f"{CASES[case]['prefix']}train.pt"
        assert file_hash(path)==previous['sha256'][str(path.relative_to(ROOT))]
        source_paths.append(path)
    (out/'method.json').write_text(json.dumps(dict(
        sources={str(p.relative_to(ROOT)):file_hash(p) for p in source_paths},
        N=len(f),threshold='T<4e4 K; 1.02*rho>1',regions=['whole domain','r<30 kpc'],
        units='mass: code mass (2.5e7 Msun); density moment: sqrt(code density); Sff: Msun/yr',
        decomposition='Symmetric three-factor product Shapley; descriptive, not causal',
        checks='15-frame smoke; positive volumes; mass conservation; source parent/time/Y; prediction Y/time; exact product and decomposition; actual train SHA256',
        limitations='raw HR cell SFR unavailable; Y remains true aggregate; Sff is diagnostic; zero eligible mass has q=0 by convention; changing aperture changes factor contributions'),
        ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--reuse',action='store_true')
    p.add_argument('--reference-dir',type=Path,default=OLD)
    p.add_argument('--cases',nargs='+',choices=CASES)
    p.add_argument('--moments-dir',type=Path)
    args=p.parse_args()
    if args.moments_dir and (args.reuse or args.smoke):p.error('--moments-dir不能与--reuse或--smoke同时使用')
    if args.reuse or args.moments_dir:
        if args.moments_dir:
            args.output_dir.mkdir(parents=True,exist_ok=False)
            (args.output_dir/'.gitignore').write_text('*.parquet\n*.csv.gz\n',encoding='utf-8')
        source=args.moments_dir or args.output_dir
        frame=pd.concat([pd.read_parquet(source/f'{n}.parquet') for n in RUNS],ignore_index=True)
    else:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        (args.output_dir/'.gitignore').write_text('*.parquet\n*.csv.gz\n',encoding='utf-8')
        frame=scan(args.output_dir,args.smoke)
    if not args.smoke:analyze(frame,args.output_dir,args.reference_dir,args.cases,args.moments_dir)
