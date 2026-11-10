"""统计 Exp2 的真实闭合、训练近邻闭合与最终预测闭合，保留所有异常值。"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from VisualScripts.ResultAnalysis.compare_fiducial_domains import CASES, validate_alignment
from VisualScripts.ResultAnalysis.analyze_sfr_unet import file_hash, load_predictions
from VisualScripts.DataAnalysis.check_disk_lr_generalization import anchors, SHORT_MAP


def describe(values):
    """保留负闭合、零和极端值；统计的是 snapshot 分布而非时间积分。"""
    v = np.asarray(values, dtype=float)
    if not len(v) or not np.isfinite(v).all():
        raise ValueError('分布必须非空且有限。')
    q = np.quantile(v, [0,.01,.05,.5,.95,.99,1])
    return dict(N=len(v), mean=v.mean(), std=v.std(), negative_fraction=(v<0).mean(),
                zero_fraction=(v==0).mean(), **dict(zip(['min','p01','p05','median','p95','p99','max'],q)))


def relation(a, ratio_floor):
    """误差分解只是恒等式；近邻拉回系数是描述量，不解释为因果系数。"""
    c, p = a.C.to_numpy(), a.Chat.to_numpy()
    e = p-c
    row = dict(N=len(a), C_mean=c.mean(), Chat_mean=p.mean(),
        MAE=abs(e).mean(), RMSE=np.sqrt(np.mean(e*e)), Bias=e.mean(),
        Baseline_MAE=abs(c).mean(), UnderFraction=(e<0).mean(),
        C_min=c.min(), C_max=c.max(), Chat_min=p.min(), Chat_max=p.max(),
        Recovery=p.sum()/c.sum() if abs(c.sum())>1e-10 else np.nan)
    good = np.isfinite(a.Cref.to_numpy())
    row['matched_N'] = int(good.sum())
    if not good.any():
        return row
    r = a.Cref.to_numpy()[good]
    d = c[good]-r
    h = p[good]-r
    stable = abs(d)>ratio_floor
    row.update(Cref_mean=r.mean(), C_matched_mean=c[good].mean(), Chat_matched_mean=p[good].mean(),
        Bias_matched=e[good].mean(), Delta_mean=d.mean(), R_mean=h.mean(),
        Reference_MAE=abs(d).mean(), Reference_Bias=(-d).mean(),
        Delta_positive_fraction=(d>0).mean(), ratio_N=int(stable.sum()),
        Alpha_median=np.median(h[stable]/d[stable]) if stable.any() else np.nan,
        Closer_to_reference=(abs(h[stable])<abs(e[good][stable])).mean() if stable.any() else np.nan,
        Through_origin_slope=float(d@h/(d@d)) if d@d>1e-12 else np.nan)
    if not np.allclose(e[good], h-d, rtol=1e-10, atol=1e-10):
        raise ValueError('闭合误差分解失败。')
    return row


def nearest_reference(train, tx, query, qx, *, methods=None, batch_size=128):
    """参考仅为 train；baseline约束与时间排除均不使用查询的 Y 或 C。"""
    methods = methods or {'x':None, 'b05':(.05,.10), 'b10':(.10,.25), 'b20':(.20,.50)}
    selected = np.concatenate([anchors(np.flatnonzero(train.group==g),train,192)
                               for g in train.group.unique()])
    mu = tx[selected].mean(axis=0,keepdims=True)
    sd = tx[selected].std(axis=(0,2,3),keepdims=True).clip(1e-5)
    rt = torch.from_numpy(((tx-mu)/sd).reshape(len(tx),-1).astype(np.float32))
    rn = (rt*rt).sum(1)
    output = {name:[] for name in methods}
    for start in range(0,len(query),batch_size):
        a = query.iloc[start:start+batch_size]
        qt = torch.from_numpy(((qx[start:start+len(a)]-mu)/sd).reshape(len(a),-1).astype(np.float32))
        d = ((qt*qt).sum(1)[:,None]+rn[None]-2*qt@rt.T).clamp_min(0)/rt.shape[1]
        exclude = ((a.group.to_numpy()[:,None]==train.group.to_numpy()[None]) &
                   (abs(a.time.to_numpy()[:,None]-train.time.to_numpy()[None])<.25))
        d.masked_fill_(torch.from_numpy(exclude),float('inf'))
        db = abs(a.B.to_numpy()[:,None]-train.B.to_numpy()[None])
        for name, setting in methods.items():
            z = d
            if setting is not None:
                relative,floor = setting
                z = d.masked_fill(torch.from_numpy(db>np.maximum(floor,relative*abs(a.B.to_numpy()))[:,None]),float('inf'))
            values, ids = torch.topk(z,k=min(5,len(train)),dim=1,largest=False)
            distances, index = np.sqrt(values.numpy()), ids.numpy()
            valid = np.isfinite(distances[:,0])
            ref = train.iloc[index[:,0]].reset_index(drop=True)
            c5 = train.C.to_numpy()[index].astype(float)
            c5[~np.isfinite(distances)] = np.nan
            med = np.full(len(a),np.nan)
            med[valid] = np.nanmedian(c5[valid],axis=1)
            result = pd.DataFrame(dict(uid=a.uid.to_numpy(), method=name,
                distance=distances[:,0], Cref=np.where(valid,ref.C,np.nan),
                Cref5=med, B_ref=np.where(valid,ref.B,np.nan),
                ref_uid=np.where(valid,ref.uid,''), ref_group=np.where(valid,ref.group,''),
                ref_time=np.where(valid,ref.time,np.nan), k_valid=np.isfinite(distances).sum(1)))
            output[name].append(result)
    return pd.concat([pd.concat(v,ignore_index=True) for v in output.values()],ignore_index=True),len(selected)


def cache_frame(prefix, split, record, inputs=True):
    path=ROOT/'.cache'/f'{prefix}{split}.pt'
    record(path)
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    t=payload['tensors']
    f=pd.DataFrame(dict(uid=payload['sample_identity']['sample_uid'],parent=payload['sample_identity']['parent_hr_uid'],
        group=np.asarray(t['groups']).reshape(-1),time=np.asarray(t['time']).reshape(-1),
        Y=np.asarray(t['target_sfr_physical']).reshape(-1)*.025,
        B=np.asarray(t['macer_sfr_baseline']).reshape(-1)*.025,
        C=np.asarray(t['closure_true_physical']).reshape(-1)*.025))
    if not np.allclose(f.Y,f.B+f.C,atol=1e-10,rtol=1e-10):
        raise ValueError('cache 不满足 Y=B+C。')
    if f.groupby('uid')[['parent','group','time','Y','B','C']].nunique().to_numpy().max()!=1:
        raise ValueError('重复 UID 的物理量不一致。')
    keep=~f.uid.duplicated()
    weights=f.uid.value_counts()
    f=f.loc[keep].reset_index(drop=True)
    f['weight']=f.uid.map(weights)
    x=np.asarray(t['x'])[keep].astype(np.float32) if inputs else None
    return f,x


def validate_manifest(manifest, frames):
    """只允许训练集内部重复采样；split 间 UID 和 parent 均隔离。"""
    for split,f in frames.items():
        for col,key in [('uid','sample_uid'),('parent','parent_hr_uid')]:
            if set(f[col])!=set(manifest[split][key]):
                raise ValueError(f'{split} 的 {key} 不匹配。')
    for col in ['uid','parent']:
        for i,left in enumerate(frames):
            for right in list(frames)[i+1:]:
                if set(frames[left][col])&set(frames[right][col]):
                    raise ValueError('跨 split 身份交叉。')


def block_summary(a, width, repeats=1000):
    """分组时间块 bootstrap，只描述给定模型、给定轨迹的抽样稳定性。"""
    good=np.isfinite(a.Cref)
    b=a.loc[good].copy()
    b['block']=b.group+':'+np.floor(b.time/width).astype(int).astype(str)
    b['delta']=b.C-b.Cref
    b['r']=b.Chat-b.Cref
    b['error']=b.Chat-b.C
    b['dh']=b.delta*b.r
    b['dd']=b.delta*b.delta
    totals=b.groupby('block')[['delta','r','error','dh','dd']].sum()
    counts=b.groupby('block').size().to_numpy()
    if len(totals)<5:
        return dict(width=width,blocks=len(totals))
    draws=np.random.default_rng(42).integers(len(totals),size=(repeats,len(totals)))
    values=totals.to_numpy()[draws].sum(1)
    n=counts[draws].sum(1)
    row=dict(width=width,blocks=len(totals))
    for i,name in enumerate(['Delta','R','Bias']):
        row[name+'_low'],row[name+'_high']=np.quantile(values[:,i]/n,[.025,.975])
    mask=values[:,4]>1e-12
    if mask.any():
        row['slope_low'],row['slope_high']=np.quantile(values[mask,3]/values[mask,4],[.025,.975])
    return row


def run(output):
    if output.exists():
        raise FileExistsError('输出目录已存在，拒绝覆盖。')
    output.mkdir(parents=True)
    (output/'.gitignore').write_text('# 逐帧明细仅保存在本地。\n*.csv.gz\n',encoding='utf-8')
    torch.set_num_threads(4)
    hashes={}
    def record(path):
        key=str(path.relative_to(ROOT))
        if key not in hashes:
            hashes[key]=file_hash(path)
        return hashes[key]
    record(Path(__file__))
    for module in ['VisualScripts/ResultAnalysis/analyze_sfr_unet.py',
                   'VisualScripts/ResultAnalysis/compare_fiducial_domains.py',
                   'VisualScripts/DataAnalysis/check_disk_lr_generalization.py']:
        record(ROOT/module)
    specs={key:dict(value) for key,value in CASES.items()}
    historical=next(iter(specs['model075']['cnn'].values()))
    hpath=ROOT/'Results'/historical/'hyperparameters.json'
    hyper=json.loads(hpath.read_text(encoding='utf-8'))
    record(hpath)
    specs['model075']['cnn']={}
    specs['model075_cnn_history']=dict(prefix=hyper['data_exp'],held=specs['model075']['held'],
        title='Model0.75 historical CNN',cnn={'CNN53_historical':historical},unet={})
    distribution, training, predictions, comparisons, binned, cis, unions = [],[],[],[],[],[],[]
    plot_records={}
    sources={}
    for case,spec in specs.items():
        print(f'开始 {case}',flush=True)
        prefix=spec['prefix']
        frames,arrays={},{}
        for split in ['train','val','test','out']:
            frames[split],arrays[split]=cache_frame(prefix,split,record,inputs=split!='val')
            frames[split]['split']=split
            frames[split]['case']=case
        # 旧CNN没有UNet式manifest，仍核验其cache各split之间的身份隔离。
        validate_manifest({s:dict(sample_uid=list(f.uid),parent_hr_uid=list(f.parent))
                           for s,f in frames.items()},frames)
        train=frames['train']
        if set(frames['out'].group)!={spec['held']} or spec['held'] in set(train.group):
            raise ValueError('留出身份不匹配。')
        # 不混用折间标准化张量；全局真实闭合只合并物理量。
        if case!='model075_cnn_history':
            unions.extend(frames.values())
        for split,f in frames.items():
            for group,a in f.groupby('group'):
                distribution.append(dict(case=case,split=split,group=SHORT_MAP[group],**describe(a.C)))
        for group,a in [('ALL',train),*list(train.groupby('group'))]:
            for mode in ['unique','sampling_weighted']:
                c=a.C.to_numpy()
                if mode=='sampling_weighted':
                    c=np.repeat(c,a.weight.to_numpy().astype(int))
                training.append(dict(case=case,group=SHORT_MAP.get(group,group),weighting=mode,**describe(c)))
        query=pd.concat([frames['test'],frames['out']],ignore_index=True)
        qx=np.concatenate([arrays['test'],arrays['out']])
        refs,anchor_count=nearest_reference(train,arrays['train'],query,qx)
        refs.to_csv(output/f'{case}_references.csv.gz',index=False)
        print(f'  近邻完成：{len(query)} queries，{len(train)} train',flush=True)
        floor=max(.1,.05*np.median(abs(train.C)))
        sources[case]=dict(prefix=prefix,train_unique=len(train),train_weighted=int(train.weight.sum()),
                           anchor_count=anchor_count,ratio_floor=floor,models={})
        models=[]
        for label,folder in spec['unet'].items():
            root=ROOT/'Results'/folder
            summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
            contract=json.loads((root/'contract.json').read_text(encoding='utf-8'))
            manifest=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'))
            for name in ['summary.json','contract.json','split_manifest.json']:
                record(root/name)
            if summary['status']!='complete' or summary['hpo'] is not False:
                raise ValueError('模型并非完成的正式训练。')
            validate_manifest(manifest,frames)
            for split in frames:
                actual=record(ROOT/'.cache'/f'{prefix}{split}.pt')
                expected=(contract['data']['train_val'][split]['scalar_sha256'] if split in ['train','val']
                          else contract['evaluation_identities'][split]['scalar_sha256'])
                if actual!=expected:
                    raise ValueError('cache 摘要不匹配。')
            for item,expected in [('metadata',contract['data']['metadata_sha256']),('norm',contract['data']['norm_sha256'])]:
                if record(ROOT/'.cache'/f'{prefix}{item}.{ "json" if item=="metadata" else "npz"}')!=expected:
                    raise ValueError('metadata/归一化摘要不匹配。')
            for split in ['test','out']:
                path=root/f'{split}_predictions.pt'
                record(path)
                d=load_predictions(path,identity=manifest[split],expected_groups=set(frames[split].group),expected_loss=contract['loss'])
                p=pd.DataFrame(dict(uid=d['sample_uid'],parent=d['parent_hr_uid'],time=d['time'],
                    Y=d['target_total']*.025,B=d['baseline_total']*.025,P=d['prediction_total']*.025))
                p=validate_alignment(p,frames[split])
                p['Chat']=p.P-p.B
                order=pd.Index(d['sample_uid']).get_indexer(p.uid)
                if not np.allclose(p.Chat,d['closure_map'].sum((1,2,3))[order]*.025,rtol=1e-10,atol=1e-10):
                    raise ValueError('有效闭合与预测地图不一致。')
                models.append((label,split,p[['uid','P','Chat']]))
                del d
            sources[case]['models'][label]=dict(folder=folder,best_epoch=summary['best_epoch'])
        for label,folder in spec['cnn'].items():
            root=ROOT/'Results'/folder
            h=json.loads((root/'hyperparameters.json').read_text(encoding='utf-8'))
            if h['data_exp']!=prefix or h['objective_mode']!='signed_closure':
                raise ValueError('CNN 的训练参考族不匹配。')
            record(root/'hyperparameters.json')
            path=root/'held_out_test_predictions.csv'
            record(path)
            p=pd.read_csv(path).rename(columns={'sample_uid':'uid','parent_hr_uid':'parent','sfr_true':'Y','sfr_macer_baseline':'B','sfr_pred':'P'})
            for col in ['Y','B','P']:
                p[col]*=.025
            p=validate_alignment(p,frames['out'])
            p['Chat']=p.P-p.B
            # 旧CSV的B只有float32精度；在SFR量级核对组成，避免闭合相减消去放大相对误差。
            if not np.allclose(p.P,p.B+p.closure_pred_physical*.025,rtol=2e-7,atol=1e-7):
                raise ValueError('CNN 闭合恒等式失败。')
            # 所有模型使用高精度同源 cache 的 B，避免旧 CSV 的 float32 B 带入修正。
            p['Chat']=p.P-frames['out'].B.to_numpy()
            models.append((label,'out',p[['uid','P','Chat']]))
            sources[case]['models'][label]=dict(folder=folder,scope='仅导出的 out；历史CNN有独立训练参考')
        for label,split,p in models:
            a=frames[split].merge(p,on='uid',validate='one_to_one')
            if not np.allclose(a.P-a.Y,a.Chat-a.C,rtol=1e-10,atol=1e-10):
                raise ValueError('SFR 与闭合误差不一致。')
            for group,b in [('ALL',a),*list(a.groupby('group'))]:
                for kind,col in [('C','C'),('Chat','Chat')]:
                    distribution.append(dict(case=case,split=split,group=SHORT_MAP.get(group,group),model=label,quantity=kind,**describe(b[col])))
            # 全部预测单独统计；无参考样本仍进入预测误差与分布。
            raw=a.assign(Cref=np.nan)
            comparisons.append(dict(case=case,model=label,split=split,method='none',subset='all',group='ALL',**relation(raw,floor)))
            for method,r in refs.groupby('method'):
                b=a.merge(r,on='uid',validate='one_to_one')
                b['model']=label
                predictions.append(b)
                b['Delta']=b.C-b.Cref
                b['R']=b.Chat-b.Cref
                masks={'matched':np.isfinite(b.Cref),'d_le_05':np.isfinite(b.Cref)&(b.distance<=.5),
                       'C_inside':b.C.between(train.C.min(),train.C.max()),
                       'C_low':b.C<train.C.min(),'C_high':b.C>train.C.max(),
                       'Delta_positive':b.Delta>floor,'Delta_negative':b.Delta<-floor}
                for subset,mask in masks.items():
                    for group,z in [('ALL',b[mask]),*list(b[mask].groupby('group'))]:
                        if len(z):
                            comparisons.append(dict(case=case,model=label,split=split,method=method,subset=subset,
                                group=SHORT_MAP.get(group,group),**relation(z,floor)))
                if method in ['x','b10']:
                    for estimator in ['Cref','Cref5']:
                        z=b.copy()
                        z['Cref']=z[estimator]
                        comparisons.append(dict(case=case,model=label,split=split,method=method+'_'+estimator,
                            subset='matched',group='ALL',**relation(z[np.isfinite(z.Cref)],floor)))
                    if split=='out':
                        for width in [.25,.5]:
                            cis.append(dict(case=case,model=label,method=method,**block_summary(b,width)))
                    for interval,z in b.groupby(pd.cut(b.Delta,[-np.inf,-10,-5,-1,-.1,.1,1,5,10,np.inf]),observed=True):
                        binned.append(dict(case=case,model=label,split=split,method=method,interval=str(interval),
                            Delta_median=z.Delta.median(),R_median=z.R.median(),error_median=(z.Chat-z.C).median(),**relation(z,floor)))
                if method=='b10' and split=='out':
                    plot_records[(case,label)]=b
        print(f'  预测核验完成：{len(models)} 个模型/split',flush=True)
        del frames,arrays,qx
        gc.collect()
    # 所有折的全 split 合并前检查同一 UID 的物理标签一致；历史不同训练族不进入此全集。
    union=pd.concat(unions,ignore_index=True)
    first=union.drop_duplicates('uid').set_index('uid')
    aligned=first.loc[union.uid]
    for col in ['Y','B','C','time']:
        if not np.allclose(union[col],aligned[col].to_numpy(),rtol=2e-7,atol=1e-7):
            raise ValueError(f'折间同 UID 的 {col} 不一致。')
    if not np.array_equal(union.parent,aligned.parent.to_numpy()) or not np.array_equal(union.group,aligned.group.to_numpy()):
        raise ValueError('折间来源身份不一致。')
    if first.parent.duplicated().any():
        raise ValueError('相同 parent 对应不同样本契约，不能静默合并。')
    galaxy=pd.DataFrame([dict(group=SHORT_MAP[g],**describe(a.C)) for g,a in first.groupby('group')])
    galaxy.to_csv(output/'galaxy_true_closure.csv',index=False)
    pd.DataFrame(distribution).to_csv(output/'closure_distributions.csv',index=False)
    pd.DataFrame(training).to_csv(output/'training_closure.csv',index=False)
    metrics=pd.DataFrame(comparisons)
    metrics.to_csv(output/'closure_relations.csv',index=False)
    pd.DataFrame(binned).to_csv(output/'delta_bins.csv',index=False)
    pd.DataFrame(cis).to_csv(output/'block_intervals.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(output/'predictions_and_references.csv.gz',index=False)
    make_plots(output,plot_records,first,training)
    (output/'method.json').write_text(json.dumps(dict(date='2026-11-11',sha256=hashes,sources=sources,
        units='Msun/yr; snapshot 总闭合，不再乘体积',reference='各模型实际去重train；排除同运行±0.25Gyr；每训练运行最多192个等时间锚点决定通道标尺，实际数量见anchor_count；全训练库最近邻；B caliper为事后诊断',
        training_predictions='正式目录仅导出test/out，未将test预测冒称train预测，也未新推理',
        definitions='C=Y-B; Chat=P-B 为最终有效闭合; Delta=C-Cref; R=Chat-Cref; error=R-Delta',
        limitations='相关量共享C；斜率和alpha非因果参数；输入距离非零；全银河统计是各折cache的UID并集；不是完整原模拟；未按out选模型；全部尖峰保留'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('全部统计、图与来源摘要完成。',flush=True)


def make_plots(output,records,galaxy,training):
    """symlog 保留负闭合和尖峰；主图固定模型，不依据留出误差选优。"""
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for group,a in galaxy.groupby('group'):
        v=np.sort(a.C)
        ax.plot(v,np.arange(1,len(v)+1)/len(v),label=SHORT_MAP[group])
    ax.set(xscale='symlog',xlabel='True total closure C [Msun/yr]',ylabel='Empirical CDF',title='Five galaxies: unique cached snapshots across folds')
    ax.legend()
    fig.savefig(output/'galaxy_closure_cdf.png',dpi=170)
    plt.close(fig)
    display={'fiducial':['UNet_original42','UNet_ReLU42','UNet_ReLU43','CNN43'],
             'model075':['UNet'],'fiducial7':['UNet','CNN68']}
    fig,axes=plt.subplots(3,2,figsize=(13,14),constrained_layout=True)
    for row,(case,models) in enumerate(display.items()):
        for model in models:
            a=records[(case,model)]
            axes[row,0].scatter(a.C,a.Chat,s=4,alpha=.25,label=model)
            valid=np.isfinite(a.Cref)
            axes[row,1].scatter(a.loc[valid,'Delta'],a.loc[valid,'R'],s=4,alpha=.2,label=model)
        for col in [0,1]:
            ax=axes[row,col]
            ax.set_xscale('symlog',linthresh=.1)
            ax.set_yscale('symlog',linthresh=.1)
            low,high=ax.get_xlim()
            ax.plot([low,high],[low,high],'k--',lw=1)
            ax.axhline(0,color='grey',lw=.5)
            ax.axvline(0,color='grey',lw=.5)
            ax.legend(fontsize=8)
        axes[row,0].set(xlabel='C = Y - B [Msun/yr]',ylabel='Predicted effective closure [Msun/yr]',title=case+': true vs predicted closure')
        axes[row,1].set(xlabel='Delta = C - Cref [Msun/yr]',ylabel='R = Chat - Cref [Msun/yr]',title=case+': identity = truth; horizontal zero = reference')
    fig.savefig(output/'closure_reference_relations.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    for ax,(case,models) in zip(axes,display.items()):
        base=records[(case,models[0])]
        valid=np.isfinite(base.Cref)
        for name,values in [('C',base.loc[valid,'C']),('Cref',base.loc[valid,'Cref'])]:
            v=np.sort(values)
            ax.plot(v,np.arange(1,len(v)+1)/len(v),label=name,lw=2)
        for model in models:
            a=records[(case,model)]
            v=np.sort(a.loc[np.isfinite(a.Cref),'Chat'])
            ax.plot(v,np.arange(1,len(v)+1)/len(v),label=model+' Chat',ls='--')
        ax.set_xscale('symlog',linthresh=.1)
        ax.set(xlabel='Closure [Msun/yr]',ylabel='Empirical CDF',title=case+' out: matched samples')
        ax.legend(fontsize=8)
    fig.savefig(output/'closure_reference_distributions.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for ax,(case,models) in zip(axes,display.items()):
        entries=[]
        for mode in ['unique','sampling_weighted']:
            stats=next(t for t in training if t['case']==case and t['group']=='ALL' and t['weighting']==mode)
            entries.append(('Train C '+mode,stats))
        base=records[(case,models[0])]
        entries.extend([('Out C',describe(base.C)),('Matched Cref',describe(base.loc[np.isfinite(base.Cref),'Cref']))])
        entries.extend((model+' Chat',describe(records[(case,model)].Chat)) for model in models)
        for i,(label,stats) in enumerate(entries):
            ax.plot([stats['min'],stats['max']],[i,i],color='#aaaaaa',lw=1)
            ax.plot([stats['p05'],stats['p95']],[i,i],color='#4477aa',lw=5,
                    label='p05-p95' if i==0 else None)
            ax.scatter(stats['median'],i,color='black',s=28,marker='D',label='median' if i==0 else None,zorder=3)
            ax.scatter(stats['mean'],i,color='#cc3311',s=32,marker='x',label='mean' if i==0 else None,zorder=4)
        ax.set_yticks(range(len(entries)),[e[0] for e in entries],fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale('symlog',linthresh=.1)
        ax.set(xlabel='Total closure [Msun/yr]',title=case+'; thin line = min-max')
    axes[-1].legend(fontsize=8)
    fig.savefig(output/'training_and_prediction_closure.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir.resolve())
