"""按UID检查异常及邻帧cache输入、训练分布；不加载raw，不重归一化或重新推理。"""
from pathlib import Path
from uuid import uuid4
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from VisualScripts.ResultAnalysis.analyze_sfr_unet import load_predictions, file_hash


def select_neighbors(times, target, neighbors, tolerance):
    """按实际时间选择，不跨数组索引猜测原始snapshot编号。"""
    times=np.asarray(times)
    if neighbors<0 or tolerance<0 or not np.isfinite(target):
        raise ValueError('邻帧数/容差/目标时间不合法。')
    order=np.argsort(times)
    if not np.isfinite(times).all() or np.any(np.diff(times[order])<=0):
        raise ValueError('同一运行时间必须有限且唯一。')
    center=int(np.argmin(abs(times[order]-target)))
    if abs(times[order[center]]-target)>tolerance:
        raise ValueError(f'没有匹配 {target} Gyr 的帧；最近为 {times[order[center]]}。')
    return order[center],order[max(0,center-neighbors):min(len(order),center+neighbors+1)]


def distribution(reference, value):
    """训练经验分位与秩；超出范围只说明样本覆盖，不判定数据错误。"""
    flat=np.asarray(reference).ravel();finite=flat[np.isfinite(flat)]
    names=['min','p01','p50','p99','max']
    values=np.quantile(finite,[0,.01,.5,.99,1]) if len(finite) else [np.nan]*5
    result=dict(zip(names,map(float,values)))
    result.update(nonfinite_count=int(len(flat)-len(finite)),value=float(value),
                  percentile=float(100*np.mean(finite<=value)) if len(finite) and np.isfinite(value) else np.nan)
    return result


def nearest_patches(train, query, theta, radial, radius=1, count=5):
    """标准化输入局部patch的RMS距离，分批计算；不是概率或自动OOD阈值。"""
    h,w=query.shape[-2:];s=(slice(None),slice(max(0,theta-radius),min(h,theta+radius+1)),
                           slice(max(0,radial-radius),min(w,radial+radius+1)))
    q=query[s].astype(float)
    if not np.isfinite(q).all():return []
    distances=np.full(len(train),np.inf)
    for start in range(0,len(train),256):
        a=train[start:start+256][(slice(None),)+s].astype(float)
        good=np.isfinite(a).all(axis=(1,2,3))
        d=np.sqrt(np.mean((a-q)**2,axis=(1,2,3)));d[~good]=np.inf
        distances[start:start+len(a)]=d
    return [(int(i),float(distances[i])) for i in np.argsort(distances)[:count] if np.isfinite(distances[i])]


def inspect_inputs(result_dir, cache_dir, *, times=(8.99375,9.405,12.44125), neighbors=5,
                   time_tolerance=1e-6, cells=None, channels=None, plots=True, output_dir=None):
    """cells可指定每个目标的(theta,r)；默认采用该模型中心帧绝对误差最大格子。"""
    root=Path(result_dir).resolve();cache=Path(cache_dir)
    contract=json.loads((root/'contract.json').read_text(encoding='utf-8'))
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'))
    if contract['cache_metadata']['experiment']!='exp2':raise ValueError('当前检查入口仅支持Exp2 out。')
    pred=load_predictions(root/'out_predictions.pt',identity=manifest['out'],
                          expected_groups=contract['cache_metadata']['held_out_groups'],expected_loss=contract.get('loss'))
    names=contract['data']['input_channels'];selected=names if channels is None else list(channels)
    if not set(selected)<=set(names):raise ValueError('未知输入通道。')
    loaded={};hashes={}
    for split in ('train','out'):
        path=cache/f"{config['data_exp']}{split}.pt"
        digest=file_hash(path)
        expected=(contract['data']['train_val']['train'] if split=='train' else contract['evaluation_identities']['out'])['scalar_sha256']
        if digest!=expected:raise ValueError(f'{split} cache SHA256与本次训练不一致。')
        hashes[split]=digest
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['sample_identity']!={k:manifest[split][k] for k in ('sample_uid','parent_hr_uid')}:
            raise ValueError(f'{split} UID/顺序与manifest不一致。')
        if payload['model_input_contract']['input_channels']!=names:raise ValueError('通道顺序不一致。')
        if payload['preprocessing_contract']['log_transform_mask']!=contract['data']['normalization']['log_transform_mask']:
            raise ValueError('输入变换契约不一致。')
        loaded[split]=payload
    train_ids=loaded['train']['sample_identity']['sample_uid']
    _,unique=np.unique(train_ids,return_index=True);unique=np.sort(unique)
    train=np.asarray(loaded['train']['tensors']['x'])[unique]
    x=np.asarray(loaded['out']['tensors']['x']);coords=np.asarray(loaded['out']['tensors']['coord'])
    time=np.asarray(loaded['out']['tensors']['time']).ravel()
    if not np.allclose(time,pred['time'],atol=1e-10,rtol=0):raise ValueError('cache与预测时间不一致。')
    if train.shape[1:]!=x.shape[1:] or x.shape[0]!=len(time) or x.shape[2:]!=pred['target_map'].shape[2:]:
        raise ValueError('输入与地图shape不一致。')
    if (set(train_ids)&set(pred['sample_uid']) or set(manifest['train']['parent_hr_uid'])&set(pred['parent_hr_uid'])):
        raise ValueError('train/out UID交叉。')
    if cells is not None and len(cells)!=len(times):raise ValueError('cells与times数量不一致。')
    destination=Path(output_dir) if output_dir else root/f'input_review_{uuid4().hex[:10]}'
    destination.mkdir(parents=True,exist_ok=False)
    global_stats={name:distribution(train[:,names.index(name)],np.nan) for name in selected}
    frames=[];points=[];fieldstats=[];nearest=[]
    for event,target in enumerate(times):
        center,indices=select_neighbors(time,target,neighbors,time_tolerance)
        error=abs(pred['prediction_map'][center,0]-pred['target_map'][center,0])
        ti,ri=tuple(cells[event]) if cells is not None else np.unravel_index(np.argmax(error),error.shape)
        if not (0<=ti<x.shape[2] and 0<=ri<x.shape[3]):raise ValueError('格子索引越界。')
        for pos,i in enumerate(indices):
            frames.append(dict(event=event,requested_time=target,time=time[i],offset=pos-int(np.where(indices==center)[0][0]),
                previous_dt_gyr=time[i]-time[indices[pos-1]] if pos else np.nan,cache_row=int(i),
                sample_uid=pred['sample_uid'][i],parent_hr_uid=pred['parent_hr_uid'][i],theta_index=int(ti),r_index=int(ri),
                r_kpc=float(coords[i,0,ti,ri]),theta_rad=float(coords[i,1,ti,ri]),
                true_sfr=pred['target_total'][i]*.025,predicted_sfr=pred['prediction_total'][i]*.025,baseline_sfr=pred['baseline_total'][i]*.025))
            for name in selected:
                k=names.index(name);v=x[i,k,ti,ri]
                for scope,ref in [('same_cell',train[:,k,ti,ri]),('local_3x3',train[:,k,max(ti-1,0):ti+2,max(ri-1,0):ri+2])]:
                    points.append(dict(event=event,time=time[i],channel=name,scope=scope,**distribution(ref,v)))
                points.append(dict(event=event,time=time[i],channel=name,scope='all_cells',
                                   **{**global_stats[name],'value':float(v)}))
                a=x[i,k];finite=a[np.isfinite(a)]
                fieldstats.append(dict(event=event,time=time[i],channel=name,nonfinite=int(a.size-len(finite)),
                                       min=float(finite.min()) if len(finite) else np.nan,max=float(finite.max()) if len(finite) else np.nan))
        for rank,(j,distance) in enumerate(nearest_patches(train,x[center],ti,ri)):
            nearest.append(dict(event=event,time=time[center],rank=rank+1,train_cache_row=int(unique[j]),
                                train_uid=train_ids[unique[j]],distance=distance,
                                train_time=float(np.asarray(loaded['train']['tensors']['time']).ravel()[unique[j]])))
        if plots:
            folder=destination/f'event{event}';folder.mkdir()
            triplet=sorted(set([indices[max(0,np.where(indices==center)[0][0]-1)],center,indices[min(len(indices)-1,np.where(indices==center)[0][0]+1)]]))
            for name in selected:
                k=names.index(name);a=x[triplet,k];finite=a[np.isfinite(a)]
                limits=(float(finite.min()),float(finite.max())) if len(finite) else (0,1)
                fig,axes=plt.subplots(1,len(triplet),figsize=(4*len(triplet),3),squeeze=False,layout='constrained')
                for ax,i in zip(axes[0],triplet):
                    im=ax.imshow(np.ma.masked_invalid(x[i,k]),origin='lower',aspect='auto',vmin=limits[0],vmax=limits[1])
                    ax.plot(ri,ti,'rx');ax.set_title(f'{name}: t={time[i]:.5f}');ax.set_xlabel('r index');ax.set_ylabel('theta index')
                fig.colorbar(im,ax=axes.ravel().tolist(),label='Cached standardized input')
                fig.savefig(folder/f'{name}_maps.png',dpi=110);plt.close(fig)
            fig,axes=plt.subplots(len(selected),1,figsize=(9,2*len(selected)),squeeze=False,layout='constrained')
            for ax,name in zip(axes[:,0],selected):
                k=names.index(name);ref=distribution(train[:,k,ti,ri],np.nan)
                ax.axhspan(ref['p01'],ref['p99'],alpha=.15,label='Train same-cell p01-p99')
                ax.plot(time[indices],x[indices,k,ti,ri],'-o',ms=3);ax.axvline(time[center],color='red',lw=.8)
                ax.set_ylabel(name);ax.ticklabel_format(useOffset=False,axis='x')
            axes[0,0].legend();axes[-1,0].set_xlabel('Snapshot time [Gyr]')
            fig.savefig(folder/'input_history.png',dpi=110);plt.close(fig)
    for name,rows in [('frames',frames),('input_points',points),('input_fields',fieldstats),('nearest_train',nearest)]:
        pd.DataFrame(rows).to_csv(destination/f'{name}.csv',index=False)
    audit=dict(result_dir=str(root),cache_hashes=hashes,channels=names,targets=list(times),neighbors=neighbors,
               time_tolerance=time_tolerance,cells=cells,training_rows=len(train_ids),unique_training_rows=len(unique),
               normalization='原cache标准化值，不重新变换',reference='训练UID去重；不使用val/out估计范围',
               notes=['距离为全部输入通道同位置3x3 patch的RMS，边界截短；不是异常概率',
                      '空间图红叉为选定格子；默认由中心帧预测绝对误差选择，非因果归因',
                      '全局分位已计算，global percentile留空；局部经验秩包含并列值',
                      '坐标从cache读取，格子索引不是原始snapshot编号；未访问raw或重新推理'])
    (destination/'inspection.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return dict(output_dir=str(destination),frames=pd.DataFrame(frames),points=pd.DataFrame(points),nearest=pd.DataFrame(nearest))
