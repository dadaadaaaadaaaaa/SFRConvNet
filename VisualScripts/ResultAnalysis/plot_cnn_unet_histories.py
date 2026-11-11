"""统一时间、物理单位和坐标轴，绘制四组CNN与UNet的留出SFR历史。"""
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
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from VisualScripts.ResultAnalysis.compare_fiducial_domains import CASES
from VisualScripts.ResultAnalysis.analyze_sfr_unet import load_predictions,file_hash,regression

CNN={
    'fiducial':'Exp2_Rh12x20_Closure_v7_LOO_Fiducial_Trial43_seed42_20261025_224050',
    'fiducial4':'Exp2_LOOfiducial4_NoAug_20260817_172856',
    'fiducial7':'Exp2_Rh12x20_Closure_v7_LOO_Fiducial_7_Trial68_seed42_20261026_125202',
    'model075':'Exp2_Rh12x20_FiveRuns_Closure_LOO_Model075_Trial53_Final_20260913_004906',
}
FID_UNET='Exp2_LOO_Fiducial_SFRUNet_ReLU_Candidate1_seed43_20261107_041950_b08b68f2'


def legacy_times(values,mapping,dt=.00125):
    """旧CSV时间是原snapshot编号乘dt；按现行重启段契约还原绝对时间。"""
    index=np.rint(np.asarray(values)/dt).astype(int)
    if not np.allclose(values,index*dt,atol=1e-6,rtol=0):raise ValueError('旧时间不满足snapshot编号契约')
    time=np.full(len(index),mapping['index_zero_time'],dtype=float)
    good=index>0
    steps=index[good]+((index[good]-1)//mapping['segment_cycle_size'])*mapping['extra_time_steps_per_segment']
    time[good]=mapping['positive_index_offset']+steps*dt
    return time,index


def align_predictions(cnn,unet,legacy=False):
    """新结果按UID对齐，旧无UID结果按已核验绝对时间对齐，并逐帧核对Y。"""
    key='time_key' if legacy else 'uid'
    if legacy:
        cnn=cnn.assign(time_key=np.round(cnn.time,5))
        unet=unet.assign(time_key=np.round(unet.time,5))
    if cnn[key].duplicated().any() or unet[key].duplicated().any():raise ValueError('重复配对键')
    paired=unet.merge(cnn,on=key,suffixes=('_unet','_cnn'),validate='one_to_one')
    if len(paired)!=len(unet):raise ValueError('CNN不能完整覆盖UNet留出帧')
    if not legacy and not np.array_equal(paired.parent_unet,paired.parent_cnn):raise ValueError('parent不一致')
    if not np.allclose(paired.time_unet,paired.time_cnn,atol=1e-6,rtol=0):raise ValueError('时间不一致')
    if not np.allclose(paired.Y_unet,paired.Y_cnn,atol=2e-5,rtol=2e-6):raise ValueError('真实SFR不一致')
    result=pd.DataFrame(dict(time=paired.time_unet,Y=paired.Y_unet,CNN=paired.P_cnn,UNet=paired.P_unet))
    if not np.isfinite(result.to_numpy()).all():raise ValueError('存在非有限预测')
    return result.sort_values('time').reset_index(drop=True),dict(
        matched_N=len(result),cnn_extra_N=len(cnn)-len(result),
        max_truth_difference=float(abs(paired.Y_unet-paired.Y_cnn).max()))


def draw_row(axes,f,title,scale):
    """每对图共享完全相同的limits/ticks/scale；缺失时间段断线。"""
    t=f.time.to_numpy();gaps=np.flatnonzero(np.diff(t)>5*np.median(np.diff(t)))+1
    xx=np.insert(t,gaps,np.nan)
    values=f[['Y','CNN','UNet']].to_numpy()
    lo=min(0.,values.min());hi=values.max();padding=(hi-lo)*.07
    limits=(lo-padding if lo<0 else 0.,hi+padding)
    for ax,model,color in zip(axes,['CNN','UNet'],['#d47726','#087f8c']):
        ax.plot(xx,np.insert(f.Y.to_numpy(),gaps,np.nan),color='#252525',lw=.9,label='True SFR',alpha=.8)
        ax.plot(xx,np.insert(f[model].to_numpy(),gaps,np.nan),color=color,lw=.8,label=model,alpha=.9)
        for i in gaps:ax.axvspan(t[i-1],t[i],color='#eeeeee',zorder=-1)
        ax.set(xlim=(.5,12.5),ylim=limits,title=f'Held-out {title} | {model}',
               xlabel='Snapshot absolute time [Gyr]',ylabel=r'SFR [$M_\odot\,\mathrm{yr}^{-1}$]')
        ax.set_xticks(np.arange(2,13,2));ax.grid(alpha=.16)
        if scale=='symlog':ax.set_yscale('symlog',linthresh=.1)
        else:ax.yaxis.set_major_locator(MaxNLocator(5))
        stats=regression(f.Y.to_numpy(),f[model].to_numpy())
        ax.text(.02,.95,f"MAE = {stats['MAE']:.3f}   $R^2$ = {stats['R2']:.3f}",
                transform=ax.transAxes,va='top',fontsize=10,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=3))
        ax.legend(loc='upper right',framealpha=.9,fontsize=10)


def save(fig,base):
    for suffix in ['png','pdf','svg']:
        path=base.with_suffix('.'+suffix)
        fig.savefig(path,dpi=220)
        if suffix=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    hashes={};sources={};frames={};metrics=[]
    def record(path):hashes[str(path.relative_to(ROOT))]=file_hash(path)
    for case in CNN:
        spec=CASES[case];folder=FID_UNET if case=='fiducial' else spec['unet']['UNet']
        root=ROOT/'Results'/folder
        summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
        contract=json.loads((root/'contract.json').read_text(encoding='utf-8'))
        manifest=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'))
        if summary['status']!='complete' or summary['hpo']:raise ValueError('UNet并非正式完成训练')
        for filename in ['summary.json','contract.json','split_manifest.json','out_predictions.pt']:record(root/filename)
        d=load_predictions(root/'out_predictions.pt',identity=manifest['out'],expected_groups={spec['held']},expected_loss=contract['loss'])
        u=pd.DataFrame(dict(uid=d['sample_uid'],parent=d['parent_hr_uid'],time=d['time'],
                            Y=d['target_total']*.025,P=d['prediction_total']*.025))
        if case=='fiducial':fiducial_unet=u.copy()
        del d;gc.collect()
        cr=ROOT/'Results'/CNN[case];csv=cr/'held_out_test_predictions.csv'
        record(csv);record(cr/'hyperparameters.json')
        c=pd.read_csv(csv)
        if set(c['type'])!={spec['held']}:raise ValueError('CNN留出身份错误')
        c=c.rename(columns={'sample_uid':'uid','parent_hr_uid':'parent','sfr_true':'Y','sfr_pred':'P'})
        # 早期Msun/yr标签未实施代码单位换算；与当前保守Y逐帧比对后统一乘0.025。
        c[['Y','P']]*=.025
        audit={}
        if case=='fiducial4':
            config=ROOT/'.config/RawDataConfig.json'
            raw=json.loads(config.read_text(encoding='utf-8'))
            mapping=raw['disk_galaxy']['fiducial_4']['time_mapping']
            if not mapping:raise ValueError('找不到fiducial4的权威时间映射')
            c['time'],index=legacy_times(c.time.to_numpy(),mapping)
            audit=dict(legacy_time_mapping=mapping,legacy_index_min=int(index.min()),legacy_index_max=int(index.max()),
                       comparison='Historical CNN: 3 runs, 8 channels, direct SFR; UNet: 5 runs, 12 channels, cell-supervised closure. Not an architecture-only ablation.')
            # 私有配置只记录哈希与必要物理映射，不复制本机路径。
            record(config)
        elif case=='model075':audit['comparison']='Historical CNN; training cache and normalization differ from UNet.'
        f,alignment=align_predictions(c,u,legacy=case=='fiducial4')
        frames[case]=f
        sources[case]=dict(CNN=CNN[case],UNet=folder,**alignment,**audit)
        for model in ['CNN','UNet']:
            metrics.append(dict(case=case,model=model,N=len(f),**regression(f.Y.to_numpy(),f[model].to_numpy())))
        print(case,len(f),'verified',flush=True)
    # 另一版Fiducial CNN仅作补充，主图仍固定用户选择的Trial43。
    alternate=CASES['fiducial']['cnn']['CNN7']
    path=ROOT/'Results'/alternate/'held_out_test_predictions.csv'
    record(path)
    c=pd.read_csv(path).rename(columns={'sample_uid':'uid','parent_hr_uid':'parent','sfr_true':'Y','sfr_pred':'P'})
    c[['Y','P']]*=.025
    supplement,alignment=align_predictions(c,fiducial_unet)
    sources['fiducial_trial7_supplement']=dict(CNN=alternate,UNet=FID_UNET,**alignment)
    for scale in ['linear','symlog']:
        fig,axes=plt.subplots(1,2,figsize=(15,4.6),sharex=True,sharey=True,layout='constrained')
        draw_row(axes,supplement,'Fiducial',scale)
        fig.suptitle('Supplement: CNN Trial7 vs UNet seed43 | identical axes',fontsize=11)
        save(fig,output/f'fiducial_trial7_supplement_{scale}')
        for case,f in frames.items():
            fig,axes=plt.subplots(1,2,figsize=(15,4.6),sharex=True,sharey=True,layout='constrained')
            draw_row(axes,f,CASES[case]['title'],scale)
            note='Historical CNN / different training setup' if case in ['fiducial4','model075'] else 'CNN and UNet: different supervision and optimization setups'
            fig.suptitle(f"{note} | aligned snapshots, identical axes | {scale} scale",fontsize=10,color='#555555')
            save(fig,output/f'{case}_cnn_vs_unet_{scale}')
        fig,axes=plt.subplots(4,2,figsize=(15,13.5),sharex=True,sharey='row',layout='constrained')
        for row,(case,f) in enumerate(frames.items()):draw_row(axes[row],f,CASES[case]['title'],scale)
        fig.suptitle('Exp2 held-out SFR histories | identical axes within each row\nFiducial4 and Model0.75 CNN are historical comparisons',fontsize=13)
        save(fig,output/f'all_four_cnn_vs_unet_{scale}')
    pd.DataFrame(metrics).to_csv(output/'metrics.csv',index=False)
    record(Path(__file__))
    (output/'sources.json').write_text(json.dumps(dict(sources=sources,sha256=hashes,
        units='Msun/yr; all plotted CSV SFR values multiplied by 0.025',
        axes='Absolute snapshot Gyr; x=[0.5,12.5]; each pair shares y-limits, ticks, scale; per-row y ranges differ between galaxies',
        handling='No smoothing, clipping or downsampling; gaps >5*median cadence are broken and shaded; only shared UNet snapshots plotted',
        scale='Main linear; supplementary symlog with linthresh=0.1 Msun/yr'),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)
