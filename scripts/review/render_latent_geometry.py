"""冻结 AE 几何检查的中文图表与报告，仅读取已完成的指标文件。"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


NAMES={'ae_overall':'AE原训练整体','ae_body':'AE原训练正文','ae_text':'AE流中文字',
       'ae_object':'AE流对象笔画','ae_scene':'AE流场景笔画','r_overall':'当前R全部编码输入',
       'r_targets':'当前R真实正文目标','r_text':'当前R文字','r_object':'当前R对象笔画','r_scene':'当前R场景笔画',
       'text_stratified_text':'1024故事文字分项','validation_object':'345类验证对象分项','validation_scene':'5000验证场景分项'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def two_nn(data):
    runs=data['dimension']['runs'];n=max(r['n'] for r in runs)
    selected=[r for r in runs if r['n']==n]
    return n,float(np.mean([r['two_nn_mle'] for r in selected]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False})
    profiles={}
    for file in sorted((out/'profiles').glob('*.json')):
        value=load(file)
        if 'spectrum' in value: profiles[value['name']]=value
    primary=[k for k in ('ae_overall','ae_body','r_overall','r_targets','ae_object','ae_scene','text_stratified_text') if k in profiles]
    colors=plt.cm.tab10(np.linspace(0,.9,len(primary)))
    figures=[]
    def save(fig,name):
        fig.savefig(out/name,dpi=160,bbox_inches='tight');plt.close(fig);figures.append(name)

    fig,ax=plt.subplots(figsize=(11,3.4),layout='constrained')
    mixtures=[]
    for corpus in ('ae','r'):
        counts=load(out/'data'/f'{corpus}_pipeline.json')['sample_counts']
        total=sum(counts.values())
        mixtures.append([sum(v for k,v in counts.items() if k.startswith(group+':'))/total*100 for group in ('text','object','scene','aux')])
    left=np.zeros(2)
    for index,(label,color) in enumerate(zip(('文字','对象笔画','场景笔画','控制/选择符'),('#3c956a','#2674b2','#d98a38','#87929c'))):
        values=np.asarray(mixtures)[:,index]
        bars=ax.barh([0,1],values,left=left,label=label,color=color)
        ax.bar_label(bars,labels=[f'{v:.1f}%' if v>=4 else '' for v in values],label_type='center',color='white')
        left+=values
    ax.set_yticks([0,1],['AE原训练流','当前R全部编码输入'])
    ax.set(xlim=(0,100),xlabel='抽样后的真实笔画频率 %',title='两套训练管线实际看到的分布')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=4,fontsize=9)
    save(fig,'pipeline_mixture.png')

    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for name,color in zip(primary,colors):
        eig=np.asarray(profiles[name]['spectrum']['eigenvalues'])
        axes[0].semilogy(np.arange(1,49),np.maximum(eig,1e-12),label=NAMES[name],color=color)
        axes[1].plot(np.arange(1,49),np.cumsum(eig)/eig.sum(),label=NAMES[name],color=color)
    axes[0].set(xlabel='方向（按方差排序）',ylabel='当前归一化后的方差',title='各方向尺度差异')
    axes[1].set(xlabel='方向数',ylabel='累计方差占比',title='主要变化集中在哪些方向')
    axes[1].axhline(.95,color='grey',linestyle=':',linewidth=.8)
    for ax in axes: ax.grid(alpha=.18)
    axes[0].legend(fontsize=8);save(fig,'spectrum.png')

    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for name,color in zip(primary,colors):
        axes[0].plot(np.arange(1,49),profiles[name]['spectrum']['snr1_noise_times'],color=color,label=NAMES[name])
    axes[0].set(title='每个主方向何时达到信噪比1',xlabel='主方向（按方差排序）',ylabel='噪声比例 t（由1向0去噪）')
    axes[0].legend(fontsize=7)
    times=np.linspace(.005,.95,200)
    eig=np.asarray(profiles['text_stratified_text']['spectrum']['eigenvalues'])
    for index in (0,1,2,7,15,31,47):
        axes[1].semilogy(times,(1-times)**2*eig[index]/times**2,label=f'第{index+1}方向')
    axes[1].axhline(1,color='grey',ls=':')
    axes[1].set(title='1024故事文字分项：各方向的SNR',xlabel='噪声比例 t',ylabel='信号方差 / 噪声方差')
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle('主方向尺度见SNR曲线；字形可辨性见实际加噪检索图',fontsize=11)
    save(fig,'noise_scales.png')

    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    for name,color in zip(primary,colors):
        runs=profiles[name]['dimension']['runs']
        ns=sorted(set(r['n'] for r in runs))
        ys=[np.mean([r['two_nn_mle'] for r in runs if r['n']==n]) for n in ns]
        low=[min(r['two_nn_mle'] for r in runs if r['n']==n) for n in ns]
        high=[max(r['two_nn_mle'] for r in runs if r['n']==n) for n in ns]
        axes[0].plot(ns,ys,'o-',label=NAMES[name],color=color)
        axes[0].fill_between(ns,low,high,color=color,alpha=.12)
        last=max(ns);rr=[r for r in runs if r['n']==last]
        axes[1].plot([np.mean([r['mle'][str(k)]['radius']['p50'] for r in rr]) for k in (10,20,40)],
                     [np.mean([r['mle'][str(k)]['pooled'] for r in rr]) for k in (10,20,40)],'o-',color=color)
    axes[0].set(title='Two-NN：随样本量的变化',xlabel='去重后无放回子样本数',ylabel='维度估计')
    axes[1].set(title='近邻MLE：随空间尺度的变化',xlabel='近邻半径中位数（标准化潜空间）',ylabel='维度估计')
    calibration=load(out/'profiles'/'calibration.json') if (out/'profiles'/'calibration.json').exists() else {}
    truth=[];est=[];mle=[]
    for key,value in calibration.items():
        rr=max(value['runs'],key=lambda r:r['n']);truth.append(int(key));est.append(rr['two_nn_mle']);mle.append(rr['mle']['20']['pooled'])
    axes[2].plot(truth,truth,'--',color='grey',label='真实维度')
    axes[2].plot(truth,est,'o-',label='Two-NN');axes[2].plot(truth,mle,'o-',label='MLE k=20')
    axes[2].set(title='同样本量的已知维度高斯校准',xlabel='真实维度',ylabel='估计维度');axes[2].legend(fontsize=8)
    axes[0].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=.18)
    save(fig,'dimension.png')

    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for name,color in zip(primary,colors):
        cluster=profiles[name]['clustering']['ks'];ks=list(map(int,cluster))
        axes[0].plot(ks,[cluster[str(k)]['real'] for k in ks],'o-',label=NAMES[name],color=color)
        axes[0].plot(ks,[cluster[str(k)]['matched_gaussian'] for k in ks],':',color=color,alpha=.55)
    axes[0].set(title='轮廓系数：实线真实数据，虚线匹配协方差高斯',xlabel='KMeans簇数',ylabel='Silhouette')
    usable=[name for name in primary if (out/'profiles'/f'{name}_gaussian_dimension.json').exists()]
    xs=np.arange(len(usable))
    axes[1].bar(xs-.18,[two_nn(profiles[n])[1] for n in usable],.36,label='真实连续样本')
    axes[1].bar(xs+.18,[two_nn(load(out/'profiles'/f'{n}_gaussian_dimension.json'))[1] for n in usable],.36,label='匹配协方差高斯')
    axes[1].set_xticks(xs,[NAMES[n] for n in usable],rotation=30,ha='right',fontsize=8)
    axes[1].set(title='各向异性本身会把维度估计压低多少',ylabel='Two-NN（最大子样本）')
    axes[0].legend(fontsize=7);axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.18)
    save(fig,'clusters_and_null.png')

    glyph=load(out/'glyph_metrics.json');extra=load(out/'glyph_geometry.json')
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    ts=np.array(list(map(float,glyph['noise'])))
    means=np.array([r['macro_recall'] for r in glyph['noise'].values()])
    ci=np.array([r['ci95'] for r in glyph['noise'].values()])
    axes[0].plot(ts,means*100,'o-',color='#1d6fb1');axes[0].fill_between(ts,ci[:,0]*100,ci[:,1]*100,alpha=.2)
    if (out/'stroke_discrimination.json').exists():
        stroke=load(out/'stroke_discrimination.json')
        st=np.array(list(map(float,stroke['noise'])))
        sy=np.array([r['macro_recall'] for r in stroke['noise'].values()])
        sc=np.array([r['ci95'] for r in stroke['noise'].values()])
        axes[0].plot(st,sy*100,'s-',color='#3c956a',label='逐笔形状（同形合并）')
        axes[0].fill_between(st,sc[:,0]*100,sc[:,1]*100,color='#3c956a',alpha=.15)
    axes[0].plot([],[],color='#1d6fb1',label='完整字形（已知字界和笔数）')
    axes[0].axvline(1/32,color='#b44f32',ls=':',label='t=1/32（均匀32步网格参考）')
    axes[0].axhline(glyph['chance_given_stroke_count']*100,color='grey',ls=':',label='已知笔数下随机选字')
    axes[0].set(title='固定位置字形：加噪后的近邻可辨性',xlabel='噪声比例 t（1为纯噪声）',ylabel='宏平均召回率 %',ylim=(0,102))
    labels=['原曲线\n固定位置','潜变量\n固定位置','原曲线\n变化位置','潜变量\n变化位置','D重建\n固定位置','D重建\n变化位置后对齐']
    values=[glyph['modes']['raw_fixed']['macro_recall'],glyph['modes']['latent_fixed']['macro_recall'],
        extra['relationships']['raw_varied']['accuracy']['macro_recall'],glyph['modes']['latent_varied']['macro_recall'],
        extra['decoded']['latent']['macro_recall'],extra['decoded']['varied']['macro_recall']]
    bars=axes[1].bar(np.arange(len(values)),np.array(values)*100,color=['#87929c','#2674b2','#87929c','#2674b2','#3c956a','#3c956a'])
    axes[1].bar_label(bars,fmt='%.2f',fontsize=8)
    axes[1].set_xticks(range(len(labels)),labels,fontsize=8);axes[1].set(ylim=(0,110),ylabel='宏平均召回率 %',title='字形信息保留与位置干扰')
    axes[0].legend(fontsize=7);axes[0].grid(alpha=.18)
    fig.suptitle('完整字形94类，每类256个查询；逐笔形状按单笔检索。冻结E/D测量。',fontsize=11)
    save(fig,'glyph_discrimination.png')

    with np.load(out/'glyph_predictions.npz') as data:
        truth=data['truth'];chars=glyph['characters'];count=len(chars)
        confusion={}
        fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
        for ax,mode,title in zip(axes,('latent_fixed','latent_varied'),('固定位置','变化位置')):
            matrix=np.zeros((count,count),np.int64)
            np.add.at(matrix,(truth,data[mode]),1)
            rates=matrix/matrix.sum(1,keepdims=True)
            confusion[mode]=dict(characters=chars,counts=matrix.tolist())
            im=ax.imshow(rates,cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
            ticks=np.arange(0,count,5)
            ax.set_xticks(ticks,[chars[i] for i in ticks],fontsize=8)
            ax.set_yticks(ticks,[chars[i] for i in ticks],fontsize=8)
            ax.set(title=title,xlabel='检索预测字符',ylabel='真实字符')
        fig.colorbar(im,ax=axes,label='每类查询的比例',shrink=.8)
        fig.suptitle('完整字形94类混淆矩阵（已知字界和笔数，每类256次查询）',fontsize=11)
        save(fig,'glyph_confusion.png')
        (out/'glyph_confusion.json').write_text(json.dumps(confusion,ensure_ascii=False),encoding='utf8')

    smooth=load(out/'smoothness.json') if (out/'smoothness.json').exists() else None
    if smooth:
        fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
        for (name,item),color in zip(smooth['groups'].items(),['#1f77b4','#d97721','#359166']):
            rows=item['perturbations']['random'];sigmas=np.array(list(map(float,rows)))
            axes[0].loglog(sigmas,[r['relative_to_span']['p50'] for r in rows.values()],'o-',color=color,label=name+' 中位数')
            axes[0].loglog(sigmas,[r['relative_to_span']['p95'] for r in rows.values()],':',color=color,label=name+' 95分位')
            axes[1].semilogx(sigmas,[100*r['component_switch_fraction']['mean'] for r in rows.values()],'o-',color=color,label=name)
            axes[2].semilogy(np.arange(1,49),item['jacobian']['median_spectrum'],color=color,label=name)
        axes[0].set(title='随机扰动导致的相对曲线变化',xlabel='标准化潜变量每维RMS扰动',ylabel='曲线RMS变化 / 原笔画跨度')
        axes[1].set(title='实际D读出的分量切换',xlabel='标准化潜变量每维RMS扰动',ylabel='切换τ点比例 %')
        axes[2].set(title='固定分量分支的Jacobian谱',xlabel='奇异值序号',ylabel='曲线RMS / 潜变量L2')
        for ax in axes: ax.legend(fontsize=7);ax.grid(alpha=.18)
        save(fig,'decoder_smoothness.png')
        fig,axes=plt.subplots(3,3,figsize=(11,9),layout='constrained')
        for row,name in enumerate(('object','scene','text')):
            d=np.load(out/f'smoothness_{name}.npz')
            with np.load(out/'data'/f'{"text" if name=="text" else "ae"}_pipeline.npz') as source:
                points,offsets=source['points'],source['offsets']
            for col,(index,label) in enumerate(zip(d['preview_ids'],('中位','95分位','最大'))):
                axis=axes[row,col];base=d['base'][index];changed=d['preview_changed'][index]
                original_index=d['indices'][index]
                original=points[offsets[original_index]:offsets[original_index+1]]
                axis.plot(original[:,0],original[:,1],':',color='#359166',marker='.',markersize=2,label='原始输入笔画')
                axis.plot(base[:,0],base[:,1],color='#647382',label='未扰动的D输出')
                axis.plot(changed[:,0],changed[:,1],color='#c75b30',label='潜变量扰动后')
                joint=np.concatenate([original,base,changed]);center=(joint.min(0)+joint.max(0))/2
                radius=max(np.ptp(joint,axis=0).max()*.58,1e-6)
                axis.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]+radius,center[1]-radius))
                axis.set_aspect('equal');axis.set_title(f'{name} · 相对变化{label} · 原跨度{d["span"][index]:.3g}',fontsize=9)
                rms=np.sqrt(np.mean(np.sum((changed-base)**2,axis=1)))
                axis.text(.03,.03,f'绝对变化RMS={rms:.3g}',transform=axis.transAxes,fontsize=7)
                axis.tick_params(labelsize=7)
                axis.locator_params(axis='both',nbins=3)
            del points,offsets
        axes[0,0].legend(fontsize=7)
        fig.suptitle('标准化潜变量随机扰动每维RMS=0.01；按完整样本的相对变化选取示例',fontsize=11)
        save(fig,'decoder_examples.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for (name,item),color in zip(smooth['groups'].items(),['#1f77b4','#d97721','#359166']):
            rows=item['perturbations']['random'];sigmas=np.array(list(map(float,rows)))
            axes[0].loglog(sigmas,[r['translation_rms']['p50'] for r in rows.values()],'o-',color=color,label=name+' 平移')
            axes[0].loglog(sigmas,[r['centered_shape_rms']['p50'] for r in rows.values()],'--',color=color,label=name+' 形状')
            axes[1].loglog(sigmas,[r['curve_rms']['p95'] for r in rows.values()],'o-',color=color,label=name+' 全部变化')
            axes[1].loglog(sigmas,[max(r['switch_extra_rms']['p95'],1e-12) for r in rows.values()],':',color=color,label=name+' 分量切换差异')
        axes[0].set(title='平移与去平移形状变化（中位数）',xlabel='潜变量每维RMS扰动',ylabel='绝对页面坐标RMS')
        axes[1].set(title='尾部：绝对变化与分量切换差异（95分位）',xlabel='潜变量每维RMS扰动',ylabel='绝对页面坐标RMS')
        for ax in axes: ax.legend(fontsize=7);ax.grid(alpha=.18)
        save(fig,'decoder_components.png')
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        for ax,(name,item) in zip(axes,smooth['groups'].items()):
            for direction,label in (('random','随机方向'),('local_high_variance','局部最大方差方向'),('local_low_variance','局部最小方差方向')):
                rows=item['perturbations'][direction]
                ax.loglog(list(map(float,rows)),[r['curve_rms']['p50'] for r in rows.values()],'o-',label=label)
            ax.set(title={'object':'对象笔画','scene':'场景笔画','text':'文字笔画'}[name],xlabel='潜变量每维RMS扰动',ylabel='绝对页面坐标RMS（中位数）')
            ax.grid(alpha=.18);ax.legend(fontsize=7)
        fig.suptitle('局部方向取64个近邻PCA的最大和最小方差方向',fontsize=11)
        save(fig,'decoder_directions.png')

    if (out/'position_geometry.json').exists():
        position=load(out/'position_geometry.json')
        names=[n for n in primary if n in position]
        fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained');xs=np.arange(len(names))
        ax.bar(xs-.18,[position[n]['top2_variance_fraction']*100 for n in names],.36,label='前两主方向占总方差')
        ax.bar(xs+.18,[position[n]['top2_variance_linearly_explained_by_centroid']*100 for n in names],.36,label='前两主方向的方差可由质心线性解释')
        ax.set_xticks(xs,[NAMES[n] for n in names],rotation=20,ha='right',fontsize=9)
        ax.set(ylabel='百分比',ylim=(0,105),title='主方差方向是否主要对应空间位置');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.18)
        save(fig,'position_geometry.png')

    rows=[];summary={}
    for name in NAMES:
        if name not in profiles: continue
        d=profiles[name];s=d['spectrum'];n,id_est=two_nn(d)
        rows.append(f'<tr><td><a href="profiles/{name}.json">{NAMES[name]}</a></td><td>{d["n"]:,}</td><td>{d["source_units"]:,}</td><td>{s["participation_ratio"]:.2f}</td><td>{s["covariance_entropy_rank"]:.2f}</td><td>{s["pca"]["0.95"]}</td><td>{id_est:.2f}（n={n:,}）</td></tr>')
        summary[name]=dict(n=d['n'],units=d['source_units'],pr=s['participation_ratio'],entropy_rank=s['covariance_entropy_rank'],pca95=s['pca']['0.95'],two_nn=id_est,id_n=n)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    observations=(out/'observations.html').read_text(encoding='utf8') if (out/'observations.html').exists() else '<p>指标已生成，文字解释尚在核对。</p>'
    sections=''.join(f'<figure><a href="{f}"><img src="{f}" loading="lazy"></a></figure>' for f in figures)
    source_links=''.join(f'<li><a href="{p.relative_to(out).as_posix()}">{html.escape(p.name)}</a></li>' for p in sorted((out/'data').glob('*.json')))
    for filename,label in (('stroke_discrimination.json','逐笔形状区分与噪声曲线'),('glyph_confusion.json','94类完整混淆计数'),
                           ('position_geometry.json','位置关联统计'),('summary.json','几何摘要'),('audit.json','来源及覆盖核验'),
                           ('../../docs/records/ae/2026-11-03_diffusion_latent_geometry_review.md','中文执行与解读记录')):
        source_links+=f'<li><a href="{filename}">{label}</a></li>'
    content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>冻结AE几何检查 · 2026-11-03</title><style>body{{max-width:1250px;margin:40px auto;padding:0 24px;background:#f6f7f9;color:#172434;font:16px/1.8 "Microsoft YaHei",sans-serif}}h1{{font-size:30px}}h2{{margin-top:40px}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dce2e8;text-align:left}}figure{{margin:24px 0;padding:10px;background:white;border-radius:10px}}img{{width:100%;height:auto}}a{{color:#1466a5}}.note{{background:#e8eef4;padding:18px;border-radius:8px}}details{{background:white;padding:16px;margin-top:16px}}code{{font-size:13px}}</style>
<h1>冻结 AE 的空间几何与字形区分度</h1><p>2026-11-03 · 使用接受的55k AE与当前扩散归一化 · AE原训练流、当前R训练流和分项样本</p>
{observations}<h2>整体与分项的数值</h2><table><thead><tr><th>样本视图</th><th>笔画数</th><th>来源单位数</th><th>参与率PR</th><th>协方差熵秩</th><th>PCA95%方向数</th><th>Two-NN</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<details><summary>统计口径与样本单位</summary><p>整体统计保留真实频率；维度与聚类分析剔除固定BOS/EOP/EOD点，维度估计另去除完全重复向量。来源单位按记录索引计数，固定控制符可在多个索引中重复出现。谱用全量样本；近邻维度用4千、8千、最多1.6万子样本，聚类用4千子样本。维度图阴影显示两次重复子样本的范围。字形95%区间按独立查询实例重采样；逐笔95%区间按整字实例分组重采样。</p></details>
{sections}<details><summary>来源、方法和原始指标</summary><p>AE真实配额为对象1843、场景1843、文字205、辅助205笔/4096笔。R按已保存八任务份额从训练流取样，再对全部真实编码输入作均匀无放回蓄水池抽样；监督目标视图按实际切块标记筛选，不计合成EOP。</p><ul>{source_links}<li><a href="glyph_metrics.json">字形近邻与噪声曲线</a></li><li><a href="glyph_geometry.json">字形间隔与重建读出</a></li><li><a href="smoothness.json">解码局部稳定性</a></li><li><a href="control_geometry.json">控制符潜空间距离</a></li><li><a href="../../docs/records/ae/2026-11-03_diffusion_latent_geometry_metrics.md">指标定义与文献</a></li></ul></details></html>'''
    (out/'report.html').write_text(content,encoding='utf8')
    print(json.dumps(dict(report=str(out/'report.html'),profiles=len(profiles),figures=figures),ensure_ascii=False))


if __name__=='__main__': main()
