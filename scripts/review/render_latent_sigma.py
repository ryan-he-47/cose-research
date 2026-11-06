"""2026-10-09：λ几何与标准差下界测量的曲线、扰动样张和本机查看页。"""

# 2026-10-12：分类目录中的脚本也可从仓库根直接运行。
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();root=args.directory
    r=json.loads((root/'measurements.json').read_text(encoding='utf-8'))
    kinds=['plain_text','aug_text','sketch_object','sketch_scene','same_slot_glyph']
    eps=np.asarray(r['sigma_values']);fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for kind in kinds:
        g=r['geometry'][kind]
        axes[0].plot(g['std_by_dimension'],label=kind)
        axes[1].plot(eps,[g['nearest_pair_gaussian_error'][str(e)]['p95'] for e in eps],label=kind)
        axes[2].plot(eps,[r['decoder_noise'][str(e)][kind]['point_coordinate_rms']['p90']*1024 for e in eps],label=kind)
    axes[0].set(xlabel='latent coordinate',ylabel='global standard deviation',yscale='log')
    axes[1].set(xlabel='minimum component sigma',ylabel='nearest pair error / p95',xscale='log')
    axes[2].set(xlabel='isotropic Gaussian sigma',ylabel='added canvas coordinate RMS at 1024px / p90',xscale='log',yscale='log')
    for axis in axes:axis.grid(alpha=.2);axis.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'geometry_and_resolution.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    gradient_cases=[('T1 add',r['t1_gradient']),*[(k+' add',v) for k,v in r['multitask_gradients'].items()]]
    gradient_cases += [('T1 quadrature',r['t1_quadrature_gradients']['T1']),
                       *[(k+' quadrature',v) for k,v in r['multitask_quadrature_gradients'].items()]]
    for name,report in gradient_cases:
        values=report['values']
        for group,style in [('body','-'),('control','--')]:
            axes[0].plot(eps,[values[str(e)][group]['parameter_gradient_l2'] for e in eps],style,label=name+' '+group)
        axes[1].plot(eps,[values[str(e)]['control']['parameter_gradient_l2']/values[str(e)]['body']['parameter_gradient_l2'] for e in eps],label=name)
    for kind in ['plain_text','sketch_object','sketch_scene','eop']:
        axes[2].plot(eps,[r['decoder_noise'][str(e)][kind]['point_coordinate_rms']['p90']*1024 for e in eps],label=kind)
    axes[0].set(ylabel='weighted parameter gradient norm',yscale='log')
    axes[1].set(ylabel='control / body gradient norm',yscale='log')
    axes[2].set(ylabel='decoder perturbation RMS / 1024px / p90',yscale='log')
    for axis in axes:axis.set(xscale='log',xlabel='sigma floor');axis.grid(alpha=.2);axis.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'gradient_and_distortion.png',dpi=180);plt.close(fig)

    a=np.load(root/'perturbations.npz',allow_pickle=True)
    chosen=['plain_text','aug_text','sketch_object','sketch_scene','same_slot_glyph','eop']
    columns=[0,2,3,4,6]
    fig,axes=plt.subplots(len(chosen),len(columns),figsize=(15,3*len(chosen)),squeeze=False)
    for row,kind in enumerate(chosen):
        idx=int(np.flatnonzero(a['bucket']==kind)[0]);base=a['baseline'][idx]
        cloud=np.concatenate([base,*[a['curves'][c,idx] for c in columns]])
        center=(cloud.max(0)+cloud.min(0))/2;size=max(float(np.ptp(cloud,axis=0).max())*1.2,.025)
        for col,c in enumerate(columns):
            axis=axes[row,col];p=a['curves'][c,idx]
            axis.plot(*base.T,color='#aaa',lw=2.1);axis.plot(*p.T,color='#176c9c',lw=1)
            axis.set(xlim=(center[0]-size/2,center[0]+size/2),ylim=(center[1]+size/2,center[1]-size/2),aspect='equal',xticks=[],yticks=[])
            axis.set_title(f'{kind} / sigma {eps[c]:g}',fontsize=9)
    fig.suptitle('One fixed Gaussian direction: gray decoded center / blue perturbed decode; same scale per row')
    fig.tight_layout(rect=(0,0,1,.985));fig.savefig(root/'noise_examples.png',dpi=175);plt.close(fig)

    geometry=[]
    for kind,g in r['geometry'].items():
        nearest=g.get('nearest_unique_rms',{})
        geometry.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in
            [kind,g['count'],g['unique_embeddings'],f"{g['coordinate_std']['p50']:.5g}",
             f"{nearest.get('p50',0):.5g}",f"{g['fp16_fp32_rms']['p50']:.5g}"] )+'</tr>')
    noise=[]
    for e in eps:
        noise.append('<tr><td>'+str(e)+'</td>'+''.join(f"<td>{r['decoder_noise'][str(e)][k]['point_coordinate_rms']['p90']*1024:.3f}</td>"
            for k in ['plain_text','aug_text','sketch_object','sketch_scene','same_slot_glyph','eop'])+'</tr>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>λ几何与σ下界</title>
<style>body{font:16px system-ui,"Microsoft YaHei";max-width:1450px;margin:32px auto;padding:0 24px;color:#203047;background:#f8fafc}img{max-width:100%;background:white;border:1px solid #ddd}table{border-collapse:collapse;background:white}th,td{border:1px solid #ddd;padding:10px;text-align:right}th:first-child,td:first-child{text-align:left}p{line-height:1.65}</style>
<h1>冻结λ的尺度与标准差下界</h1><p>所有测量只读，无optimizer更新。σ表示标准差，方差为σ²。全局尺度、局部间距和解码分辨率分开查看。</p>
<h2>实际嵌入几何</h2><table><tr><th>分桶</th><th>笔数</th><th>不同嵌入数</th><th>逐维std中位数</th><th>不同嵌入近邻RMS中位数</th><th>FP16/FP32差RMS中位数</th></tr>GEOMETRY</table>
<p>近邻去掉完全相同的嵌入；样本重复数量另存JSON。最邻近一对中心的等权、等宽高斯区分错误为Φ(−距离/2σ)，用于看局部宽度，不能代替语义识别。</p><img src="geometry_and_resolution.png">
<h2>下界对应的解码扰动</h2><p>把分量中心放在真实λ，加N(0,σ²I)，每笔8个方向；解码使用FP32。表为相对中心解码新增的画布坐标RMS的p90，乘1024方便看像素量。它表示最小宽度的采样分辨率，均值本身仍可精确定位。</p>
<table><tr><th>σ</th><th>无增强文字</th><th>增强文字</th><th>对象</th><th>场景</th><th>同字槽字形</th><th>EOP</th></tr>NOISE</table><img src="noise_examples.png">
<h2>实际FP16梯度</h2><p>当前T1与原多任务R，单个首回答页按原loss权重，仅反向；loss scale512。add：σ=s+ε；quadrature：σ=√(s²+ε²)。全部候选下界使用同一权重与前文。诊断页的正文/控制分母与正式大批不同，原始数值保留在JSON。</p><img src="gradient_and_distortion.png"><p><a href="measurements.json">完整逐维、近邻、扰动、梯度和固定GMM数据</a></p></html>'''
    (root/'index.html').write_text(page.replace('GEOMETRY',''.join(geometry)).replace('NOISE',''.join(noise)),encoding='utf-8')
    print(json.dumps({'page':str(root/'index.html')}))


if __name__=='__main__':
    main()
