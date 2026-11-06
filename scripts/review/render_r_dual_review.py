"""2026-10-09：双臂目的对应的本机样张、条件评分和实际训练曲线。"""
import argparse
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.review.render_formal_r_review import svg_pages


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plot_page(axis,records):
    axis.set(xlim=(0,1),ylim=(1,0),aspect='equal',xticks=[],yticks=[])
    axis.axhspan(.9,1,color='#f1f5f9')
    generated=[r for r in records if r.get('generated') and r['kind'] not in ('eop','eod','bos')]
    if not records: return
    page=(generated or records)[0]['page_id']
    for r in records:
        if r['page_id']!=page: continue
        xy=np.asarray(r['points'])
        color='#bc3e70' if r['kind'] in ('eod','eop','bos') else '#126f9e' if r.get('generated') else '#777'
        axis.plot(*xy.T,color=color,linewidth=.9)


def records(root,name,file):
    path=root/name/(file+'.json')
    return read(path).get('strokes',[]) if path.exists() else []


def montages(root,baseline):
    t1=root/'t1';old=root/'baseline_t1';correct=root/'corrective'
    if (t1/'summary.json').exists():
        cases=read(t1/'summary.json')
        fig,axes=plt.subplots(len(cases),3,figsize=(11,3*len(cases)),squeeze=False)
        for row,item in enumerate(cases):
            name=item['name']
            for col,(directory,file) in enumerate([(old,'greedy'),(t1,'greedy'),(t1,'sample')]):
                ink=records(directory,name,file)
                label='old' if directory==old else file
                if directory==old and not ink:
                    ink=[r for r in records(t1,name,'greedy') if not r.get('generated')]
                    label='condition'
                plot_page(axes[row,col],ink)
                axes[row,col].set_title(name+' / '+label,fontsize=9)
            axes[row,1].set_xlabel(item['prompt'][:70],fontsize=8)
        fig.suptitle('T1: actual first generated page; gray prefix / blue output / pink control',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.985));fig.savefig(root/'t1_generation.png',dpi=160);plt.close(fig)
        selected=[c for c in cases if 'teacher_metrics' in c]
        fig,axes=plt.subplots(len(selected),4,figsize=(14,3.5*len(selected)),squeeze=False)
        for row,item in enumerate(selected):
            name=item['name']
            for col,(file,label) in enumerate([('reference','source'),('ae_reference','AE'),('teacher_forced','T1 true prefix'),('baseline','old true prefix')]):
                ink=records(old if file=='baseline' else t1,name,'teacher_forced' if file=='baseline' else file)
                plot_page(axes[row,col],ink)
                axes[row,col].set_title(name+' / '+label,fontsize=9)
        fig.tight_layout();fig.savefig(root/'t1_teacher.png',dpi=180);plt.close(fig)
        for item in cases[:4]:
            for variant in item['generation']:
                fig,axis=plt.subplots(figsize=(7,7))
                plot_page(axis,records(t1,item['name'],variant))
                axis.set_title(item['name']+' / '+variant)
                fig.tight_layout();fig.savefig(t1/item['name']/(variant+'_first_page.png'),dpi=220);plt.close(fig)
    if (correct/'summary.json').exists():
        cases=read(correct/'summary.json')
        for filename,selected in [('corrective_objects.png',[c for c in cases if c['task']=='O1']),
                                  ('corrective_scenes.png',[c for c in cases if c['task'] in ('S2','S4')])]:
            if not selected: continue
            fig,axes=plt.subplots(len(selected),5,figsize=(16,3.2*len(selected)),squeeze=False)
            for row,item in enumerate(selected):
                name=item['name']
                for col,(directory,file,label) in enumerate([(correct,'reference','source'),(baseline,'greedy','old greedy'),
                        (correct,'greedy','corrective greedy'),(correct,'sample','corrective sample'),(correct,'teacher_forced','true prefix')]):
                    ink=records(directory,name,file)
                    if file=='reference': ink=[r for r in ink if r.get('targetable')]
                    plot_page(axes[row,col],ink)
                    axes[row,col].set_title(name+' / '+label,fontsize=8)
            fig.tight_layout();fig.savefig(root/filename,dpi=160);plt.close(fig)


def logs(path):
    rows=[];retries=[]
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        try: row=json.loads(line)
        except ValueError: continue
        if row.get('success'): rows.append(row)
        if row.get('capacity_oom_retry'): retries.append(row)
    return rows,retries


def window(rows,start,stop):
    selected=[r for r in rows if start<=r['successful_inputs']<stop]
    counts=Counter();sums=Counter()
    for r in selected:
        for task,value in r['task_nll'].items():
            n=r['body_targets'].get(task,0)
            if value is not None and n:
                counts[task]+=n;sums[task]+=value*n
    return {'printed_updates':len(selected),'task_nll':{k:sums[k]/counts[k] for k in counts}}


def curves(root,assets):
    report={}
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for i,arm in enumerate(['t1','corrective']):
        rows,retries=logs(assets/(arm+'.log'))
        x=np.asarray([r['successful_inputs']/1e6 for r in rows])
        for task in (['T1'] if arm=='t1' else ['T1','O1','O2','S1','S2','S3','S4','S5']):
            axes[i,0].plot(x,[r['task_nll'].get(task) for r in rows],label=task,lw=1)
        axes[i,0].set_title(arm+' / task body NLL');axes[i,0].legend(ncol=4,fontsize=8)
        for bucket in ['text','sketch','control']:
            values=[(r['successful_inputs']/1e6,r.get('gmm_buckets',{}).get(bucket,{}).get('posterior_sigma')) for r in rows]
            values=[v for v in values if v[1] is not None]
            if values: axes[i,1].plot(*np.asarray(values).T,label=bucket)
        axes[i,1].set_yscale('log');axes[i,1].set_title(arm+' / posterior sigma');axes[i,1].legend()
        axes[i,2].plot(x,[r['lr'] for r in rows]);axes[i,2].set_title(arm+' / actual LR')
        for axis in axes[i]: axis.set_xlabel('successful phase input strokes / million');axis.grid(alpha=.2)
        report[arm]={'latest_logged_update':{k:rows[-1][k] for k in ['step','phase_step','successful_inputs','lr']},
            'oom_retries':len(retries),'windows':{name:window(rows,a,b) for name,a,b in
                [('20-40M',20e6,40e6),('80-100M',80e6,100e6),('100-120M',100e6,120e6),('180-192M',180e6,192e6),('192-200M',192e6,201e6)]}}
    fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=160);plt.close(fig)
    (root/'training_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


def gallery(root,baseline):
    data=[]
    for arm,label in [('t1','T1无增强 final'),('baseline_t1','原R：相同无增强条件'),('corrective','纠偏臂：中期checkpoint')]:
        directory=root/arm
        if not (directory/'summary.json').exists(): continue
        for item in read(directory/'summary.json'):
            name=item['name'];d=directory/name;ref=read(d/'reference.json')
            gen=read(d/'greedy.json')
            value={'arm':arm,'arm_label':label,**item,
                'condition':svg_pages([r for r in gen['strokes'] if not r.get('generated')]),
                'reference':svg_pages([r for r in ref['strokes'] if r.get('targetable')]),
                'greedy':svg_pages([r for r in gen['strokes'] if r.get('generated')]),
                'sample':svg_pages([r for r in read(d/'sample.json')['strokes'] if r.get('generated')]),
                'teacher_forced':svg_pages(records(directory,name,'teacher_forced')),
                'ae_reference':svg_pages(records(directory,name,'ae_reference')),
                'latent':svg_pages(records(directory,name,'latent')),
                'old':svg_pages([r for r in records(baseline if arm=='corrective' else root/'baseline_t1',name,'greedy') if r.get('generated')])}
            data.append(value)
    encoded=json.dumps(data,ensure_ascii=False).replace('</',r'<\/')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R双臂验收</title>
<style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f3f5f7;color:#1d2b3a;margin:0}header,main{padding:24px;max-width:1800px;margin:auto}select{font:inherit;padding:9px;margin-right:12px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.panel{background:white;padding:12px;border:1px solid #dce2e8;border-radius:8px}.panel h3{margin:0 0 12px;font-size:17px}.page svg{width:100%;border:1px solid #ddd}.pageid{font-size:12px;color:#64748b}.extra{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}#prompt{margin:16px 0;white-space:pre-wrap}#summary{margin:15px 0;line-height:1.6}@media(max-width:1000px){.grid,.extra{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>
<header><h1>T1语言探索与纠偏R：本机实际样张</h1><p>T1已完成2亿输入；纠偏臂仍运行，本页使用已保存的1.076亿新增输入checkpoint。</p><p>灰：实际前文；蓝：模型输出；粉：识别到的控制符。样张均来自FP16本机推理。</p><a href="training_curves.png">训练曲线</a> · <a href="t1_generation.png">T1概览</a> · <a href="t1_teacher.png">T1真实前文预测</a> · <a href="corrective_objects.png">对象对照</a> · <a href="corrective_scenes.png">场景对照</a></header>
<main><select id="arm"></select><select id="case"></select><div id="prompt"></div><div id="summary"></div><div class="grid">
<section class="panel"><h3>实际条件</h3><div id="condition"></div></section><section class="panel"><h3>源文字/图形参考</h3><div id="reference"></div></section><section class="panel"><h3>Greedy</h3><div id="greedy"></div></section><section class="panel"><h3>Sample 0.5</h3><div id="sample"></div></section></div>
<details open><summary>同条件原R与真实前文预测</summary><div class="extra"><section class="panel"><h3>原R Greedy</h3><div id="old"></div></section><section class="panel"><h3>真实前文下下一笔</h3><div id="teacher_forced"></div></section><section class="panel"><h3>AE重建参考</h3><div id="ae_reference"></div></section><section class="panel"><h3>直接λ回馈观察</h3><div id="latent"></div></section></div></details><details><summary>本例指标</summary><pre id="metrics"></pre></details></main>
<script>const data=DATA;const el=id=>document.getElementById(id);for(const a of [...new Set(data.map(c=>c.arm))]){const o=document.createElement('option');o.value=a;o.textContent=data.find(c=>c.arm===a).arm_label;el('arm').appendChild(o)}function choices(){el('case').replaceChildren();for(const c of data.filter(c=>c.arm===el('arm').value)){const o=document.createElement('option');o.value=c.name;o.textContent=c.name;el('case').appendChild(o)}render()}function render(){const c=data.find(c=>c.arm===el('arm').value&&c.name===el('case').value);for(const id of ['condition','reference','greedy','sample','old','teacher_forced','ae_reference','latent'])el(id).innerHTML=c[id]||'<p>本例未生成此项</p>';el('prompt').textContent='条件：'+(typeof c.prompt==='string'?c.prompt:JSON.stringify(c.prompt));el('summary').textContent=JSON.stringify(c.generation);el('metrics').textContent=JSON.stringify({teacher_metrics:c.teacher_metrics,generation:c.generation},null,2)}el('arm').addEventListener('change',choices);el('case').addEventListener('change',render);choices();</script></html>'''.replace('DATA',encoded)
    (root/'index.html').write_text(page,encoding='utf-8')


def comparisons(root):
    report={}
    for arm in ['t1','baseline_t1','corrective']:
        p=root/arm
        if not (p/'completion.json').exists(): continue
        mixture=read(p/'mixture_summary.json')
        report[arm]={'completion':read(p/'completion.json'),
            'mixture':{kind:{field:(value[field]['mean'] if isinstance(value[field],dict) else value[field])
                for field in ['count','nll','top_pi_mse','posterior_mse','top_pi_sigma','posterior_sigma','top_pi_matches_posterior','mu_derivative_norm']}
                for kind,value in mixture.items()}}
        if (p/'language_pairs.json').exists(): report[arm]['language_pairs']=read(p/'language_pairs.json')
    old=ROOT/'tmp_smoke/r_health_diagnosis_20261009_final/mixture_summary.json'
    if old.exists():
        report['original_multitask_mixture']={kind:{field:(value[field]['mean'] if isinstance(value[field],dict) else value[field])
            for field in ['count','nll','top_pi_mse','posterior_mse','top_pi_sigma','posterior_sigma','top_pi_matches_posterior','mu_derivative_norm']}
            for kind,value in read(old).items()}
    (root/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    tables=[]
    for arm in ['t1','baseline_t1']:
        for pair in report.get(arm,{}).get('language_pairs',[]):
            rows=''.join('<tr><td>'+html.escape(context)+'</td>'+''.join(f'<td>{s["mean_nll"]:.3f}</td>' for s in scores)+'</tr>'
                for context,scores in zip(pair['contexts'],pair['scores']))
            head=''.join('<th>'+html.escape(c)+'</th>' for c in pair['candidates'])
            tables.append(f'<h2>{arm} · {pair["name"]}</h2><table><tr><th>前文</th>{head}</tr>{rows}</table><p>条件交换方向值：{pair["context_swap_preference_shift"]:.4f}</p>')
    (root/'language_pairs.html').write_text('<!doctype html><meta charset="utf-8"><title>T1上下文候选观察</title><style>body{font:16px system-ui;margin:32px;max-width:1000px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:12px}h2{margin-top:32px}</style><h1>3个手工上下文例子</h1><p>每列续句在两种前文下使用相同的笔画、排版和页号。表中为正文每笔平均NLL，越低密度越高。方向值为正表示第二前文相对更偏向第二续句。实际自由续写见<a href="index.html">样张页</a>。</p>'+''.join(tables),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,default=ROOT/'tmp_smoke/r_dual_review_20261009')
    parser.add_argument('--assets',type=Path,default=ROOT/'tmp_smoke/r_dual_review_assets_20261009')
    parser.add_argument('--baseline',type=Path,default=ROOT/'tmp_smoke/r_formal_review_20261009')
    args=parser.parse_args()
    args.review.mkdir(parents=True,exist_ok=True)
    curves(args.review,args.assets);montages(args.review,args.baseline);gallery(args.review,args.baseline);comparisons(args.review)
    print(json.dumps({'page':str(args.review/'index.html')}))


if __name__=='__main__':
    main()
