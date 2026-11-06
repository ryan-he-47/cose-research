"""2026-10-22：全面验收汇总、对象概览、原始问答表；CPU，不加载权重。"""
import argparse
from collections import Counter,defaultdict
import html
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def draw_case(ax,records):
    ax.set(xlim=(0,1),ylim=(1,0),aspect='equal',xticks=[],yticks=[])
    records=[r for r in records if r.get('generated') and r['kind'] not in ('bos','eop','eod')]
    page=records[0]['page_id'] if records else None
    for row in records:
        if row['page_id']==page:ax.plot(*np.asarray(row['points']).T,color='#2859a4',lw=1)


def previews(out):
    summary=read(out/'joint_summary.json')
    cases=[c for c in summary['cases'] if c['task']=='O1']
    for mode in ('greedy','sample'):
        fig,axes=plt.subplots(3,4,figsize=(13,10))
        for ax,c in zip(axes.flat,cases):
            value=read(out/c['name']/(mode+'.json'));draw_case(ax,value['strokes']);ax.set_title(c['label'])
        fig.suptitle('O1 / '+mode+' / generated strokes only');fig.tight_layout()
        fig.savefig(out/('objects_'+mode+'.png'),dpi=140);plt.close(fig)
    scenes=[c for c in summary['cases'] if c['task']=='S2']
    fig,axes=plt.subplots(len(scenes),3,figsize=(11,3.5*len(scenes)),squeeze=False)
    for row,c in enumerate(scenes):
        reference=read(out/c['name']/'reference.json')['strokes']
        gt=[dict(r,generated=True) for r in reference if r.get('targetable')]
        for col,(name,records) in enumerate((('reference',gt),('greedy',read(out/c['name']/'greedy.json')['strokes']),('sample',read(out/c['name']/'sample.json')['strokes']))):
            draw_case(axes[row,col],records);axes[row,col].set_title(c['name']+' / '+name)
    fig.tight_layout();fig.savefig(out/'scenes_overview.png',dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--previews_only',action='store_true')
    args=parser.parse_args();out=args.out;previews(out)
    if args.previews_only:return
    source=read(out/'curves/summary.json');audit=read(out/'audit.json')
    lineage=read(out/'source_lineage.json');diagnostic=read(out/'diagnostic_summary.json')
    text=read(out/'text/summary.json');stage=read(Path('tmp_smoke/r_joint_stage_20261020/text/summary.json'))
    initial=read(out/'joint/validation_initial.json')
    teachers=read(out/'balanced_teacher_summary.json');recognition=read(out/'recognition.json')
    scenes=read(out/'scene_abilities.json');localization=read(out/'localization.json');joint=read(out/'joint_summary.json')
    previous_path=out/'recognition_861m/recognition.json'
    previous=read(previous_path) if previous_path.exists() else None
    text_rows=[]
    for name,r in (('文字基座',initial),('8.61亿图文',stage['teacher_forcing']),('23.86亿图文',text['teacher_forcing'])):
        a=r['text_aux'];text_rows.append([name,f"{a['all']['shape_ce']:.4f}",f"{1-a['all']['shape_error']:.2%}",
            f"{1-a['first']['shape_error']:.2%}",f"{1-a['continuation']['shape_error']:.2%}",f"{1-a['all']['space_error']:.2%}",f"{r['content_nll']:.2f}"])
    graph=[];first=[]
    for model,rows in teachers.items():
        for r in rows:
            if r['task'] in ('O1','S2','S4') and r['kind'] in ('sketch','selection'):
                graph.append([model,r['split'],r['task'],r['count'],f"{r['nll']:.3f}",f"{r['greedy_latent_mse']:.4f}",f"{r['sigma']:.4f}"])
            if r['split']=='valid' and r['task'] in ('O2','S1','S3','S5') and r['kind'] in ('text','answer_initial'):
                first.append([model,r['task'],r['kind'],r['count'],f"{r['shape_ce']:.4f}",f"{r['shape_accuracy']:.2%}"])
    loc_groups=defaultdict(list)
    for r in localization['cases']:loc_groups['有明确几何词' if r.get('explicit_geometry_words') else '无明确几何词'].append(r)
    loc_summary={k:dict(count=len(v),correct=sum(r['nearest_object_correct'] for r in v),
        median_center_error=float(np.median([r['center_error'] for r in v if r['center_error'] is not None]))) for k,v in loc_groups.items()}
    stops=defaultdict(Counter)
    for c in joint['cases']:
        for mode in ('greedy','sample'):
            value=read(out/c['name']/(mode+'.json'));stops[c['task']+'/'+mode][value['stop_reason']]+=1
    summary=dict(step=audit['step'],phase_step=audit['phase_step'],joint_inputs=audit['successful_inputs'],
        mainline_inputs=source['total_lineage_inputs'],text=text['teacher_forcing'],
        recognition=recognition['balanced'],recognition_anchor=recognition['anchor32'],
        previous_recognition=previous['balanced'] if previous else None,
        scene_metrics=scenes['metrics'],qa_by_type=scenes['by_query'],
        localization={k:v for k,v in localization.items() if k!='cases'},localization_by_words=loc_summary,
        generation_stops={k:dict(v) for k,v in stops.items()},
        source_lineage=lineage,diagnostic=diagnostic,
        training_paused=True,anneal_started=False,optimizer_updates=0)
    density=read(out/'density_probe.json') if (out/'density_probe.json').exists() else None
    if density:summary['density_probe']=density
    write(out/'full_summary.json',summary)
    css='''body{font:16px/1.6 system-ui;color:#23344b;margin:24px;max-width:1500px}h2{border-top:2px solid #cdd7e4;padding-top:18px;margin-top:32px}table{border-collapse:collapse;margin:12px 0;font-size:14px}td,th{border:1px solid #cbd5e1;padding:6px 12px;max-width:650px;vertical-align:top}th{background:#edf3fa}img{max-width:100%}.note{padding:18px;background:#edf3fa}details{padding:12px;border:1px solid #d6dfeb;margin:12px 0}.scroll{overflow:auto}a{color:#175ea8}pre{white-space:pre-wrap}'''
    page=f'''<!doctype html><meta charset="utf-8"><title>暂停模型完整验收 / {audit['step']}</title><style>{css}</style>
<h1>暂停模型完整验收 / 累计{audit['step']}步</h1><div class="note"><p>图文阶段{audit['phase_step']}步，{audit['successful_inputs']:,}输入。实际沿用28702步文字模型及Adam；该初始化是代理自行选择，用户未要求。</p>
<p><b>训练保持停止，未启动退火。</b>本次全为本机CUDA/FP16只读推理。AE逐项一致，模型权重有限，模型/头推理前后hash一致。</p>
<p><a href="curves/index.html">完整离线交互曲线与CSV</a> · <a href="index.html">全部逐笔与GMM样张</a> · <a href="text/index.html">文字续写</a> · <a href="full_summary.json">汇总JSON</a></p></div>
<h2>文字能力</h2>'''
    page+=table(('模型','形状CE','形状准确率','字符首笔','字符续接','逐笔空格','文字NLL'),text_rows)
    page+='<p>同128篇valid/161044目标。形状分类是笔画级；正文NLL是48维联合密度。自由文字仍有错拼和重复，完整四前文及两例GMM闭环见文字页。</p>'
    page+='<h2>对象绘制与场景绘制</h2><p>greedy为最高π分量均值；sample温度0.5。下面只画生成笔画，完整条件/原笔/AE参考和教师强制预测在样张页。</p>'
    page+='<h3>12类对象 / greedy</h3><img src="objects_greedy.png"><details><summary>同12类采样</summary><img src="objects_sample.png"></details>'
    page+='<h3>场景生成</h3><img src="scenes_overview.png">'
    page+='<details><summary>图形训练/留出与中期对照指标</summary>'+table(('模型','分区','任务','笔画数','NLL','最高π λ MSE','最高π σ'),graph)+'</details>'
    page+='<h2>自由识图 O2</h2>'
    rec_rows=[['23.86亿',345,recognition['balanced']['correct'],f"{recognition['balanced']['correct']/345:.2%}"]]
    if previous:rec_rows.insert(0,['8.61亿',345,previous['balanced']['correct'],f"{previous['balanced']['correct']/345:.2%}"])
    page+=table(('检查点','每类一例','完整类别正确','比例'),rec_rows)
    page+=f"<p>另保留原32例锚点：当前{recognition['anchor32']['correct']}/32。只给真实图形/问题前缀，用标准字体、形状/逐笔空格头和字体笔顺约束闭环；无外部词典、无真值标签回馈。</p>"
    page+='<details><summary>所有类别的实际回答</summary>'+table(('真值','生成','正确'),[[r['expected'],r['generated'],r['correct']] for r in recognition['cases']])+'</details>'
    page+='<h2>场景问答 S3 与圈内识别 S5</h2>'
    page+='<p><b>本批存在性9题全部回答yes；计数31题全部回答1；方向36题全部回答yes。</b>总命中率中的这部分来自固定答案先验。</p>'
    page+=table(('任务/类型','数量','回答正确','该组最多见答案数量'),[[r['task']+'/'+r['kind'],r['count'],r['correct'],r['majority_correct']] for r in scenes['by_query'] if r['task']!='S1'])
    page+='<details><summary>全部问题与回答</summary>'+table(('任务','问题','真值','生成','正确'),[[r['task'],r['question'] or '识别圈内对象',r['expected'],r['generated'],r['correct']] for r in scenes['cases'] if r['task']!='S1'])+'</details>'
    page+='<h2>圈选 S4</h2>'
    page+=table(('检查点','选中目标','中心误差均值','中心误差中位数','bbox IoU均值','bbox IoU中位数'),
        [[k,v['correct'],f"{v['center_mean']:.4f}",f"{v['center_median']:.4f}",f"{v['bbox_iou_mean']:.4f}",f"{v['bbox_iou_median']:.4f}"] for k,v in diagnostic['localization'].items()])
    page+=f"<p>{localization['count']}例，预测圈中心最近对象与目标一致{localization['nearest_object_correct']}例；中心误差中位数{localization['median_center_error']:.4f}（单位页），椭圆外接bbox IoU中位数{localization['median_bbox_iou']:.4f}。</p>"
    page+=table(('实际问题含几何词','数量','选中目标','中心误差中位数'),[[k,v['count'],v['correct'],f"{v['median_center_error']:.4f}"] for k,v in loc_summary.items()])
    page+='<p>此分桶按实际问题中的leftmost/above等词；它仍可同时包含类别识别。下图为前12例锚点，全部96例在表中。</p><img src="localization_overview.png"><details><summary>全部圈选问题与落点</summary>'
    page+=table(('编号','问题','最近对象','目标一致','中心误差','bbox IoU'),[[r['index'],r['question'],r.get('predicted_nearest_label','无笔画'),r['nearest_object_correct'],r['center_error'],r['bbox_iou']] for r in localization['cases']])+'</details>'
    page+='<h2>描述 S1 与布局落墨 S2</h2>'
    page+=table(('任务','指标'),[['S1',scenes['metrics']['S1']],['S2 greedy',scenes['metrics']['S2']['greedy']],['S2 sample',scenes['metrics']['S2']['sample']]])
    page+='<p>S1类名/方位词为词面覆盖统计；S2只统计目标框内落墨比例、目标框是否被碰到，不能替代对象类别与数量正确性。下面保留全部描述供直接查看。</p>'
    page+='<details><summary>场景描述逐例</summary>'+table(('真值描述','生成描述'),[[r['expected'],r['generated']] for r in scenes['cases'] if r['task']=='S1'])+'</details>'
    page+='<h2>逐笔回答与答案开头</h2><p>answer_initial仅为回答第一笔的形状准确率；自由整类结果另列。</p><div class="scroll">'+table(('模型','任务','分桶','笔数','形状CE','准确率'),first)+'</div>'
    page+='<h2>生成停止状态</h2>'+table(('任务/解码','停止统计'),[[k,dict(v)] for k,v in stops.items()])
    if density:
        page+='<h3>整个R混合密度的众数</h3>'+table(('任务','条件','生成笔数','正文笔数','停止'),
            [[r['task'],r['label'],r['generated_strokes'],r['body_strokes'],r['stop']] for r in density['cases']])
    counters=audit['counters'];body=counters['successful_body'];control=counters['successful_control']
    graphic=sum(counters['body:'+t] for t in ('O1','S2','S4'))
    summary['supervision']=dict(body=body,control=control,graphic_body=graphic,text_body=body-graphic,
        supervised_efficiency=(body+control)/audit['successful_inputs'],graphic_fraction_of_body=graphic/body)
    page+='<h2>实际训练量与数据曝光</h2>'
    page+=f"<p>图文输入{audit['successful_inputs']:,}笔，其中监督正文{body:,}笔、控制符{control:,}笔。图形/圈选正文{graphic:,}笔，占监督正文{graphic/body:.2%}；输入成本与目标份额不同。</p>"
    page+=table(('任务','输入笔数','监督正文笔数','正文目标份额'),[[t,counters['inputs:'+t],counters['body:'+t],f"{counters['body:'+t]/body:.2%}"] for t in ('T1','O1','O2','S1','S2','S3','S4','S5')])
    page+=f"<p>图文阶段T1游标重新从0开始；本阶段消费前{lineage['joint_T1_cursor']:,}篇，文字基座已消费前{lineage['parent_T1_cursor']:,}篇。数据池、seed、增强和排版相同，前段发生重复曝光。</p>"
    write(out/'full_summary.json',summary)
    (out/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('step','joint_inputs','recognition','previous_recognition','scene_metrics','localization_by_words')}))


if __name__=='__main__':main()
