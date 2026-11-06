"""2026-10-26：从头图文最终验收、50021步能力对照和退火观察汇总。"""
import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.review.review_joint_scratch_stage import read, write
from scripts.review.render_joint_full_review import table
from scripts.review.render_joint_scratch_comparison import images, paired, transitions
from scripts.review.render_joint_full_curves import read_rows


def annealing_overview(out, curve):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    start=curve['anneal_start_inputs']/1e8;end=max(curve['total_inputs'],curve['final']['inputs'])/1e8
    rows=[r for r in read_rows(out/'validation.jsonl') if r['successful_inputs']/1e8>=start-1.2]
    fig,axes=plt.subplots(1,3,figsize=(14,3.9))
    values=[('文字几何NLL',[r['content_nll'] for r in rows]),
            ('形状CE',[r['text_aux']['all']['shape_ce'] for r in rows]),
            ('字符首笔准确率 / %',[100*(1-r['text_aux']['first']['shape_error']) for r in rows])]
    for ax,(title,y) in zip(axes,values):
        ax.plot([r['successful_inputs']/1e8 for r in rows],y,'o-',color='#146c94',ms=3,lw=1.5)
        ax.axvspan(start,end,color='#efaa40',alpha=.16)
        ax.axvline(start,color='#d97822',ls='--',lw=1)
        ax.set(xlim=(start-1.2,end),title=title,xlabel='成功输入 / 亿');ax.grid(alpha=.2)
    fig.suptitle('退火前后：同128篇文字验证；橙色为34.56→36亿退火段')
    fig.tight_layout();fig.savefig(out/'annealing_overview.png',dpi=150);plt.close(fig)


def drawing_stops(directory):
    rows=[]
    for case in read(directory/'joint_summary.json')['cases']:
        if case['task']!='O1':continue
        for mode in ('greedy','sample'):
            r=read(directory/case['name']/(mode+'.json'))
            rows.append(dict(label=case['label'],mode=mode,stop=r['stop_reason'],
                             generated_strokes=sum(s['generated'] for s in r['strokes'])))
    return rows


def model_summary(directory):
    text=read(directory/'text/summary.json')['teacher_forcing']
    recognition=read(directory/'recognition.json')
    scene=read(directory/'scene_abilities.json')
    location=read(directory/'localization.json')
    return dict(text=dict(content_nll=text['content_nll'],shape_ce=text['text_aux']['all']['shape_ce'],
                    shape_accuracy=1-text['text_aux']['all']['shape_error'],
                    first_accuracy=1-text['text_aux']['first']['shape_error'],
                    continuation_accuracy=1-text['text_aux']['continuation']['shape_error'],
                    space_accuracy=1-text['text_aux']['all']['space_error']),
                recognition=recognition['balanced'],recognition_anchor=recognition['anchor32'],
                scene=scene['metrics'],localization=dict(correct=location['nearest_object_correct'],
                    center_mean=float(np.mean([r['center_error'] for r in location['cases']])),
                    center_median=location['median_center_error'],
                    iou_mean=float(np.mean([r['bbox_iou'] for r in location['cases']])),
                    iou_median=location['median_bbox_iou']))


def render(out, baseline):
    audit=read(out/'audit.json');curve=read(out/'curves/summary.json')
    legacy=ROOT/'tmp_smoke/r_joint_full_review_20261022'
    snapshots=[model_summary(p) for p in (legacy,baseline,out)]
    prior,final=snapshots[1:]
    old_rec,new_rec=[read(p/'recognition.json')['cases'] for p in (baseline,out)]
    balanced=set(read(out/'probe_bundle.json')['splits']['valid']['balanced_indices'])
    rp=[(a,b) for a,b in paired(old_rec,new_rec) if a['index'] in balanced]
    old_scene,new_scene=[read(p/'scene_abilities.json') for p in (baseline,out)]
    sp=paired(old_scene['cases'],new_scene['cases'],('task','index'))
    assert all(a['expected']==b['expected'] and a['question']==b['question'] for a,b in sp)
    old_loc,new_loc=[read(p/'localization.json')['cases'] for p in (baseline,out)]
    lp=paired(old_loc,new_loc)
    location_transition=transitions([(dict(a,correct=a['nearest_object_correct']),dict(b,correct=b['nearest_object_correct'])) for a,b in lp])
    query=defaultdict(list)
    for a,b in sp:query[(a['task'],a['query_kind'])].append((a,b))
    query_rows=[];distributions={}
    for (task,kind),rows in query.items():
        outputs=[Counter(pair[i]['generated'] for pair in rows) for i in (0,1)]
        query_rows.append([task+'/'+kind,len(rows),sum(a['correct'] for a,b in rows),sum(b['correct'] for a,b in rows),
                           outputs[0].most_common(3),outputs[1].most_common(3)])
        distributions[task+'/'+kind]=[dict(d) for d in outputs]
    loc_keywords={name:{str(k):dict(count=sum(r['explicit_geometry_words']==k for r in rows),
                    correct=sum(r['nearest_object_correct'] for r in rows if r['explicit_geometry_words']==k)) for k in (True,False)}
                    for name,rows in (('50021',old_loc),('final',new_loc))}
    server=read(out/'server_finished.json')
    result=dict(step=audit['step'],successful_inputs=audit['successful_inputs'],target_inputs=audit['target_inputs'],
                source_commit=audit['source_commit'],job=server['job'],stop=server['stop'],
                snapshots=dict(zip(('legacy_joint','step50021','final'),snapshots)),
                annealing=curve,recognition_transitions=transitions(rp),localization_transitions=location_transition,
                localization_by_words=loc_keywords,answer_distributions=distributions,
                teacher=read(out/'teacher_comparison.json'),weights_unchanged=True,optimizer_updates=0,
                frozen_ae_equal=audit['frozen_ae_equal'],weights_finite=audit['weights_finite'],
                object_generation_stops=dict(step50021=drawing_stops(baseline),final=drawing_stops(out)),
                pre_anneal_checkpoint_available=False,
                comparison_note='退火附近变化用完整验证曲线；自由能力对照为50021步到最终，包含退火前后新增训练。')
    images(out,baseline,labels=('50021','final'))
    annealing_overview(out,curve)
    write(out/'final_summary.json',result)
    overview=[
        ['文字形状准确率',*[f"{r['text']['shape_accuracy']:.2%}" for r in snapshots]],
        ['字符首笔准确率',*[f"{r['text']['first_accuracy']:.2%}" for r in snapshots]],
        ['文字正文几何NLL',*[f"{r['text']['content_nll']:.3f}" for r in snapshots]],
        ['O2完整类别/345',*[r['recognition']['correct'] for r in snapshots]],
        ['S3场景问答/96',*[r['scene']['S3']['correct'] for r in snapshots]],
        ['S5圈内识别/96',*[r['scene']['S5']['correct'] for r in snapshots]],
        ['S4选中目标/96',*[r['localization']['correct'] for r in snapshots]],
        ['S1描述真实类名词面召回',*[f"{r['scene']['S1']['class_recall']:.2%}" for r in snapshots]],
    ]
    anneal_rows=[]
    for label,key,percentage in [('正文几何NLL','content_nll',False),('控制符NLL','control_nll',False),
            ('形状CE','shape_ce',False),('形状准确率','shape_accuracy',True),('字符首笔准确率','first_accuracy',True),
            ('字符续接准确率','continuation_accuracy',True),('逐笔空格准确率','space_accuracy',True)]:
        a,b=curve['last_before'][key],curve['final'][key]
        anneal_rows.append([label,f'{a:.2%}' if percentage else f'{a:.5f}',
                            f'{b:.2%}' if percentage else f'{b:.5f}',
                            f'{(b-a)*100:+.3f}个百分点' if percentage else f'{b-a:+.5f}'])
    css='body{font:16px/1.65 system-ui;margin:28px;color:#24344b;max-width:1550px}table{border-collapse:collapse;margin:16px 0}td,th{border:1px solid #cbd5e1;padding:6px 10px;vertical-align:top;max-width:560px}th{background:#edf3f8}img{max-width:100%}h2{border-top:2px solid #dbe3ed;margin-top:38px;padding-top:20px}pre{white-space:pre-wrap;background:#f0f5f8;padding:14px}.scroll{overflow:auto}.note{background:#fff4df;padding:16px}'
    page=f'<!doctype html><meta charset="utf-8"><title>从头图文最终验收 / {audit["step"]}步</title><style>{css}</style><h1>从头图文最终验收 / {audit["step"]}步</h1>'
    page+=f'<p>2026-10-26，单V100作业685064正常完成，结束原因input_budget。最终成功输入{audit["successful_inputs"]:,}，原预算{audit["target_inputs"]:,}。冻结AE逐项相同，两组Adam均到最终步，权重有限；本机CUDA/FP16只读验收，零优化更新。</p>'
    page+='<div class="note"><b>两种对照分别查看：</b>退火段为34.56→36亿输入，LR从1e-4降至1e-5；其起点权重已被两份过程权重保留策略轮换，但完整验证记录仍在。退火变化采用退火前最近验证点和退火内全部验证点；实际能力采用已保留的50021步（32.8459亿输入）与最终模型比较。此间同时包含退火前的新增训练。</div>'
    page+='<p><a href="curves/index.html">打开完整离线交互曲线</a> · <a href="curves/annealing_validation.png">退火验证PNG</a> · <a href="final_summary.json">全部汇总JSON</a></p>'
    page+='<h2>能力概览</h2>'+table(('项目','初版joint归档','50021步','最终54825步'),overview)
    page+='<p>三列沿用相同检查条件；初版joint分数复用原归档。下方并排绘图与逐例变化均为50021步→最终，greedy最高π均值；sample temperature=0.5、seed42；文字使用原形状/空格头和字体笔顺闭环，beam=1，未启用候选重排。</p>'
    page+='<h2>退火前后的验证变化</h2>'+table(('同128篇valid','退火前最近点','最终','变化'),anneal_rows)
    page+=f'<p>退火前最近点为第{curve["last_before"]["step"]}步/{curve["last_before"]["inputs"]:,}输入，距退火起点约{(curve["anneal_start_inputs"]-curve["last_before"]["inputs"])/1e6:.2f}百万输入；退火内共有{curve["anneal_validation_points"]}个验证点。下图保留原始点。</p><img src="annealing_overview.png"><details><summary>全部文字验证分项</summary><img src="curves/annealing_validation.png"></details>'
    page+='<h2>文字续写与直接GMM回馈</h2>'
    text=[read(p/'text/summary.json') for p in (baseline,out)]
    for i in range(4):
        page+='<h3>'+html.escape(text[0]['cases'][i]['prompt'])+'</h3>'
        for name,item in zip(('50021步','最终'),text):
            row=item['cases'][i];parts=row['generated_page_texts']
            body='\n'.join(v for k,v in sorted(parts.items(),key=lambda q:int(q[0])) if int(k)>0)
            page+='<b>'+name+'</b><pre>'+html.escape(body)+'</pre>'
    page+='<p><a href="text/index.html">最终文字完整样张</a></p><details><summary>直接GMM前两页对照</summary><img src="text_gmm_comparison.png"></details>'
    page+='<h2>对象与场景绘制</h2><p>每一对图依次为50021步、最终。场景图另列原始目标。</p><img src="objects_comparison_greedy.png"><details><summary>相同条件采样</summary><img src="objects_comparison_sample.png"></details><img src="scenes_comparison.png">'
    stop_rows=[[name,mode,sum(r['mode']==mode and r['stop']=='eod_eop' for r in rows),
                [r['label'] for r in rows if r['mode']==mode and r['stop']!='eod_eop']]
               for name,rows in result['object_generation_stops'].items() for mode in ('greedy','sample')]
    page+=table(('对象绘制','模式','自然结束/12','未自然结束的类'),stop_rows)
    graph=[];answer=[]
    for label,key in (('50021','old'),('最终','scratch')):
        for r in result['teacher'][key]:
            if r['task'] in ('O1','S2','S4') and r['kind'] in ('sketch','selection'):
                graph.append([label,r['split'],r['task'],r['count'],f"{r['nll']:.4f}",f"{r['greedy_latent_mse']:.5f}",f"{r['sigma']:.5f}"])
            if r['split']=='valid' and r['kind'] in ('text','answer_initial'):
                answer.append([label,r['task'],r['kind'],r['count'],f"{r['shape_accuracy']:.2%}",f"{r['shape_ce']:.4f}"])
    page+='<details><summary>图形几何分项：同源train/valid</summary>'+table(('模型','分区','任务','笔数','NLL','最高π均值λ MSE','最高πσ'),graph)+'</details>'
    page+='<h2>自由识图</h2><p>逐例变化：'+html.escape(str(result['recognition_transitions']))+'</p><details><summary>345类逐例回答</summary>'+table(('真实类别','50021步','最终','旧正确','最终正确'),[[a['expected'],a['generated'],b['generated'],a['correct'],b['correct']] for a,b in rp])+'</details>'
    page+='<h2>场景问答、圈内识别与描述</h2>'+table(('任务/类型','题数','50021正确','最终正确','50021常见输出','最终常见输出'),query_rows)
    page+='<details><summary>全部逐例回答</summary>'+table(('任务','问题','原目标','50021步','最终'),[[a['task'],a['question'],a['expected'],a['generated'],b['generated']] for a,b in sp])+'</details>'
    caption_rows=[[name,f"{r['scene']['S1']['class_precision']:.2%}",f"{r['scene']['S1']['class_recall']:.2%}",f"{r['scene']['S1']['position_word_recall']:.2%}"] for name,r in zip(('50021','最终'),(prior,final))]
    page+=table(('描述词面观察','真实类名precision','真实类名recall','方位词覆盖'),caption_rows)+'<p>词面类名命中不检查实例和位置绑定，实际描述见逐例原文。</p>'
    page+='<h2>圈选</h2>'
    loc_rows=[[name,r['localization']['correct'],f"{r['localization']['center_mean']:.4f}",f"{r['localization']['center_median']:.4f}",f"{r['localization']['iou_mean']:.4f}",f"{r['localization']['iou_median']:.4f}"] for name,r in zip(('50021','最终'),(prior,final))]
    page+=table(('模型','选对/96','中心均值','中心中位数','bbox IoU均值','bbox IoU中位数'),loc_rows)
    page+='<p>逐例变化：'+html.escape(str(location_transition))+'；几何词分组：'+html.escape(str(loc_keywords))+'</p><img src="localization_comparison.png">'
    page+='<details><summary>全部96个圈选</summary>'+table(('编号','问题','50021选对','最终选对','50021中心误差','最终中心误差'),[[a['index'],a['question'],a['nearest_object_correct'],b['nearest_object_correct'],f"{a['center_error']:.4f}",f"{b['center_error']:.4f}"] for a,b in lp])+'</details>'
    page+='<h2>回答首笔与全部笔画</h2>'+table(('模型','任务','分桶','笔数','形状准确率','形状CE'),answer)
    page+='<h2>训练分项与GMM选择</h2><img src="curves/annealing_task_nll.png"><img src="curves/annealing_gmm.png"><p>责任分量统计使用真实目标；最高π均值对应实际greedy分量选择，两者分别看。</p>'
    page+='<p>原始检查点、完整日志、初始化/配方、coverage、逐例JSON、CSV和全部曲线在本目录；服务器训练已结束，未提交新训练或追加预算。</p>'
    (out/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(step=result['step'],snapshots=result['snapshots'],
          recognition_transitions=result['recognition_transitions'],localization_transitions=location_transition),ensure_ascii=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=ROOT/'tmp_smoke/r_joint_scratch_review_20261026')
    args=parser.parse_args();render(args.out.resolve(),args.baseline.resolve())


if __name__=='__main__':main()
