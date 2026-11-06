"""2026-10-27：扩大评估完整结束后，从逐项汇总生成统计审阅报告。"""
import argparse
from collections import Counter,defaultdict
import csv
import html
import json
import math
from pathlib import Path
import re


def canonical(value):
    text=re.sub(r'[^a-z0-9]+',' ',str(value).lower()).strip()
    return re.sub(r'^(?:a|an|the) ','',text)


def response(expected,actual):
    expected=canonical(expected);actual=canonical(actual)
    if expected in ('yes','no'):return actual.split()[0] if actual.split() else '<empty>'
    if expected.isdecimal():return dict(zero='0',one='1',two='2',three='3',four='4',five='5').get(actual,actual)
    return actual


def proportion(rows):
    n=len(rows);k=sum(r['correct'] for r in rows)
    if not n:return dict(count=0,correct=0,accuracy=None,ci95_wilson=None)
    p=k/n;z=1.959963984540054;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(count=n,correct=k,accuracy=p,ci95_wilson=[center-half,center+half])


def answer_distribution(rows):
    """离线拆分真实答案与混淆计数；错误/无输出仍留在原分母。"""
    groups=defaultdict(list);confusion=defaultdict(Counter)
    for row in rows:
        expected=canonical(row['expected']);actual=response(row['expected'],row.get('generated','<empty>'))
        groups[expected].append(row);confusion[expected][actual]+=1
    details={key:dict(**proportion(value),generated=dict(confusion[key]))
        for key,value in sorted(groups.items())}
    return dict(count=len(rows),by_answer=details,
        balanced_accuracy=sum(r['accuracy'] for r in details.values())/len(details) if details else None,
        validation_majority_accuracy=max((len(value) for value in groups.values()),default=0)/len(rows) if rows else None)


def offline_details(out):
    read_rows=lambda name:[json.loads(line) for line in (out/name).read_text(encoding='utf-8').splitlines() if line.strip()]
    question_rows=read_rows('free_S3.jsonl')
    questions={kind:answer_distribution([r for r in question_rows if r['kind']==kind])
        for kind in sorted({r['kind'] for r in question_rows})}
    candidate_rows=read_rows('candidates.jsonl');candidate_groups={}
    for count in sorted({r['object_count'] for r in candidate_rows}):
        group=[r for r in candidate_rows if r['object_count']==count]
        candidate_groups[str(count)]=dict(count=len(group),topk={str(k):sum(r['topk'][str(k)] for r in group)/len(group) for k in (1,2,4,8,32)},
            target_mass=sum(r['target_mass'] for r in group)/len(group))
    circles=read_rows('free_S4.jsonl');circle_kinds={}
    for kind in sorted({str(r.get('query_kind') or '<no output>') for r in circles}):
        group=[r for r in circles if str(r.get('query_kind') or '<no output>')==kind]
        circle_kinds[kind]=proportion(group)
    object_counts={r['index']:r['object_count'] for r in candidate_rows};circle_counts={}
    if len(object_counts)!=len(candidate_rows) or {r['index'] for r in circles}!=set(object_counts):raise ValueError('候选与自由圈选场景未对齐')
    for count in sorted(set(object_counts.values())):
        group=[r for r in circles if object_counts[r['index']]==count]
        circle_counts[str(count)]=dict(**proportion(group),uniform_object_baseline=1/count)
    paired=read_rows('paired.jsonl');pair_details={}
    for family in sorted({r['family'] for r in paired}):
        groups=defaultdict(dict)
        for row in paired:
            if row['family']==family:groups[row['pair']][row['variant']]=row
        if any(set(g)!={0,1} for g in groups.values()):raise ValueError('成对记录不完整')
        value=dict(count=len(groups),neither_correct=0,only_first_correct=0,only_second_correct=0,both_correct=0)
        unchanged=0;has_text=all('generated' in r for g in groups.values() for r in g.values())
        for g in groups.values():
            key=('both_correct' if g[0]['correct'] and g[1]['correct'] else 'only_first_correct' if g[0]['correct']
                else 'only_second_correct' if g[1]['correct'] else 'neither_correct')
            value[key]+=1
            if has_text:unchanged+=response(g[0]['expected'],g[0]['generated'])==response(g[1]['expected'],g[1]['generated'])
        if has_text:value['unchanged_answer']=unchanged
        pair_details[family]=value
    value=dict(questions=questions,candidate_by_object_count=candidate_groups,circle_by_kind=circle_kinds,circle_by_object_count=circle_counts,pairs=pair_details)
    (out/'offline_details.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return value


def plots(out,free,readouts,candidates,pairs):
    # 独立图片可直接导出；各任务按各自定义展示，不拼成总分。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['font.family']=['Microsoft YaHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    values=[free['O2'],next(r for r in readouts if r['kind']=='O2'),*free['S3']['by_kind'].values(),free['S4'],free['S5']]
    labels=['O2 自由命名','O2 闭集读出',*[f'S3 {k}' for k in free['S3']['by_kind']],'S4 最近对象','S5 圈内命名']
    means=np.array([r['accuracy'] for r in values])*100
    lows=np.array([r['ci95_wilson'][0] for r in values])*100;highs=np.array([r['ci95_wilson'][1] for r in values])*100
    fig,axes=plt.subplots(1,3,figsize=(17,5.4),layout='constrained')
    y=np.arange(len(values));axes[0].barh(y,means,color='#2563eb')
    axes[0].errorbar(means,y,xerr=np.maximum(0,np.stack((means-lows,highs-means))),fmt='none',ecolor='#172033',capsize=3)
    axes[0].set(yticks=y,yticklabels=labels,xlim=(0,100),xlabel='准确率 / %',title='各项输出（95% Wilson区间）');axes[0].invert_yaxis()
    for i,(mean,row) in enumerate(zip(means,values)):axes[0].text(mean+1,i,f'{mean:.1f}% · N={row["count"]:,}',va='center',fontsize=8)
    ks=list(candidates['center_assignment_topk']);ys=[100*candidates['center_assignment_topk'][k]['accuracy'] for k in ks]
    axes[1].plot(ks,ys,marker='o',color='#0d9488');axes[1].set(ylim=(0,103),xlabel='最高π的前k个分量',ylabel='包含目标中心归属候选 / %',title='圈选候选覆盖（真值事后诊断）')
    families=list(pairs);axes[2].bar(families,[100*pairs[k]['accuracy'] for k in families],color='#7c3aed')
    axes[2].set(ylim=(0,100),ylabel='两份均正确 / %',title='成对画面变化（按场景对计数）');axes[2].tick_params(axis='x',rotation=30)
    for ax in axes:ax.grid(axis='x' if ax is axes[0] else 'y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(out/'metrics.png',dpi=160);fig.savefig(out/'metrics.svg');plt.close(fig)
    closed=next(r for r in readouts if r['kind']=='O2')['per_class']
    labels=sorted(free['O2']['per_class'],key=lambda k:free['O2']['per_class'][k]['accuracy'])
    fig,ax=plt.subplots(figsize=(13,4.5),layout='constrained')
    ax.plot([100*free['O2']['per_class'][k]['accuracy'] for k in labels],label='自由命名',color='#2563eb')
    ax.scatter(range(len(labels)),[100*closed[k]['accuracy'] for k in labels],s=9,alpha=.55,label='闭集读出',color='#ea580c')
    ax.set(xlabel='345个类别（按本次自由准确率排序）',ylabel='类内准确率 / %',ylim=(0,100),title='逐类表现：每类100份原验证图；排序由本次结果决定')
    ax.legend();ax.grid(alpha=.15);fig.savefig(out/'class_accuracy.png',dpi=160);fig.savefig(out/'class_accuracy.svg');plt.close(fig)


def render(out,data_root):
    out=Path(out);data_root=Path(data_root)
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    manifest=json.loads((data_root/'manifest.json').read_text(encoding='utf-8'))
    completion=read('completion.json')
    required=('features','readouts','free','candidates','pairs')
    if not set(required)<=set(completion['phases']):raise ValueError('扩大评估未全部完成，不能生成总体结论页')
    free={task:read(f'free_{task}_summary.json') for task in ('O2','S3','S4','S5')}
    readouts=read('readouts.json');candidates=read('candidates_summary.json');pairs=read('paired_summary.json')
    nobjects=len(manifest['splits']['valid']['objects']);nscenes=len(manifest['splits']['valid']['scene_indices'])
    for task,row in free.items():
        if row['count']!=(nobjects if task=='O2' else nscenes):raise ValueError(f'{task}分母与固定抽样清单不同')
    if candidates['count']!=nscenes:raise ValueError('GMM候选评估分母不同')
    configured_pairs=read('run.json')['arguments']['pairs_per_family']
    if len(pairs)!=5 or any(row['count']!=configured_pairs for row in pairs.values()):raise ValueError('配对评估分母不同')
    details=offline_details(out);plots(out,free,readouts,candidates,pairs)
    def pct(x):return '—' if x is None else f'{x*100:.2f}%'
    def interval(value):return '—' if value is None else ' ～ '.join(pct(x) for x in value)
    def table(headers,rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>扩大样本评估</title>',
        '<style>body{font:16px/1.65 system-ui,"Microsoft Yahei",sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#172033;background:#f1f5f9}section{background:white;padding:24px;margin:20px 0;border-radius:10px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{border:1px solid #dbe2eb;padding:8px;text-align:left}th{background:#f8fafc}a{color:#1d4ed8}summary{cursor:pointer}</style>',
        '<h1>最终54825步模型：扩大样本评估</h1>',
        f'<p>原验证源分层抽取{nobjects:,}份对象原图；覆盖{nscenes:,}份场景配方。正式模型优化更新0，权重hash未变。</p>']
    rows=[]
    for task,name in [('O2','自由写对象类名'),('S3','自由场景问答'),('S4','自由圈选：最近对象正确'),('S5','自由圈内命名')]:
        r=free[task];rows.append([name,r['count'],r['correct'],pct(r['accuracy']),interval(r['ci95_wilson']),r['errors']])
    parts.append('<section><h2>实际自由输出</h2>'+table(['任务','评估清单N','正确','准确率','95%比例区间','执行错误'],rows)+
        '<p>未生成有效答案或圈选的情况计入分母，逐例停止原因在JSONL中。原验证场景配方池全量结果是该池上的实际比例；区间只用于观察向相同生成机制推广的不确定性。</p>'+
        '<p>文字判定沿用已有协议：转小写、标点归为空格并去掉开头a/an/the；类别名要求归一后完全相同。S3的yes/no判首个词，计数把zero至five映射为0至5。自由文字不套345类词典。</p></section>')
    parts.append('<section><img style="width:100%" src="metrics.png" alt="各项准确率、候选覆盖和成对正确率"><p><a href="metrics.svg">导出SVG</a></p></section>')
    parts.append('<section><h2>问答按题型拆分</h2>'+table(['题型','N','正确','准确率','95%区间','真实答案等权平均','验证集多数答案参照'],
        [[k,r['count'],r['correct'],pct(r['accuracy']),interval(r['ci95_wilson']),pct(details['questions'][k]['balanced_accuracy']),pct(details['questions'][k]['validation_majority_accuracy'])] for k,r in free['S3']['by_kind'].items()])+
        '<p>多数答案参照根据当前验证集分布事后计算；冻结读出表另列训练源确定的参照。等权平均把各真实答案的召回率同权平均。</p>'+''.join(
        '<details><summary>'+html.escape(kind)+'：按真实答案拆分</summary>'+table(['真实答案','N','正确','召回率','95%区间','实际输出分布'],
        [[key,row['count'],row['correct'],pct(row['accuracy']),interval(row['ci95_wilson']),json.dumps(row['generated'],ensure_ascii=False)] for key,row in group['by_answer'].items()])+'</details>'
        for kind,group in details['questions'].items())+'</section>')
    rows=[]
    for r in readouts:
        if r['kind']=='S4':
            rows.append([r['kind'],r['train'],r['valid'],f"中心误差 {r['center_error']['mean']:.5f}",f"训练均值 {r['train_mean_baseline']['mean']:.5f}"])
        else:rows.append([r['kind'],r['train'],r['valid'],pct(r['accuracy']),pct(r['majority_accuracy'])])
    parts.append('<section><h2>冻结R后的CPU线性读出</h2>'+table(['任务','拟合N','验证N','结果','训练源多数/均值参照'],rows)+
        '<p>对象读出在345个候选中选择，不能直接当作自由写字的准确率。拟合/验证使用独立原图和场景，正则系数固定为10。</p></section>')
    rows=[]
    for task in ('O2',):
        r=free[task]
        rows.append(['自由对象类名',pct(r['macro_accuracy']),interval(r['macro_ci95_stratified_bootstrap']),pct(r.get('raw_population_weighted_accuracy')),interval(r.get('raw_population_weighted_ci95'))])
    for r in readouts:
        if r['kind']=='O2':rows.append(['闭集对象读出',pct(r['macro_accuracy']),interval(r['macro_ci95_stratified_bootstrap']),pct(r.get('raw_population_weighted_accuracy')),interval(r.get('raw_population_weighted_ci95'))])
    parts.append('<section><h2>类别分层与原池权重</h2>'+table(['输出方式','类别宏平均','分层重采样95%区间','原验证池类别数量加权','加权95%区间'],rows)+
        '<p>类内均匀无放回抽样。比例区间和重采样区间未作有限总体修正，不用它们掩盖每类样本量；逐类N和区间如下。</p></section>')
    closed=next(r for r in readouts if r['kind']=='O2')['per_class'];csv_rows=[]
    for label,r in free['O2']['per_class'].items():
        c=closed[label];csv_rows.append([label,r['count'],r['correct'],r['accuracy'],*r['ci95_wilson'],c['count'],c['correct'],c['accuracy'],*c['ci95_wilson']])
    with (out/'per_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['class','free_n','free_correct','free_accuracy','free_ci_low','free_ci_high','readout_n','readout_correct','readout_accuracy','readout_ci_low','readout_ci_high']);writer.writerows(csv_rows)
    parts.append('<section><details><summary>全部345类：样本量、准确率和区间</summary>'+table(['类别','自由N','自由正确','自由准确率','自由95%区间','闭集正确/N','闭集准确率'],
        [[r[0],r[1],r[2],pct(r[3]),interval(r[4:6]),f'{r[7]}/{r[6]}',pct(r[8])] for r in csv_rows])+'</details><p><a href="per_class.csv">下载逐类CSV</a></p></section>')
    parts.append('<section><img style="width:100%" src="class_accuracy.png" alt="全部类别准确率"><p><a href="class_accuracy.svg">导出SVG</a></p></section>')
    geometry=free['S4']
    parts.append('<section><h2>自由圈选的几何结果</h2>'+table(['量','有几何输出N','均值','中位数'],
        [[name,geometry[key]['count'],f"{geometry[key]['mean']:.5f}" if 'mean' in geometry[key] else '—',f"{geometry[key]['median']:.5f}" if 'median' in geometry[key] else '—'] for name,key in [('中心距离（1×1页面）','center_error'),('圈选框IoU','bbox_iou')]])+
        '<p>最近对象正确只衡量选择中心归属；圈的大小及形状另看几何误差。无几何输出计入上面的准确率分母，均值只基于可测输出。</p>'+
        table(['圈选配方模式（含回退）','N','正确','准确率','95%区间'],[[key,row['count'],row['correct'],pct(row['accuracy']),interval(row['ci95_wilson'])] for key,row in details['circle_by_kind'].items()])+
        '<p>class/position/relation_size是配方选择的模式；不满足位置/关系条件时，问题可能回退到其他描述，实际原文保存在逐例记录中。</p>'+
        table(['对象数','N','正确','准确率','95%区间','均匀选对象参照'],[[key,row['count'],row['correct'],pct(row['accuracy']),interval(row['ci95_wilson']),pct(row['uniform_object_baseline'])] for key,row in details['circle_by_object_count'].items()])+'</section>')
    parts.append('<section><h2>圈选候选分量</h2>'+table(['候选范围','N','至少一个候选中心归属目标','95%区间'],
        [[k,r['count'],pct(r['accuracy']),interval(r['ci95_wilson'])] for k,r in candidates['center_assignment_topk'].items()])+
        '<p>给定目标页控制符，尚未给真实圈选笔画。Top-k中是否包含目标是事后真值诊断，不能作为可部署解码器的正确率；中心归属正确也不等于圈的完整形状正确。</p>'+
        table(['场景对象数','N','Top1','Top4','Top32','目标分量π总和均值'],[[key,row['count'],pct(row['topk']['1']),pct(row['topk']['4']),pct(row['topk']['32']),f"{row['target_mass']:.5f}"] for key,row in details['candidate_by_object_count'].items()])+'</section>')
    parts.append('<section><h2>成对变化</h2>'+table(['类型','独立场景对N','两份都正确','比例','95%区间'],
        [[k,r['count'],r['correct'],pct(r['accuracy']),interval(r['ci95_wilson'])] for k,r in pairs.items()])+
        '<p>同一对的两份变体相关，统计单位使用整对；每组来源在观察输出前固定。</p>'+
        table(['类型','两份均错','仅第一份对','仅第二份对','两份均对','文字答案未变化'],[[key,row['neither_correct'],row['only_first_correct'],row['only_second_correct'],row['both_correct'],row.get('unchanged_answer','—')] for key,row in details['pairs'].items()])+'</section>')
    parts.append('<section><details><summary>S5圈内命名：逐类样本量与结果</summary>'+table(['类别','N','正确','准确率','95%区间'],
        [[key,row['count'],row['correct'],pct(row['accuracy']),interval(row['ci95_wilson'])] for key,row in free['S5']['per_class'].items()])+'</details>'+
        '<p>S5使用全部场景，类别分布来自原配方，逐类样本量并非固定100；整体分数按原场景分布计算。</p></section>')
    parts.append('<section><h2>范围与原始记录</h2><p>这些数字描述本项目345类草图及其合成场景，不能直接代表任意真实图像/场景。'
        '比例区间使用<a href="https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/agrecoul.htm">NIST列出的Wilson公式</a>。'
        '统计区间不弥补来源选择偏差。</p><p><a href="run.json">运行与源清单hash</a> · <a href="readouts.json">完整读出</a> · '
        '<a href="free_O2.jsonl">逐例对象回答</a> · <a href="free_S3.jsonl">逐例问答</a> · <a href="free_S4.jsonl">逐例圈选</a> · '
        '<a href="free_S5.jsonl">逐例圈内识别</a> · <a href="candidates.jsonl">逐例候选</a> · <a href="paired.jsonl">逐对结果</a></p></section></html>')
    (out/'report.html').write_text(''.join(parts),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    args=p.parse_args();render(args.out,args.data)
