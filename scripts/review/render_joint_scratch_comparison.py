"""2026-10-26：汇总从头图文R与初版joint的同源能力、样张与训练曲线。"""
import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.review.review_joint_scratch_stage import read, write
from scripts.review.render_joint_full_review import draw_case, table
from scripts.review.render_joint_full_curves import read_rows, trace


def curves(out, baseline, step):
    sets = dict(
        old=[r for p in (baseline/'cose_r_joint_661550.out', baseline/'cose_r_joint_661737.out')
             for r in read_rows(p) if r.get('success') and r.get('step', 0) <= 65042 and 'update_seconds' in r],
        scratch=[r for p in (out/'cose_r_joint_scratch_674183.out', out/'cose_r_joint_scratch_685064.out')
                 for r in read_rows(p) if r.get('success') and r.get('step', 0) <= step and 'update_seconds' in r])
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    summary = {}
    for ax, task in zip(axes.flat, ('T1','O1','O2','S1','S2','S3','S4','S5')):
        summary[task] = {}
        for name, rows in sets.items():
            series = trace(rows, 'task_nll.'+task, bin_size=50000000)
            ax.plot(series['x'], series['y'], label=name, lw=1.4)
            last = [r['task_nll'][task] for r in rows[-100:] if task in r.get('task_nll', {})]
            summary[task][name] = dict(last_100_logged_median=statistics.median(last), logged_updates=len(rows))
        ax.set_title(task);ax.set_xlabel('Joint-phase inputs / million');ax.set_ylabel('NLL');ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.suptitle('Joint training: 50M-input median; old model had prior text training')
    fig.tight_layout();fig.savefig(out/'task_nll_comparison.png', dpi=140);plt.close(fig)
    write(out/'training_comparison.json', summary)
    return summary


def images(out, baseline, labels=('old','scratch')):
    drawing_baseline = out/'baseline_replay' if (out/'baseline_replay/joint_summary.json').exists() else baseline
    cases = [c for c in read(out/'joint_summary.json')['cases'] if c['task']=='O1']
    for mode in ('greedy', 'sample'):
        fig, axes = plt.subplots(6, 4, figsize=(13, 18))
        for i, case in enumerate(cases):
            for j, (name, path) in enumerate(((labels[0], drawing_baseline), (labels[1], out))):
                ax = axes[i//2, (i%2)*2+j]
                draw_case(ax, read(path/case['name']/(mode+'.json'))['strokes'])
                ax.set_title(case['label']+' / '+name)
        fig.suptitle('Same prompt / '+mode+' / generated strokes only')
        fig.tight_layout();fig.savefig(out/('objects_comparison_'+mode+'.png'), dpi=135);plt.close(fig)
    cases = [c for c in read(out/'joint_summary.json')['cases'] if c['task']=='S2']
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 3.6*len(cases)), squeeze=False)
    for i, c in enumerate(cases):
        ref = [dict(r, generated=True) for r in read(out/c['name']/'reference.json')['strokes'] if r.get('targetable')]
        values = [('reference', ref), (labels[0], read(drawing_baseline/c['name']/'greedy.json')['strokes']),
                  (labels[1], read(out/c['name']/'greedy.json')['strokes'])]
        for ax, (name, strokes) in zip(axes[i], values):
            draw_case(ax, strokes);ax.set_title(c['name']+' / '+name)
    fig.tight_layout();fig.savefig(out/'scenes_comparison.png', dpi=135);plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    from src.text_renderer import VectorFont
    from scripts.review.review_joint_stage import dataset_for
    font = VectorFont.from_svg(ROOT/'datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg')
    dataset = dataset_for(read(out/'probe_bundle.json'), 'valid', font)
    for i, ax in enumerate(axes.flat):
        value = read(out/f'localization_{i:02d}.json')['result']
        ax.set(xlim=(0,1), ylim=(1,0), aspect='equal', xticks=[], yticks=[])
        for r in value['strokes']:
            if not r['generated'] and r['kind']=='sketch':
                ax.plot(*np.asarray(r['points']).T, color='#bbb', lw=.8)
        location_baseline = out/'baseline_replay' if (out/'baseline_replay/localization.json').exists() else baseline
        old = read(location_baseline/f'localization_{i:02d}.json')
        for e in dataset.get_for_task('S4', i).to_sequence():
            if e.targetable and e.kind=='selection':
                ax.plot(*e.rawxy.T, color='#399c63', lw=1, ls='--')
        for name, item, color in ((labels[0], old, '#3377bb'), (labels[1], read(out/f'localization_{i:02d}.json'), '#e47728')):
            for r in item['result']['strokes']:
                if r['generated'] and r['kind'] not in ('bos','eod','eop'):
                    ax.plot(*np.asarray(r['points']).T, color=color, lw=1.3)
        a = old['metrics']['nearest_object_correct']
        b = read(out/f'localization_{i:02d}.json')['metrics']['nearest_object_correct']
        ax.set_title(f'{i}: {labels[0]}={a}, {labels[1]}={b}', fontsize=10)
    fig.suptitle(f'S4 / gray: input; green: target; blue: {labels[0]}; orange: {labels[1]}')
    fig.tight_layout();fig.savefig(out/'localization_comparison.png', dpi=135);plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(11, 18))
    for prompt in range(2):
        for col, (name, directory) in enumerate(((labels[0], baseline), (labels[1], out))):
            strokes = read(directory/'text'/f'gmm{prompt}.json')['strokes']
            generated = [r for r in strokes if r['generated'] and r['kind'] not in ('bos','eop','eod')]
            pages = sorted({r['page_id'] for r in generated})[:2]
            for j in range(2):
                ax = axes[prompt*2+j,col]
                records = [r for r in generated if j < len(pages) and r['page_id']==pages[j]]
                draw_case(ax, records);ax.set_title(f'T1 prompt {prompt} / body page {j+1} / {name}')
    fig.suptitle('Direct GMM feedback / first two generated pages')
    fig.tight_layout();fig.savefig(out/'text_gmm_comparison.png', dpi=130);plt.close(fig)


def paired(old, new, keys=('index',)):
    def key(row):return tuple(row[k] for k in keys)
    a, b = {key(r):r for r in old}, {key(r):r for r in new}
    assert a.keys() == b.keys()
    return [(a[k], b[k]) for k in a]


def transitions(pairs):
    return dict(both_correct=sum(a['correct'] and b['correct'] for a,b in pairs),
                improved=sum(not a['correct'] and b['correct'] for a,b in pairs),
                regressed=sum(a['correct'] and not b['correct'] for a,b in pairs),
                both_wrong=sum(not a['correct'] and not b['correct'] for a,b in pairs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT/'tmp_smoke/r_joint_full_review_20261022')
    args = parser.parse_args();out, baseline = args.out.resolve(), args.baseline.resolve()
    audit = read(out/'audit.json')
    recipe_comparison = read(out/'recipe_comparison.json') if (out/'recipe_comparison.json').exists() else dict(
        old=audit.get('comparison_cfg', {}), scratch=audit['cfg'])
    teachers = read(out/'teacher_comparison.json')
    text = [read(p/'text/summary.json') for p in (baseline, out)]
    recognition = [read(p/'recognition.json') for p in (baseline, out)]
    scenes = [read(p/'scene_abilities.json') for p in (baseline, out)]
    localization = [read(p/'localization.json') for p in (baseline, out)]
    replay = read(out/'baseline_runtime_replay.json') if (out/'baseline_runtime_replay.json').exists() else None
    if replay:
        for i, row in enumerate([r for r in replay['cases'] if r['task']=='T1']):
            text[0]['cases'][i] = row['case']
        for filename, reports in (('recognition',recognition),('scene_abilities',scenes),('localization',localization)):
            if (out/'baseline_replay'/(filename+'.json')).exists():
                reports[0] = read(out/'baseline_replay'/(filename+'.json'))
    text_rows = []
    for name, item in zip(('初版joint', '从头图文'), text):
        r = item['teacher_forcing'];a = r['text_aux']
        text_rows.append([name, f"{a['all']['shape_ce']:.4f}", f"{1-a['all']['shape_error']:.2%}",
                          f"{1-a['first']['shape_error']:.2%}", f"{1-a['continuation']['shape_error']:.2%}",
                          f"{1-a['all']['space_error']:.2%}", f"{r['content_nll']:.2f}"])
    bundle = read(out/'probe_bundle.json')
    balanced = set(bundle['splits']['valid']['balanced_indices'])
    recognition_pairs = paired(recognition[0]['cases'], recognition[1]['cases'])
    recognition_pairs = [(a,b) for a,b in recognition_pairs if a['index'] in balanced]
    qa_pairs = paired(scenes[0]['cases'], scenes[1]['cases'], ('task','index'))
    # 参考答案与问题逐项核对，确保自由输出来自相同条件。
    assert all(a['expected'] == b['expected'] and a['question'] == b['question'] for a,b in qa_pairs)
    groups = defaultdict(list)
    for a,b in qa_pairs:groups[(a['task'], a['query_kind'])].append((a,b))
    query_rows = []
    distributions = {}
    for (task, kind), values in groups.items():
        counts = [Counter(r['generated'] for r in (pair[i] for pair in values)) for i in (0,1)]
        query_rows.append([task+'/'+kind, len(values), sum(a['correct'] for a,b in values),
                           sum(b['correct'] for a,b in values), counts[0].most_common(3), counts[1].most_common(3)])
        distributions[task+'/'+kind] = [dict(c) for c in counts]
    location_pairs = paired(localization[0]['cases'], localization[1]['cases'])
    location_transition = transitions([(dict(a, correct=a['nearest_object_correct']), dict(b, correct=b['nearest_object_correct']))
                                       for a,b in location_pairs])
    location_rows = []
    for name, report in zip(('初版joint', '从头图文'), localization):
        rows = report['cases']
        location_rows.append([name, report['nearest_object_correct'], f"{np.mean([r['center_error'] for r in rows]):.4f}",
                              f"{report['median_center_error']:.4f}", f"{np.mean([r['bbox_iou'] for r in rows]):.4f}",
                              f"{report['median_bbox_iou']:.4f}"])
    graph, first = [], []
    for name, rows in ((n,teachers[k]) for n,k in (('初版joint','old'),('从头图文','scratch'))):
        for r in rows:
            if r['task'] in ('O1','S2','S4') and r['kind'] in ('sketch','selection'):
                graph.append([name, r['split'], r['task'], r['count'], f"{r['nll']:.3f}", f"{r['greedy_latent_mse']:.5f}",f"{r['sigma']:.4f}"])
            if r['split']=='valid' and r['task'] in ('O2','S1','S3','S5') and r['kind'] in ('text','answer_initial'):
                first.append([name,r['task'],r['kind'],r['count'],f"{r['shape_ce']:.4f}",f"{r['shape_accuracy']:.2%}"])
    summary = dict(step=audit['step'], successful_inputs=audit['successful_inputs'],
                   old_step=audit['comparison_step'], old_joint_inputs=audit['comparison_joint_inputs'],
                   recognition=[r['balanced'] for r in recognition], recognition_transitions=transitions(recognition_pairs),
                   scene_metrics=[r['metrics'] for r in scenes], answer_distributions=distributions,
                   localization=location_rows, localization_transitions=location_transition,
                   text_metrics=text_rows, teacher_metrics=teachers, optimizer_updates=0,
                   recipe_comparison=recipe_comparison,
                   baseline_runtime_replay=replay,
                   frozen_ae_equal=audit['frozen_ae_equal'], weights_unchanged=True,
                   decode=dict(curves='greedy highest-pi mean / sample temperature .5, seed42',
                               text='shape/space prefix, beam1, canonical font; no optional candidate reranking'))
    curves(out, baseline, audit['step']);images(out, baseline)
    write(out/'comparison_summary.json', summary)
    css = 'body{font:16px/1.65 system-ui;margin:28px;color:#24334b;max-width:1500px}table{border-collapse:collapse;margin:15px 0}td,th{border:1px solid #ccd5df;padding:6px 10px;max-width:580px;vertical-align:top}th{background:#edf2f7}img{max-width:100%}pre{white-space:pre-wrap;background:#f2f5f8;padding:15px}h2{margin-top:40px;border-top:2px solid #dde4ed;padding-top:20px}.scroll{overflow:auto}'
    page = f'<!doctype html><meta charset="utf-8"><title>从头图文50k：初版joint对照</title><style>{css}</style><h1>从头图文第{audit["step"]}步：初版joint对照</h1>'
    page += f'<p>2026-10-26；新模型{audit["successful_inputs"]:,}成功输入，初版joint第65042累计步/36340图文步、{audit["comparison_joint_inputs"]:,}图文输入，另有此前文字训练。两者使用逐项一致的冻结55k AE。服务器训练继续运行，本次使用固定检查点在本机CUDA/FP16推理。</p>'
    page += '<p>沿用旧检查的原始草图、96个场景、128篇文字valid及四个前文。曲线以图文阶段输入为横轴；总损失含动态辅助权重，能力对照以同条件输出及分项为准。文字使用原形状/空格头与字体笔顺闭环，曲线保持原greedy/sample设置。</p>'
    old_alpha = recipe_comparison['old'].get('r_text_aux_weight')
    new_alpha = recipe_comparison['scratch'].get('r_text_aux_weight')
    if old_alpha is not None and new_alpha is not None:
        page += f'<p>R的σ下界均为0.003/quadrature；当前动态辅助α分别为旧{old_alpha:.2f}、新{new_alpha:.2f}，因此总loss接近时，分类与几何分项仍可能朝不同方向变化。</p>'
    overview = [
        ['文字形状准确率', text_rows[0][2], text_rows[1][2]],
        ['O2完整类别/345', *[r['balanced']['correct'] for r in recognition]],
        ['S3场景问答/96', *[r['metrics']['S3']['correct'] for r in scenes]],
        ['S5圈内识别/96', *[r['metrics']['S5']['correct'] for r in scenes]],
        ['S4圈选正确/96', *[r['nearest_object_correct'] for r in localization]],
        ['S1描述类名词面召回', *[f"{r['metrics']['S1']['class_recall']:.2%}" for r in scenes]],
    ]
    page += '<h2>主要差异</h2>'+table(('观察项','初版joint','从头图文'),overview)
    page += '<p>文字分类与部分图文对应改善，长文字仍有错拼和重复；存在性仍全部答yes，计数仍全部答1。圈选NLL下降，但闭环目标选择退步。描述的类名词面召回只看是否提到真实类名，位置、数量、拼写与句子质量需结合原文查看。</p>'
    page += '<h2>文字逐笔指标与自由续写</h2>'+table(('模型','形状CE','形状准确率','字符首笔','字符续接','空格准确率','正文NLL'), text_rows)
    for i, old in enumerate(text[0]['cases'][:4]):
        new = text[1]['cases'][i]
        page += '<h3>'+html.escape(old['prompt'])+'</h3>'
        for name,item in (('初版joint',old),('从头图文',new)):
            pages = item['generated_page_texts'];body = '\n'.join(v for k,v in sorted(pages.items(), key=lambda x:int(x[0])) if int(k)>0)
            page += '<b>'+name+'</b><pre>'+html.escape(body)+'</pre>'
    page += '<p><a href="text/index.html">新模型完整文字/GMM样张</a></p><details><summary>直接GMM回馈的文字几何</summary><img src="text_gmm_comparison.png"></details><h2>对象绘制</h2><p>每两幅为同一条件的初版joint与从头图文。</p><img src="objects_comparison_greedy.png"><details><summary>同条件采样</summary><img src="objects_comparison_sample.png"></details>'
    page += '<h2>场景绘制</h2><img src="scenes_comparison.png"><details><summary>同源图形教师强制</summary>'+table(('模型','分区','任务','笔画','NLL','最高π均值MSE','最高πσ'),graph)+'</details>'
    page += '<h2>自由识图</h2>'+table(('模型','345类正确','比例'),[[n,r['balanced']['correct'],f"{r['balanced']['correct']/345:.2%}"] for n,r in zip(('初版joint','从头图文'),recognition)])
    page += '<p>逐例变化：'+html.escape(str(summary['recognition_transitions']))+'</p><details><summary>345类逐例回答</summary>'+table(('真值','初版joint','从头图文','旧正确','新正确'),[[a['expected'],a['generated'],b['generated'],a['correct'],b['correct']] for a,b in recognition_pairs])+'</details>'
    page += '<h2>场景问答、描述与圈内识别</h2>'+table(('任务/类型','题数','旧正确','新正确','旧常见回答','新常见回答'),query_rows)
    page += '<details><summary>逐例问答和描述</summary>'+table(('任务','问题','真值','初版joint','从头图文'),[[a['task'],a['question'],a['expected'],a['generated'],b['generated']] for a,b in qa_pairs])+'</details>'
    page += '<h2>圈选</h2>'+table(('模型','选对/96','中心均值','中心中位数','bbox IoU均值','bbox IoU中位数'),location_rows)
    page += '<p>逐例变化：'+html.escape(str(location_transition))+'</p><img src="localization_comparison.png">'
    page += '<h2>回答开头与后续笔画</h2>'+table(('模型','任务','分桶','笔数','形状CE','准确率'),first)
    page += '<h2>训练分项曲线</h2><img src="task_nll_comparison.png"><p>每5000万成功输入取中位数；旧模型已有文字预训练。原日志和JSON位于本目录。</p>'
    if replay:
        short = [r for r in replay['cases'] if r['task']!='T1']
        long = [r for r in replay['cases'] if r['task']=='T1']
        page += f'<p>旧模型以当前推理实现重放了12类对象×两种模式、4场景×两种模式、96圈选及四组长续写。另抽取的{len(short)}个对象/场景文字回答中，{sum(r["same_text"] for r in short)}个与原记录逐字一致；四组长续写有{sum(r["same_text"] for r in long)}组逐字一致。对照图和文字引用采用重放结果，整批旧识图/问答分数及教师强制指标复用原始记录。</p>'
    page += '<p><a href="comparison_summary.json">机器可读汇总</a> · <a href="teacher_comparison.json">同源教师强制分项</a> · <a href="audit.json">检查点核对</a></p>'
    (out/'report.html').write_text(page, encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('step','recognition','recognition_transitions','localization_transitions','text_metrics')},ensure_ascii=False))


if __name__ == '__main__':
    main()
