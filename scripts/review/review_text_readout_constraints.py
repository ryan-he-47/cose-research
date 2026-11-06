"""2026-10-13：本机仅推理；共享前缀笔顺约束与原文字读出四前文并排。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import html
import json
import time
import torch
from scripts.demo.demo_text_readout import load_readout
from scripts.demo.demo_pretrain_generation import save_native_svg
from scripts.review.train_text_readout import generate,write
from scripts.review.diagnose_formal_r_health import model_hash
from scripts.review.render_formal_r_review import svg_pages
from scripts.review.render_r_dual_review import plot_page
from src.st.text_readout_policy import audit_font_sequence


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def concise(value):
    result={k:v for k,v in value.items() if k not in ('strokes','constraint_trace')}
    if 'font_audit' in result:result['font_audit']={k:v for k,v in result['font_audit'].items() if k!='details'}
    return result


def render(root,baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=read(root/'summary.json')
    totals={key:sum(row['policy'][key] for row in summary) for key in
        ['steps','forced_steps','raw_illegal_steps','ambiguous_starts','shared_stroke_steps',
         'identity_revisions','grouped_choice_overrides']}
    totals['forced_rate']=totals['forced_steps']/totals['steps']
    totals['blocked_rate']=totals['raw_illegal_steps']/totals['steps']
    write(root/'totals.json',totals)
    fig,axes=plt.subplots(4,3,figsize=(11,14),squeeze=False)
    cases=[];table=[]
    for index,item in enumerate(summary):
        values={'constrained':read(root/f'prompt{index}.json'),'unconstrained':read(baseline/f'prompt{index}.json')}
        values['gmm']=read(Path('tmp_smoke/r_t1_final_review_20261013/t1')/f'T1_prompt{index}'/'greedy.json')
        for col,key in enumerate(['unconstrained','constrained','gmm']):
            plot_page(axes[index,col],values[key]['strokes']);axes[index,col].set_title(f'prompt{index} / {key}')
        panels={k:svg_pages([r for r in value['strokes'] if r.get('generated')]) for k,value in values.items()}
        panels['condition']=svg_pages([r for r in values['constrained']['strokes'] if not r.get('generated')])
        cases.append(dict(index=index,prompt=item['prompt'],panels=panels,
            constrained=concise(values['constrained']),unconstrained=concise(values['unconstrained'])))
        policy=item['policy']
        cells=[index,item['old_audit']['violating_steps'],item['new_audit']['violating_steps'],
               f"{policy['raw_illegal_steps']}/{policy['steps']}",policy['forced_steps'],
               policy['ambiguous_starts'],policy['identity_revisions'],item['stop_reason']]
        table.append('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in cells)+'</tr>')
    fig.tight_layout();fig.savefig(root/'generation.png',dpi=150);plt.close(fig)
    encoded=json.dumps(cases,ensure_ascii=False).replace('</',r'<\/')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>字体笔顺约束：冻结R自回归</title><style>body{font:16px system-ui,"Microsoft YaHei";color:#243442;background:#f3f5f7;max-width:1550px;margin:20px auto;padding:20px}td,th{border:1px solid #ccd;padding:9px}table{border-collapse:collapse}pre{white-space:pre-wrap;overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.panel{padding:12px;background:white;border:1px solid #ddd}svg{width:100%}select{font:inherit;padding:10px}@media(max-width:950px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>
<h1>字体笔顺约束下的自回归</h1><p>同一R、同一读出头与四个前文，仅添加字体及控制符约束。第一笔相同时保留多个字符，后续分岔再确定；短字终点可与长字后续笔竞争。无外部词典。</p>
<p>每个字从第0笔开始，未完成字形时仅允许候选字的后续笔；BOS不能生成，EOD后必须EOP，空页不能结束或再次翻页。相同空格位、实际可见曲线和点数相同的类别概率相加。</p>
<p><a href="generation.png">首个生成页概览</a> · <a href="summary.json">完整摘要</a> · <a href="completion.json">冻结来源核对</a></p>
<table><tr><th>前文</th><th>旧违规步</th><th>新违规步</th><th>原argmax非法/总笔数</th><th>唯一合法分支</th><th>共享首笔次数</th><th>延迟改字次数</th><th>停止</th></tr>TABLE</table>
<p>违规步由最终笔画序列独立校验，包含跳笔及提前中断字符。旧版的invalid_part_transitions只统计一部分跳笔，因此旧违规总数会更大。</p>
<select id="case"></select><p id="prompt"></p><div class="grid"><section class="panel"><h3>字体约束</h3><div id="constrained"></div></section><section class="panel"><h3>原无约束读出</h3><div id="unconstrained"></div></section><section class="panel"><h3>原GMM最高π</h3><div id="gmm"></div></section><section class="panel"><h3>约束版实际前文</h3><div id="condition"></div></section></div>
<details open><summary>约束版实际文字与介入次数</summary><pre id="metrics"></pre></details><p>追加字导致的旧排版回流仍逐例记录并刷新KV。笔数上限时先完整结束当前字；共享前缀里有完整短字时可定为该字，最后这一处计入cap_finalizations。</p>
<script>const data=DATA;const el=id=>document.getElementById(id);data.forEach(c=>{let o=document.createElement('option');o.value=c.index;o.textContent='prompt'+c.index;el('case').appendChild(o)});function render(){let c=data[+el('case').value];el('prompt').textContent=c.prompt;Object.keys(c.panels).forEach(k=>el(k).innerHTML=c.panels[k]);el('metrics').textContent=JSON.stringify(c.constrained,null,2)}el('case').onchange=render;render()</script></html>'''
    (root/'index.html').write_text(page.replace('TABLE',''.join(table)).replace('DATA',encoded),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head',type=Path,default=Path('tmp_smoke/r_text_readout_20261013/head_final.pt'))
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--baseline',type=Path,default=Path('tmp_smoke/r_text_readout_20261013'))
    parser.add_argument('--out',type=Path,default=Path('tmp_smoke/r_text_readout_constrained_20261013'))
    parser.add_argument('--generation_strokes',type=int,default=480)
    parser.add_argument('--max_seconds',type=float,default=900)
    parser.add_argument('--render_only',action='store_true')
    args=parser.parse_args()
    if args.render_only:
        render(args.out,args.baseline);print(str(args.out/'index.html'));return
    if args.max_seconds<=0 or args.generation_strokes<1:parser.error('上限必须为正')
    if not torch.cuda.is_available():parser.error('本次仅在本机CUDA/FP16推理')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'requested.json').exists():raise FileExistsError('已有实验记录，请用新目录')
    write(args.out/'requested.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    args.constraint='prefix';torch.set_num_threads(4)
    started=time.monotonic();deadline=started+args.max_seconds
    generator,head,vocab,layout,state=load_readout(args.head,args.checkpoint)
    head_before=model_hash(head)
    summary=[]
    for index in range(4):
        old=read(args.baseline/f'prompt{index}.json')
        value=generate(generator,head,vocab,old['prompt'],deadline,args,layout)
        new_audit=value['font_audit']
        if new_audit['violations'] or new_audit['unfinished_character'] or new_audit['pending_eod']:
            write(args.out/'invalid_generation.json',value)
            raise AssertionError('约束结果未通过独立完整字形/控制符审计')
        write(args.out/f'prompt{index}.json',value)
        save_native_svg(value,args.out/f'prompt{index}.svg',f'Font prefix constraint / prompt{index}')
        old_audit=audit_font_sequence(old['strokes'],vocab)
        write(args.out/f'prompt{index}_old_audit.json',old_audit)
        item=dict(index=index,prompt=old['prompt'],stop_reason=value['stop_reason'],
            generated_strokes=value['generated_strokes'],text_pages=value['generated_page_texts'],
            old_audit={k:v for k,v in old_audit.items() if k!='details'},
            new_audit={k:v for k,v in new_audit.items() if k!='details'},
            policy=value['constraint_summary'],layout_refreshes=value['layout_refreshes'],
            conditioning_refreshes=value['conditioning_refreshes'])
        summary.append(item);write(args.out/'summary.json',summary)
        print(json.dumps(item,ensure_ascii=False),flush=True)
    after=model_hash(generator.model);head_after=model_hash(head)
    assert after==state['r_model_hash'] and head_before==head_after
    write(args.out/'completion.json',dict(complete=True,seconds=time.monotonic()-started,
        R_unchanged=True,head_unchanged=True,R_hash=after,head_hash=head_after,
        head_file_sha256=hashlib.sha256(args.head.read_bytes()).hexdigest(),
        head_source=str(args.head.resolve()),inference_only=True,precision='FP16',gpu='local',
        constraint='glyph_prefix_with_delayed_identity',uses_dictionary=False))
    render(args.out,args.baseline);print(str(args.out/'index.html'),flush=True)


if __name__=='__main__':main()
