"""核验诊断样本覆盖、配对来源、有限值、误差分解及报告的本地链接。"""
from __future__ import annotations

import argparse
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import torch

from scripts.review.render_page_diffusion_diagnostics import TASKS, MODELS, metric_path
from src.st.page_diffusion_codec import file_sha256
from src.st.page_diffusion_runtime import write_json


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]

    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key in ('src','href') and value and not value.startswith(('https:','http:','#','data:','javascript:')):
                self.links.append(unquote(value.split('#')[0]))


def check_finite(value):
    if isinstance(value,torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value,np.ndarray) and value.dtype.kind in 'fc':
        assert np.isfinite(value).all()
    elif isinstance(value,dict):
        for x in value.values():check_finite(x)
    elif isinstance(value,(list,tuple)):
        for x in value:check_finite(x)
    elif isinstance(value,(float,np.floating)):
        assert np.isfinite(value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(4)
    root=args.out.resolve();manifest=json.loads((root/'evaluation_manifest.json').read_text(encoding='utf8'))
    coverage={};decomposition_max=0.;sources={}
    for task in TASKS:
        ref=None
        for model in MODELS:
            path=metric_path(root,model,task,f'valid_{task}.npz')
            a=dict(np.load(path));check_finite(a)
            current=list(zip(a['ids'],a['source']))
            if ref is None:ref=current
            else:assert ref==current,(task,model,'来源不配对')
            for key in ('full','wrong_condition','previous_pages_removed','none'):
                if key not in a:continue
                diff=np.abs(a[key][...,0]-a[key][...,1:].sum(-1))
                decomposition_max=max(decomposition_max,float(diff.max()))
                assert np.allclose(a[key][...,0],a[key][...,1:].sum(-1),rtol=2e-5,atol=2e-7)
            source=json.loads((path.parent/'source.json').read_text(encoding='utf8'))
            assert source['sha256']==manifest['checkpoints'][model]['sha256']
            sources[str(path.relative_to(root))]=source['sha256']
        expected=1024 if task=='T1' else 3450 if task.startswith('O') else 512
        assert len(ref)==expected and len({v[1] for v in ref})==expected
        if task.startswith('O'):
            classes=Counter(a['labels']);assert len(classes)==345 and set(classes.values())=={10}
        coverage[task]=dict(documents=len(ref),sources=len({v[1] for v in ref}))
    selection=json.loads((root/'selection.json').read_text(encoding='utf8'))
    assert len({s['sha256'] for s in selection['stories']})==1024
    coverage['text_parquets']=dict(Counter(s['source_id'].split(':')[0] for s in selection['stories']))
    assert len(coverage['text_parquets'])==4
    for path in (root/'metrics').glob('*/*.npz'):
        check_finite(dict(np.load(path)))
    binary_files=list((root/'metrics').glob('*/*.pt'))+list((root/'metrics_uniform_text').glob('*/*.pt'))
    for path in binary_files:
        check_finite(torch.load(path,map_location='cpu',weights_only=False,mmap=True))
    for task in ('O2','S1','S3','S4','S5'):
        for model in ('parent_visual','final_visual'):
            path=root/'metrics'/model/f'valid_{task}.npz'
            assert path.exists()
    for task in ('T1','S1'):
        assert metric_path(root,'final_raw',task,f'roundtrip_valid_{task}.pt').exists()
    check_finite(json.loads((root/'summary.json').read_text(encoding='utf8')))
    cases=list((root/'panel').glob('*_*/result.json'))
    assert len(cases)==40,len(cases)
    missing=[];links=0
    for path in (root/'report.html',root/'panel/report.html'):
        parser=Links();parser.feed(path.read_text(encoding='utf8'))
        for link in parser.links:
            links+=1
            if not (path.parent/link).exists():missing.append((str(path),link))
    assert not missing,missing
    relevant=[Path(__file__),*Path('scripts/review').glob('page_diffusion_*.py'),
              Path('scripts/review/probe_page_diffusion.py'),Path('scripts/review/render_page_diffusion_diagnostics.py'),
              Path('scripts/review/run_page_diffusion_diagnostics.py'),Path('scripts/review/run_page_diffusion_followup.py'),
              Path('scripts/review/audit_page_diffusion_1b.py'),Path('scripts/checks/check_page_diffusion_probe.py')]
    manifest['diagnostic_code_sha256']={str(p):file_sha256(p) for p in relevant}
    owners=[Path('src/st')/name for name in ('page_diffusion.py','page_diffusion_data.py',
            'page_diffusion_sampling.py','page_diffusion_codec.py','generation_pretrain.py','pipeline_pretrain.py')]
    owners += [Path('src')/name for name in ('formal_scene_tasks.py','page_document.py','text_augmentation.py')]
    owners += [Path('scripts/review/joint_eval_batch.py'),Path('scripts/review/review_page_diffusion.py')]
    manifest['owning_code_sha256']={str(p):file_sha256(p) for p in owners}
    manifest['primary_text_data']='data_uniform_text/valid_T1.pt'
    manifest['primary_text_metrics']='metrics_uniform_text'
    write_json(root/'evaluation_manifest.json',manifest)
    result=dict(coverage=coverage,finite=True,decomposition_max=decomposition_max,
                checkpoint_sources_match=True,panel_results=len(cases),checked_links=links,
                raw_binary_files=len(binary_files),text_sampling_frame=selection['text_sampling_frame'])
    write_json(root/'audit.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
