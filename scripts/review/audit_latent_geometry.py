"""核对已生成几何评估的来源、覆盖、数组和报告链接；不调用模型。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha256(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4<<20),b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out
    source=load(out/'review_source.json')
    report=dict(source=source,corpora={},checks=[])
    code_paths=[Path(__file__),Path(__file__).with_name('review_latent_geometry.py'),
                Path(__file__).with_name('render_latent_geometry.py'),Path(__file__).with_name('export_latent_geometry.py'),
                *Path(__file__).parent.glob('latent_geometry_*.py')]
    report['analysis_code_sha256']={p.name:sha256(p) for p in code_paths}
    report['checkout_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    report['library_versions']={name:version(name) for name in ('numpy','scipy','scikit-learn','torch','matplotlib')}
    normal=load(Path(source['files']['normalization']['path']))
    mean=np.asarray(normal['mean'],np.float32);scale=normal['scale']
    for name in ('ae','r','text','validation'):
        path=out/'data'/f'{name}_pipeline.npz'
        manifest=load(path.with_suffix('.json'))
        encoding=load(out/f'{name}_encoding.json')
        digest=sha256(path)
        assert digest==manifest['sha256']==encoding['source_sha256'],name
        for key in ('ae','normalization'):
            assert encoding[f'{key}_sha256']==source['files'][key]['sha256'],name
        with np.load(path) as raw,np.load(out/f'{name}_encoded.npz') as encoded:
            n=len(encoded['z']);lengths=np.diff(raw['offsets'])
            assert n==manifest['kept']==len(lengths)==encoding['n']
            assert np.isfinite(raw['points']).all() and 0<lengths.min()<=lengths.max()<=76
            assert raw['offsets'][0]==0 and raw['offsets'][-1]==len(raw['points'])
            for key in ('z','latent','z_fp32'):
                assert np.isfinite(encoded[key]).all(),(name,key)
            for key in ('kind','group','unit','task','target','window','label'):
                assert np.array_equal(raw[key],encoded[key]),(name,key)
            error=float(np.max(np.abs(encoded['z']-(encoded['latent']-mean)/scale)))
            assert error<1e-6,(name,error)
            row=dict(n=n,sha256=digest,normalization_max_error=error,
                     minimum_points=int(lengths.min()),maximum_points=int(lengths.max()),
                     counts=dict(Counter(encoded['kind'])),
                     source_units={str(g):len(np.unique(encoded['unit'][encoded['group']==g])) for g in np.unique(encoded['group'])})
            if name=='r':
                body=encoded['target']&~np.isin(encoded['kind'],['bos','eop','eod'])
                row['body_targets']=dict(Counter(encoded['kind'][body]))
            if name=='validation':
                counts=Counter(encoded['label'][encoded['group']=='object'])
                assert len(counts)==345 and set(counts.values())=={100}
            report['corpora'][name]=row
    with np.load(out/'glyphs.npz') as glyph:
        assert len(glyph['characters'])==94
        for split,count in ((0,128),(1,256)):
            assert set(Counter(glyph['labels'][glyph['splits']==split]).values())=={count}
        for key in ('latent','varied','raw'):
            assert np.isfinite(glyph[key]).all()
        report['glyphs']=dict(classes=94,reference_instances=12032,query_instances=24064,
                             total_strokes=int(glyph['parts'].sum()))
    smooth=load(out/'smoothness.json')
    assert set(smooth['groups'])=={'object','scene','text'}
    for name,group in smooth['groups'].items():
        assert group['n']==group['unique_sources']==1024
        for direction in group['perturbations'].values():
            for value in direction.values():
                assert value['rms_decomposition_max_error']<1e-7,(name,value)
    controls=load(out/'control_geometry.json')
    for time,rates in controls['noise'].items():
        for name,rate in rates.items():
            low,high=controls['noise_ci95'][time][name]
            assert 0<=low<=rate+1e-12 and rate-1e-12<=high<=1
    report['checks']+=['源数据SHA256与编码来源一致','已保存归一化逐元素一致','全部输入保留变长且不超过76点',
                      '345类验证对象每类100个','94类字形每类128参考和256查询',
                      '三类平滑性各1024个来源','曲线变化的平移/形状平方分解成立']

    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            self.links.extend(v for k,v in attrs if k in ('href','src') and v)
    if (out/'report.html').exists():
        page=Links();page.feed((out/'report.html').read_text(encoding='utf8'))
        local=[v for v in page.links if not v.startswith(('http:','https:','#','data:'))]
        missing=[v for v in local if not (out/v.split('#')[0]).exists()]
        assert not missing,missing
        from PIL import Image
        for file in out.glob('*.png'):
            with Image.open(file) as picture:picture.verify()
        report['checks'].append('HTML本地链接存在、全部PNG可解码')
    report['passed']=True
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(passed=True,corpora={n:d['n'] for n,d in report['corpora'].items()},checks=len(report['checks']))))


if __name__=='__main__':main()
