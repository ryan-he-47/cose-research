"""2026-10-20：仅标准库，定点导出真实train/valid对象和场景RAW，供本机阶段验收。"""
import argparse
import json
import math
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from scripts.review.export_formal_review import npy_entry, record_at
except ModuleNotFoundError:
    # 固定旧工作树不Apply时，与辅助脚本一起从Git对象放在运行目录执行。
    from export_formal_review import npy_entry, record_at


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--all_classes',action='store_true',help='覆盖全部345类，并保留旧32例锚点')
    parser.add_argument('--scenes',type=int,default=12,help='场景配方数；含前12例锚点')
    args=parser.parse_args()
    if args.scenes<1:parser.error('场景数须为正')
    manifest=json.loads(args.manifest.read_text())
    index=Path(manifest['raw_index']);raw=Path(manifest['raw_dir'])
    source=json.loads((index/'manifest.json').read_text())
    result=dict(source_manifest=manifest,object_count=32,scene_count=args.scenes,splits={})
    for split in ('train','valid'):
        info=source['splits'][split];size=info['source']['count'];classes=info['source']['classes']
        multiplier=104729
        while math.gcd(multiplier,size)!=1:multiplier+=2
        samples=[]
        sample_labels=('cat','fish','airplane','bicycle','dog','bird','car','house','tree','flower','chair','clock') if args.all_classes else ('cat','fish','airplane','bicycle')
        for label in sample_labels:
            raw_index=npy_entry(index/f'{split}_class_ends.npy',classes.index(label))[0]
            i=((raw_index-manifest['seed'])*pow(multiplier,-1,size))%size
            samples.append(dict(label=label,index=i))
        balanced=[]
        if args.all_classes:
            rng=random.Random(20261022)
            for c in range(len(classes)):
                start=npy_entry(index/f'{split}_class_ends.npy',c)[0]
                stop=npy_entry(index/f'{split}_class_ends.npy',c+1)[0]
                raw_id=start+rng.randrange(stop-start)
                balanced.append(((raw_id-manifest['seed'])*pow(multiplier,-1,size))%size)
        scene=Path(manifest['scene_index'])/split
        scene_info=json.loads((scene/'manifest.json').read_text())
        if args.scenes>scene_info['count']:raise ValueError('请求场景数超过配方池')
        scene_indices=list(range(min(12,args.scenes)))
        cursor=0
        while len(scene_indices)<args.scenes:
            value=(cursor*104729+manifest['seed'])%scene_info['count'];cursor+=1
            if value not in scene_indices:scene_indices.append(value)
        recipe_list=[record_at(scene/'recipes.jsonl',npy_entry(scene/'offsets.npy',i)[0]) for i in scene_indices]
        recipes=dict(zip(map(str,scene_indices),recipe_list)) if args.all_classes or args.scenes!=12 else recipe_list
        ids={(i*multiplier+manifest['seed'])%size for i in [*range(32),*balanced,*(s['index'] for s in samples)]}
        ids.update(obj['source_index'] for recipe in recipe_list for obj in recipe['objects'])
        records={}
        for i in sorted(ids):
            file_id,offset=npy_entry(index/f'{split}_offsets.npy',i,2)
            records[str(i)]=record_at(raw/info['files'][file_id],offset)
        result['splits'][split]=dict(raw_manifest=info['source'],recipes=recipes,
            recipe_count=scene_info['count'],records=records,samples=samples,
            object_indices=list(dict.fromkeys([*range(32),*balanced])),
            balanced_indices=balanced,anchor_indices=list(range(32)),scene_indices=scene_indices)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(bytes=args.output.stat().st_size,
        drawings={k:len(v['records']) for k,v in result['splits'].items()})))


if __name__=='__main__':main()
