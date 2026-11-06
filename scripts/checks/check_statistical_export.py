"""2026-10-27：有界导出的固定抽样、分片校验与中断恢复，本机小资产检查。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np


def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'tmp_smoke/statistical_export_check_20261027')
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)
    raw=out/'raw';index=out/'index';scenes=out/'scenes';raw.mkdir();index.mkdir();scenes.mkdir()
    for c,label in enumerate(('cat','dog')):
        with (raw/f'{label}.ndjson').open('wb') as f:
            for i in range(40):
                f.write((json.dumps(dict(word=label,key_id=f'{label}-{i}',drawing=[[[.1,.2],[.2,.3]]],recognized=True))+'\n').encode())
    splits={}
    for split,lower,upper in (('train',0,30),('valid',30,40)):
        rows=[]
        for c,label in enumerate(('cat','dog')):
            offsets=[]
            with (raw/f'{label}.ndjson').open('rb') as f:
                while True:
                    offset=f.tell();line=f.readline()
                    if not line:break
                    offsets.append(offset)
            rows.extend((c,offsets[i]) for i in range(lower,upper))
        np.save(index/f'{split}_offsets.npy',np.array(rows,np.uint64));np.save(index/f'{split}_class_ends.npy',np.array([0,upper-lower,2*(upper-lower)],np.uint64))
        splits[split]=dict(source=dict(count=len(rows),classes=['cat','dog']),files=['cat.ndjson','dog.ndjson'])
        folder=scenes/split;folder.mkdir();count=12 if split=='train' else 4
        scene_offsets=[]
        with (folder/'recipes.jsonl').open('wb') as f:
            for i in range(count):
                scene_offsets.append(f.tell());f.write((json.dumps(dict(recipe_id=i,objects=[dict(source_index=i%len(rows))]))+'\n').encode())
        np.save(folder/'offsets.npy',np.array(scene_offsets,np.uint64));write(folder/'manifest.json',dict(count=count))
    write(index/'manifest.json',dict(splits=splits))
    manifest=out/'source.json';write(manifest,dict(raw_index=str(index),raw_dir=str(raw),scene_index=str(scenes),seed=42))
    destination=out/'export'
    command=[sys.executable,str(ROOT/'scripts/review/export_joint_statistical.py'),'--manifest',str(manifest),
             '--out',str(destination),'--objects_per_class','3','--train_scenes','3','--shard_records','2']
    subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
    before=(destination/'manifest.json').read_bytes()
    subprocess.run(command+['--resume'],check=True,stdout=subprocess.DEVNULL)
    after=json.loads((destination/'manifest.json').read_text());previous=json.loads(before)
    assert after['splits']==previous['splits']
    assert after['train_valid_drawing_key_overlap']==0
    for split,data in after['splits'].items():
        assert len(data['objects'])==6
        for shard in data['records_shards']:
            payload=(destination/shard['file']).read_bytes()
            assert hashlib.sha256(payload).hexdigest()==shard['sha256']
            assert len(payload.splitlines())==shard['count']
    # 模拟只完成部分分片：清单保留第一片，其余删去，恢复后固定选样和输出仍相同。
    state=destination/'train_state.json';value=json.loads(state.read_text())
    for shard in value['shards'][1:]:(destination/shard['file']).unlink()
    write(state,dict(shards=value['shards'][:1]))
    subprocess.run(command+['--resume'],check=True,stdout=subprocess.DEVNULL)
    resumed=json.loads((destination/'manifest.json').read_text())
    assert resumed['splits']==previous['splits']
    write(out/'checks.json',dict(fixed_sampling=True,shard_hashes=True,completed_resume_equal=True,partial_resume_equal=True,
                               train_valid_overlap=0,local_fixture_only=True))
    print((out/'checks.json').read_text())


if __name__=='__main__':main()
