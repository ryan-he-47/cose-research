"""2026-10-07：仅标准库，抽取少量正式配方与原RAW记录供本机人工审核。"""
from __future__ import annotations

# 2026-10-12：分类目录中的脚本也可从仓库根直接运行。
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))


import argparse
import ast
import json
import math
from pathlib import Path
import struct


def npy_entry(path, index, columns=1):
    # 正式资产为numpy.uint64小端数组；只seek目标行，不加载全池。
    with Path(path).open('rb') as file:
        if file.read(6) != b'\x93NUMPY':
            raise ValueError('非NPY索引')
        major, minor = file.read(2)
        size = struct.unpack('<H' if major == 1 else '<I', file.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(file.read(size).decode('latin1'))
        if header['descr'] not in ('<u8', '=u8') or header['fortran_order']:
            raise ValueError('索引格式不同')
        if index < 0 or index >= header['shape'][0] or math.prod(header['shape'][1:]) != columns:
            raise ValueError('索引越界/维度不同')
        file.seek(index*columns*8, 1)
        return struct.unpack('<'+'Q'*columns, file.read(columns*8))


def record_at(path, offset):
    with Path(path).open('rb') as file:
        file.seek(offset)
        return json.loads(file.readline())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    scene = Path(manifest['scene_index'])/'train'
    raw_index, raw_dir = Path(manifest['raw_index']), Path(manifest['raw_dir'])
    raw_info = json.loads((raw_index/'manifest.json').read_text())['splits']['train']
    recipes = [record_at(scene/'recipes.jsonl',npy_entry(scene/'offsets.npy',i)[0]) for i in range(24)]
    records = {}
    for index in sorted({obj['source_index'] for recipe in recipes for obj in recipe['objects']}):
        file_id, offset = npy_entry(raw_index/'train_offsets.npy',index,2)
        records[str(index)] = record_at(raw_dir/raw_info['files'][file_id],offset)
    stories, story_root = [], Path(manifest['story_index'])
    story_info = json.loads((story_root/'manifest.json').read_text())
    size = story_info['splits']['train']
    multiplier = 104729
    while math.gcd(multiplier,size) != 1:
        multiplier += 2
    for i in range(3):
        entry = npy_entry(story_root/'train.npy',(i*multiplier+manifest['seed'])%size)[0]
        stories.append(record_at(story_root/'stories.jsonl',npy_entry(story_root/'offsets.npy',entry)[0]))
    result = {'source_manifest':manifest,'raw_manifest':raw_info['source'],'recipes':recipes,
              'records':records,'stories':stories,'selection':'前24个正式训练配方、T1前三个实际索引；不筛选美观程度'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'recipes':len(recipes),'drawings':len(records),'stories':len(stories),'bytes':args.output.stat().st_size}))


if __name__ == '__main__':
    main()
