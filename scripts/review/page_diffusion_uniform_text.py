"""文字诊断使用四个分片的全部行范围；CPU准备与GPU编码分开。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts.review.page_diffusion_probe_data import valid_stories,encode_documents,SEED
from src.st.pipeline_pretrain import PretrainDocumentDataset
from src.st.page_diffusion_codec import FrozenPageCodec
from src.st.generation_pretrain import PageGenerator
from src.st.page_diffusion_runtime import write_json
from src.text_renderer import VectorFont
from src.text_augmentation import TextAugmentationConfig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','encode'),required=True);a=p.parse_args()
    out=a.out/'data_uniform_text';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);start=time.monotonic()
    if a.phase=='prepare':
        stories=valid_stories(Path('E:/Trash_things/cose/datasets/datasets--roneneldan--TinyStories/snapshots'),1024)
        font=VectorFont.from_svg('E:/Trash_things/cose/datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg')
        ds=PretrainDocumentDataset(font,stories,seed=42,split='valid',augmentation=TextAugmentationConfig(enabled=True))
        rng=np.random.default_rng(SEED);documents=[]
        for i in range(len(stories)):
            windows=list(ds.get_for_task('T1',i).iter_windows(1024))
            documents.append(windows[int(rng.integers(len(windows)))])
            if (i+1)%128==0:print(json.dumps(dict(stage='uniform_text_prepare',done=i+1,seconds=time.monotonic()-start)),flush=True)
        torch.save(documents,out/'documents.pt')
        write_json(out/'selection.json',dict(frame='四个train Parquet的全部行范围均匀抽取131072候选；正式归一化、去重、正文哈希valid划分；从通过者中无放回选1024篇',
            stories=[dict(source_id=s.source_id,sha256=hashlib.sha256(s.text.encode()).hexdigest()) for s in stories],seed=SEED))
    else:
        state=torch.load(a.out/'run/last.pt',map_location='cpu',weights_only=False,mmap=True)
        codec=FrozenPageCodec.from_state(state['codec'],encode_chunk_size=384).cuda();geometry=PageGenerator(codec)
        docs=torch.load(out/'documents.pt',map_location='cpu',weights_only=False,mmap=True);rows=[]
        for i in range(0,len(docs),16):
            rows.extend(encode_documents(docs[i:i+16],geometry,state['model']['latent_mean'],state['model']['latent_scale'],'valid',i))
        torch.save(rows,out/'valid_T1.pt')
        write_json(out/'valid_T1.json',dict(documents=len(rows),source_units=len({r['source'] for r in rows}),
            task='T1',blocks=sum(len(r['blocks']) for r in rows),selection='selection.json'))
        combined=json.loads((a.out/'data/selection.json').read_text(encoding='utf8'))
        text=json.loads((out/'selection.json').read_text(encoding='utf8'))
        combined.update(stories=text['stories'],text_sampling_frame=text['frame'],text_data_file='data_uniform_text/valid_T1.pt')
        write_json(a.out/'selection.json',combined)
    print(json.dumps(dict(stage='complete',phase=a.phase,seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':main()
