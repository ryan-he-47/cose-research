"""重建已选验证场景的圈选答案，供几何归属率与 IoU 统计。"""
import argparse
import json
from pathlib import Path
from scripts.review.joint_eval_batch import load_dataset
from src.text_augmentation import TextAugmentationConfig
from src.text_renderer import VectorFont
from src.st.page_diffusion_runtime import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    font=VectorFont.from_svg('E:/Trash_things/cose/datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg')
    dataset,_=load_dataset(Path('E:/Trash_things/cose/tmp_smoke/r_joint_statistical_data_20261027'),'valid',font)
    dataset.augmentation=TextAugmentationConfig(enabled=True)
    ids=json.loads((a.out/'data/selection.json').read_text(encoding='utf8'))['scene_indices']
    result={}
    for i,index in enumerate(ids):
        document=dataset.get_for_task('S4',index)
        m=document.metadata
        result[str(index)]={k:m[k] for k in ('objects','target_instance_id','selection_box','question','query_kind')}
        if (i+1)%128==0:print(json.dumps(dict(stage='scene_metadata',done=i+1)),flush=True)
    write_json(a.out/'scene_metadata.json',result)


if __name__=='__main__':main()
