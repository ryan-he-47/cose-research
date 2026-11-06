"""2026-10-13：加载已训练的小读出头，以标准文字笔画回馈冻结R，支持自选前文。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import argparse
import time
import torch
from scripts.demo.demo_pretrain_generation import load_generator, save_result
from scripts.review.diagnose_formal_r_health import model_hash
from src.st.text_readout_generation import generate_text_readout as generate
from src.page_document import PageLayoutConfig
from src.st.text_readout import StrokeVocabulary, StrokeReadout
from src.text_normalization import normalize_text


def load_readout(head_path,checkpoint=None,device='cuda'):
    state=torch.load(head_path,map_location='cpu',weights_only=False)
    generator,font,metadata=load_generator(checkpoint or state['r_source']['checkpoint'],device)
    generator.model.requires_grad_(False)
    if model_hash(generator.model)!=state['r_model_hash']:
        raise ValueError('当前E/D/R与这个读出头的冻结来源不同')
    if metadata['data_manifest']['font_sha256']!=state['font_sha256']:
        raise ValueError('字体来源不同')
    vocab=StrokeVocabulary(font)
    if vocab.dump()!=state['vocabulary']:
        raise ValueError('文字笔画词表不同')
    if state.get('kind')=='shape_space':
        from src.st.text_shape_readout import ShapeVocabulary, ShapeSpaceReadout
        catalog=ShapeVocabulary(vocab)
        if catalog.dump()!=state['shape_vocabulary']:
            raise ValueError('笔画形状词表不同')
        head=ShapeSpaceReadout(state['input_width'],catalog,state['hidden']).to(generator.device)
    else:
        head=StrokeReadout(state['input_width'],len(vocab),state['hidden']).to(generator.device)
    head.load_state_dict(state['head']);head.eval();head.requires_grad_(False)
    layout_spec=dict(state['layout']);layout_spec['body_box']=tuple(layout_spec['body_box'])
    return generator,head,vocab,PageLayoutConfig(**layout_spec),state


def load_joint_readout(checkpoint,device='cuda'):
    """2026-10-14：读取正式R检查点内随R联合更新的辅助头。"""
    from src.st.text_shape_readout import ShapeVocabulary
    generator,font,state=load_generator(checkpoint,device)
    if generator.model.text_aux is None:raise ValueError('检查点没有联合训练的文字辅助头')
    manifest=state['text_aux_manifest']
    vocab=StrokeVocabulary(font);catalog=ShapeVocabulary(vocab,include_non_symbol=
        manifest['shape_vocabulary'].get('non_symbol_class') is not None)
    if catalog.dump()!=manifest['shape_vocabulary'] or state['data_manifest']['font_sha256']!=manifest['font_sha256']:
        raise ValueError('检查点标签表/字体来源不同')
    generator.model.text_aux.catalog=catalog
    generator.model.requires_grad_(False)
    layout_spec=dict(state['data_manifest'].get('page_layout',{}))
    if 'body_box' in layout_spec:layout_spec['body_box']=tuple(layout_spec['body_box'])
    return generator,generator.model.text_aux,vocab,PageLayoutConfig(**layout_spec),{
        'r_model_hash':model_hash(generator.model),'step':state['step']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--head',type=Path)
    source.add_argument('--r_aux_checkpoint',type=Path,help='正式R检查点内随R更新的辅助头')
    parser.add_argument('--checkpoint',type=Path,help='缺省使用头记录的原R路径；权重hash必须一致')
    parser.add_argument('--text',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    parser.add_argument('--constraint',choices=['none','prefix'],default='none',
        help='旧头prefix用字体笔顺约束；新形状/空格头总是使用形状前缀与落点策略')
    parser.add_argument('--generation_strokes',type=int,default=480)
    parser.add_argument('--beam_width',type=int,choices=[1,2,3,4],default=1,help='形状/空格分岔搜索宽度，1沿用贪心')
    parser.add_argument('--beam_length_penalty',type=float,default=1.,help='完整候选的逐笔长度归一指数')
    parser.add_argument('--batch_forced',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--max_seconds',type=float,default=600)
    args=parser.parse_args()
    if args.generation_strokes<1 or args.max_seconds<=0 or not args.text.strip():
        parser.error('前文不能为空，笔数和时间上限必须为正')
    if args.out.exists():
        parser.error('输出目录已存在，请使用新目录')
    normalized=normalize_text(args.text)
    if not normalized.is_supported:
        parser.error(f'当前字体不支持这些字符：{normalized.unsupported}')
    torch.set_num_threads(4)
    generator,head,vocab,layout,state=(load_joint_readout(args.r_aux_checkpoint,args.device) if args.r_aux_checkpoint
                                     else load_readout(args.head,args.checkpoint,args.device))
    started=time.monotonic()
    result=generate(generator,head,vocab,normalized.text,started+args.max_seconds,
                    args,layout)
    result['elapsed_seconds']=time.monotonic()-started
    result['head_source']=str((args.head or args.r_aux_checkpoint).resolve())
    result['r_model_hash']=state['r_model_hash']
    save_result(result,args.out,title='Frozen R / canonical text readout')
    print({k:v for k,v in result.items() if k!='strokes'})


if __name__=='__main__':main()
