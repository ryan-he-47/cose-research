"""冻结R，测量文字前文对整字去噪和自由生成的作用，以及读出层的字形梯度。"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.review.page_diffusion_probe_core import (
    cache_histories, common_noise, decode, history, load_model, predict_blocks, selected, targets,
)
from scripts.review.page_diffusion_shape_probe import ShapeProbe
from scripts.review.render_page_diffusion_diagnostics import estimate, resamples
from src.page_document import PageLayoutConfig, _render_page_text
from src.st.generation_pretrain import PageGenerator
from src.st.page_diffusion_runtime import autocast, write_json


TIMES = (1., .875, .75, .625, .5, .375, .25, .125, .03125)
FOLDS = 16
GEOMETRY = Path('tmp_smoke/ae_latent_geometry_20261103/glyphs.npz')
FONT = Path('E:/Trash_things/cose/datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg')


def stable_id(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], 'big')


class Letters:
    def __init__(self, device):
        self.strokes = ShapeProbe(FONT, GEOMETRY, device)
        data = np.load(GEOMETRY)
        parts, splits, labels = (data[key] for key in ('parts','splits','labels'))
        self.label = {str(c): i for i,c in enumerate(data['characters'])}
        self.reference = {}
        raw = data['raw']
        for n in np.unique(parts):
            keep = (parts == n) & (splits == 0)
            curves = raw[keep, :n].copy()
            curves -= curves.mean((1,2), keepdims=True)
            self.reference[int(n)] = (torch.tensor(curves.reshape(len(curves),-1),device=device),
                                      torch.tensor(labels[keep],device=device))
        # 同一笔形、增强和字号，仅改变整字原点。参考实例拟合，独立增强实例验算。
        rows = np.repeat(np.arange(len(parts)), parts)
        cols = np.concatenate([np.arange(n) for n in parts])
        shift = data['origins'][rows] - np.array([.35,.45])
        displacement = (data['varied'][rows,cols]-data['latent'][rows,cols]).astype(np.float64)
        train = splits[rows] == 0
        linear = np.linalg.lstsq(shift[train], displacement[train], rcond=None)[0]
        axes = np.linalg.qr(linear.T)[0]
        actual = displacement[~train]
        self.axes = torch.tensor(axes,dtype=torch.float32,device=device)
        self.geometry = dict(test_strokes=len(actual),
            projected_energy=float(np.square(actual@axes).sum()/np.square(actual).sum()),
            translation_fit_r2=float(1-np.square(actual-shift[~train]@linear).sum()/
                                     np.square(actual-actual.mean(0)).sum()))

    def metadata(self, row):
        block = selected(row)
        pages = sorted(set(row['pages'][row['kinds']==0].tolist()))
        assert len(pages) == len(row['text'])
        groups = []
        text = row['text'][pages.index(block['page'])]
        indices = torch.where((row['pages']==block['page']) & (row['kinds']==0))[0].tolist()
        offset = 0
        for ch in text:
            n = self.strokes.vocab.parts.get(ch,0)
            if n:
                groups.append((ch,indices[offset:offset+n]))
                offset += n
        assert offset == len(indices), row['id']
        _, size, _ = _render_page_text(text,self.strokes.font,PageLayoutConfig())
        target_slots = {i:j for j,i in enumerate(block['target']) if i>=0}
        letters = [(self.label[c],[target_slots[i] for i in ids]) for c,ids in groups
                   if c.isascii() and (c.isalpha() or (row['task']!='T1' and c.isdigit()))
                   and all(i in target_slots for i in ids)]
        past = [(c,ids) for c,ids in groups if ids[-1]<block['start']][-32:]
        return letters, size/.085, past

    def classify(self, shapes, scale):
        # [字,时间,笔,点,xy]，保留字内各笔的相对位置。
        n = shapes.shape[2]
        reference, labels = self.reference[n]
        centered = (shapes-shapes.mean((2,3),keepdim=True))/scale[:,None,None,None,None]
        flat = centered.flatten(2).reshape(-1,reference.shape[1])
        classes = []
        for batch in flat.split(192):
            classes.append(labels[torch.cdist(batch,reference).argmin(-1)])
        return torch.cat(classes).reshape(shapes.shape[:2])


@torch.inference_mode()
def histories_for_batch(rows, metadata, letters, model, codec, encoder):
    device = next(model.parameters()).device
    control = [history(r) for r in rows]
    control = [(z.clone(),p,l) for z,p,l in control]
    swapped = [(z.clone(),p,l) for z,p,l in control]
    moves, changes = [], np.zeros(len(rows),dtype=int)
    for b,(row,(_,_,past)) in enumerate(zip(rows,metadata)):
        pools = defaultdict(list)
        rng = np.random.default_rng(stable_id(row['source']))
        for ch,ids in past:
            if ch.isascii() and ch.isalpha():
                width = letters.strokes.font.glyphs[ch].advance
                width_bin = round(2*np.log2(max(width,1)))
                pools[(len(ids),width_bin)].append((ch,ids))
        for group in pools.values():
            if len({ch for ch,_ in group})<2:
                continue
            ordered = [group[i] for i in rng.permutation(len(group))]
            # 循环置换保留字符库和每槽笔数，选择能改变最多字符的位置偏移。
            shift = max(range(1,len(group)), key=lambda k:
                        sum(a[0]!=ordered[(j+k)%len(group)][0] for j,a in enumerate(ordered)))
            for j,(ch,ids) in enumerate(ordered):
                other,donor = ordered[(j+shift)%len(group)]
                if ch!=other:
                    changes[b] += 1
                    moves.append((b,ids,donor))
    if moves:
        all_indices = sorted({(b,i) for b,ids,donor in moves for i in ids+donor})
        lookup = {key:i for i,key in enumerate(all_indices)}
        z = torch.stack([rows[b]['z'][i] for b,i in all_indices]).to(device)
        original = decode(codec,model,z,points=76)
        changed, destinations, owners = [], [], []
        for b,ids,donor in moves:
            dest = [lookup[b,i] for i in ids]
            source = original[[lookup[b,i] for i in donor]]
            shifted = source - source.mean((0,1)) + original[dest].mean((0,1))
            changed.append(shifted)
            destinations.extend(dest)
            owners.extend((b,i) for i in ids)
        curves = torch.cat((original[destinations],torch.cat(changed)))
        with autocast(device):
            encoded = model.normalize(encoder.encode_tensor(curves)).float().cpu()
        original_z, changed_z = encoded.chunk(2)
        for i,(b,slot) in enumerate(owners):
            control[b][0][slot]=original_z[i]
            swapped[b][0][slot]=changed_z[i]
    return control,swapped,changes


def glyph_metrics(curves, target, metadata, alphabet, derivatives=False):
    batch, times = curves.shape[:2]
    device = curves.device
    count = np.zeros(batch,int)
    correct = np.zeros((batch,times),int)
    first = np.zeros_like(correct)
    pools = defaultdict(list)
    for b,(items,scale,_) in enumerate(metadata):
        count[b] = len(items)
        for index,(label,slots) in enumerate(items):
            pools[len(slots)].append((b,slots,label,scale,index))
    mask = torch.zeros((batch,16),device=device)
    shape_grad, center_grad = (torch.zeros_like(curves),torch.zeros_like(curves)) if derivatives else (None,None)
    for group in pools.values():
        documents,slots,labels,scales,order = zip(*group)
        ix = torch.tensor(documents,device=device)[:,None]
        pos = torch.tensor(slots,device=device)
        scale = torch.tensor(scales,device=device,dtype=torch.float32)
        values = curves[ix,:,pos].transpose(1,2)
        hits = (alphabet.classify(values,scale)==torch.tensor(labels,device=device)[:,None]).cpu().numpy()
        np.add.at(correct,np.asarray(documents),hits)
        is_first = np.asarray(order)==0
        first[np.asarray(documents)[is_first]] = hits[is_first]
        if derivatives:
            delta = values-target[ix,pos][:,None]
            center = delta.mean((2,3),keepdim=True)
            factor = scale[:,None,None,None,None].square()
            shape_grad[ix,:,pos] = ((delta-center)/factor).transpose(1,2)
            center_grad[ix,:,pos] = (center.expand_as(delta)/factor).transpose(1,2)
            mask[ix,pos] = 1
    return count,correct,first,mask,shape_grad,center_grad


def decoder_gradients(model,codec,values,shape_vector,center_vector):
    # 只对D输入求梯度。常量缩放避免FP16反传下溢，输出还原到FP32。
    dims = values.shape
    shape,center = [],[]
    tau = torch.linspace(0,1,32,device=values.device)
    flat = values.reshape(-1,48)
    for begin in range(0,len(flat),96):
        end = min(begin+96,len(flat))
        z = flat[begin:end].detach().clone().requires_grad_(True)
        with autocast(z.device):
            mu,_,pi = codec.decoder(model.denormalize(z),tau[None].expand(len(z),-1))
        ids = pi.detach().argmax(-1)[...,None,None].expand(-1,-1,1,2)
        xy = mu.float().gather(2,ids).squeeze(2)*.6
        for results,vector,retain in ((shape,shape_vector,True),(center,center_vector,False)):
            upstream = vector.reshape(-1,32,2)[begin:end]*256
            dz, = torch.autograd.grad(xy,z,grad_outputs=upstream,retain_graph=retain)
            results.append(dz.detach()/256)
    return torch.cat(shape).reshape(dims),torch.cat(center).reshape(dims)


def alignment(folds, axes):
    # 一折的可见字形梯度检验其余折的读出层方向；偏置作为额外一列一起计算。
    unit = lambda x: x / np.maximum(np.linalg.norm(x,axis=(-2,-1),keepdims=True),1e-20)
    projection = axes@axes.T
    results = defaultdict(list)
    for held in range(len(folds)):
        original = folds[:,0].sum(0)-folds[held,0]
        position = projection@original
        weighted = position+2*(original-position)
        for index,label in ((1,'shape'),(2,'center')):
            test = unit(folds[held,index])
            for key,direction in (('current',original),('twice_residual',weighted)):
                results[label+'_'+key].append((unit(direction)*test).sum((-2,-1)))
    return {key:np.stack(rows).tolist() for key,rows in results.items()}


@torch.inference_mode()
def free_stages(rows,model,codec,encoder,alphabet):
    device = next(model.parameters()).device
    bands = ((.75,1.),(.5,.75),(.25,.5),(0.,.25))
    results = []
    for begin in range(0,len(rows),16):
        batch = rows[begin:begin+16]
        control = cache_histories(model,[r['control'] for r in batch])
        swapped = cache_histories(model,[r['swapped'] for r in batch])
        pages = torch.tensor([r['page'] for r in batch],device=device)
        local = torch.tensor([r['local'] for r in batch],device=device)
        size = torch.tensor([r['scale'] for r in batch],device=device)[:,None].expand(-1,16)
        start = common_noise(batch,salt=200).to(device)
        outcomes = []
        for band in (None,*bands):
            z = start[:,None].clone()
            with autocast(device):
                for step in range(32,0,-1):
                    t = step/32
                    past,valid = swapped if band and band[0]<t<=band[1] else control
                    pred = predict_blocks(model,past,valid,z,torch.tensor([t],device=device),pages,local)
                    z = ((step-1)/step)*z+pred/step
            curves = decode(codec,model,z[:,0],points=76)
            distance = (curves[:,:,None,None]-encoder.template_tensor[None,None]).square().sum(-1).mean(-1).sqrt().amin((-1,-2))
            alive = (distance<=encoder.control_tolerance).long().cumsum(1)==0
            decoded = decode(codec,model,z[:,0],points=32)
            shape = alphabet.strokes.classify(decoded,size)
            outcomes.append((alive.cpu().numpy(),shape.cpu().numpy()))
        base_alive,base_shape = outcomes[0]
        for i,row in enumerate(batch):
            effects = []
            for alive,shape in outcomes[1:]:
                common = base_alive[i]&alive[i]
                effects.append([int((common&(base_shape[i]!=shape[i])).sum()),int(common.sum()),
                                int(base_alive[i].sum()!=alive[i].sum())])
            results.append(dict(id=row['id'],changed=row['changed'],effects=effects))
    array = np.array([r['effects'] for r in results]);counts = np.array([r['changed'] for r in results])
    out = {}
    for name,keep in (('all',np.ones(len(rows),bool)),('edited',counts>0)):
        ix = resamples(int(keep.sum()))
        out[name] = [dict(band=list(band),shape_change=estimate(array[keep,k,0],ix,array[keep,k,1]),
                          stopping_position_change=estimate(array[keep,k,2],ix)) for k,band in enumerate(bands)]
    return out,results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--task',choices=('T1','O2','S1','S3','S5'),default='T1')
    parser.add_argument('--limit',type=int,default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    model,codec,checkpoint = load_model(args.review/'run/last.pt')
    del checkpoint
    model.requires_grad_(False);codec.requires_grad_(False)
    device = next(model.parameters()).device
    encoder = PageGenerator(codec)
    alphabet = Letters(device)
    data_dir='data_uniform_text' if args.task=='T1' else 'data'
    metric_dir='metrics_uniform_text' if args.task=='T1' else 'metrics'
    rows = torch.load(args.review/data_dir/f'valid_{args.task}.pt',map_location='cpu',mmap=True,weights_only=False)
    if args.limit:
        rows=rows[:args.limit]
    paths=torch.load(args.review/metric_dir/'final_raw'/f'sampling_valid_{args.task}.pt',map_location='cpu',mmap=True,weights_only=False)
    free_ids={r['id'] for r in paths};del paths
    donors={r['id']:rows[(i+max(1,len(rows)//2))%len(rows)] for i,r in enumerate(rows)}
    rows.sort(key=lambda r:selected(r)['start'])
    times=torch.tensor(TIMES,device=device)
    width=model.cfg.width+1
    folds=torch.zeros(FOLDS,3,len(times),48,width,device=device)
    observations,free_rows = [],[]
    for begin in range(0,len(rows),8):
        batch=rows[begin:begin+8]
        metadata=[alphabet.metadata(r) for r in batch]
        raw_hist=[history(r) for r in batch]
        if args.task=='T1':
            clean_hist,swap_hist,changes=histories_for_batch(batch,metadata,alphabet,model,codec,encoder)
            modes=(('raw',raw_hist),('control',clean_hist),('swapped',swap_hist))
        else:
            clean_hist=raw_hist
            swap_hist=[history(r,'wrong_condition',donors[r['id']]) for r in batch]
            changes=np.asarray([int(donors[r['id']]['source']!=r['source']) for r in batch])
            modes=(('raw',raw_hist),('swapped',swap_hist))
        target,_,_,_=targets(model,batch)
        eps=common_noise(batch).to(device)
        noisy=(1-times[None,:,None,None])*target[:,None]+times[None,:,None,None]*eps[:,None]
        pages=torch.tensor([selected(r)['page'] for r in batch],device=device)
        local=torch.tensor([selected(r)['local'] for r in batch],device=device)
        truth=decode(codec,model,target)
        count,ceiling,_,_,_,_=glyph_metrics(truth[:,None],truth,metadata,alphabet)
        metrics=dict(ceiling=ceiling)
        for mode,histories in modes:
            cache,valid=cache_histories(model,histories)
            features=[]
            handle=model.output.register_forward_pre_hook(lambda module,inputs: features.append(inputs[0].detach())) if mode=='raw' else None
            try:
                pred=predict_blocks(model,cache,valid,noisy,times,pages,local)
            finally:
                if handle is not None:handle.remove()
            del cache
            curves=decode(codec,model,pred)
            _,correct,first,mask,vs,vc=glyph_metrics(curves,truth,metadata,alphabet,derivatives=mode=='raw')
            metrics[mode],metrics[mode+'_first']=correct,first
            if mode=='raw':
                gs,gc=decoder_gradients(model,codec,pred,vs,vc)
                hidden=features[0].float().reshape(len(batch),len(times),16,-1)
                hidden=torch.cat((hidden,torch.ones_like(hidden[...,:1])),dim=-1)
                residual=(pred-target[:,None])*mask[:,None,:,None]
                for i,r in enumerate(batch):
                    fold=stable_id(r['source'])%FOLDS
                    for j,vector in enumerate((residual,gs,gc)):
                        folds[fold,j]+=vector[i].transpose(-2,-1)@hidden[i]
        if args.task!='T1':
            metrics['control']=metrics['raw']
            metrics['control_first']=metrics['raw_first']
        for i,r in enumerate(batch):
            observations.append(dict(id=r['id'],source=r['source'],label=r['label'],letters=int(count[i]),
                changed=int(changes[i]),**{k:v[i].tolist() for k,v in metrics.items()}))
            if r['id'] in free_ids or args.limit:
                free_rows.append(dict(id=r['id'],changed=int(changes[i]),control=clean_hist[i],swapped=swap_hist[i],
                                      page=int(pages[i]),local=int(local[i]),scale=metadata[i][1]))
        if (begin+len(batch))%128==0 or begin+len(batch)==len(rows):
            print(json.dumps(dict(phase='teacher',documents=begin+len(batch))),flush=True)
    np.savez_compressed(args.out/'text_readout_gradients.npz',folds=folds.cpu().numpy(),axes=alphabet.axes.cpu().numpy())
    write_json(args.out/'text_teacher_rows.json',observations)
    if not torch.isfinite(folds).all():raise ValueError('读出层梯度出现非有限值')
    n=np.asarray([r['letters'] for r in observations]);changed=np.asarray([r['changed'] for r in observations])
    data={k:np.asarray([r[k] for r in observations]) for k in ('raw','control','swapped','control_first','swapped_first','ceiling')}
    summary=dict(task=args.task,times=TIMES,geometry=alphabet.geometry,documents=len(n),letters=int(n.sum()),
                 intervention='matched_local_letters' if args.task=='T1' else 'replace_previous_pages_keep_own_answer_prefix',
                 edited_documents=int((changed>0).sum()),changed_letters=int(changed.sum()) if args.task=='T1' else None,
                 alignment=alignment(folds.cpu().double().numpy(),alphabet.axes.cpu().double().numpy()),teacher={})
    for name,keep in (('all',n>0),('edited',(changed>0)&(n>0))):
        labels=np.asarray([r['label'] for r in observations])[keep] if args.task=='O2' else None
        ix=resamples(int(keep.sum()),labels)
        summary['teacher'][name]=dict(n=int(keep.sum()),letters=int(n[keep].sum()),
            ceiling=estimate(data['ceiling'][keep,0],ix,n[keep]),times=[])
        for j,t in enumerate(TIMES):
            values={key:estimate(data[key][keep,j],ix,n[keep]) for key in ('raw','control','swapped')}
            values['context_gain']=estimate(data['control'][keep,j]-data['swapped'][keep,j],ix,n[keep])
            values['first_gain']=estimate(data['control_first'][keep,j]-data['swapped_first'][keep,j],ix)
            summary['teacher'][name]['times'].append(values)
    write_json(args.out/'text_stage.json',summary)
    summary['free'],raw=free_stages(free_rows,model,codec,encoder,alphabet)
    write_json(args.out/'text_free_rows.json',raw)
    write_json(args.out/'text_stage.json',summary)
    print(json.dumps(dict(phase='complete',documents=len(n),free_documents=len(raw))),flush=True)


if __name__=='__main__':
    main()
