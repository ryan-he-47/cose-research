"""冻结字体近邻探针：使用真实笔序标签，位置和页面字号在几何读出时去除。"""
import numpy as np
import torch
from src.text_renderer import VectorFont
from src.page_document import _render_page_text, PageLayoutConfig
from src.st.text_readout import StrokeVocabulary
from src.st.text_shape_readout import ShapeVocabulary


class ShapeProbe:
    def __init__(self,font_path,glyph_path,device):
        self.font=VectorFont.from_svg(font_path)
        self.vocab=StrokeVocabulary(self.font)
        self.shapes=ShapeVocabulary(self.vocab)
        self.device=device
        data=np.load(glyph_path)
        rows=np.repeat(np.arange(len(data['parts'])),data['parts'])
        parts=np.concatenate([np.arange(n) for n in data['parts']])
        keep=data['splits'][rows]==0
        r,p=rows[keep],parts[keep]
        curves=data['raw'][r,p]
        curves=curves-curves.mean(1,keepdims=True)
        self.reference=torch.tensor(curves.reshape(len(curves),-1),device=device)
        labels=[self.shapes.identity_to_shape[str(data['characters'][data['labels'][i]]),int(j)] for i,j in zip(r,p)]
        self.reference_labels=torch.tensor(labels,device=device)

    def target_labels(self,row,part):
        text_pages=sorted(set(row['pages'][row['kinds']==0].tolist()))
        assert len(text_pages)==len(row['text']),(row['id'],text_pages,row['text'])
        identities=torch.full((len(row['z']),),-1,dtype=torch.long)
        sizes=torch.ones(len(row['z']))
        for page,text in zip(text_pages,row['text']):
            labels=[self.shapes.identity_to_shape[c,j] for c in text if c in self.vocab.parts
                    for j in range(self.vocab.parts[c])]
            indices=torch.where((row['pages']==page)&(row['kinds']==0))[0]
            assert len(indices)==len(labels),(row['id'],page,len(indices),len(labels))
            identities[indices]=torch.tensor(labels)
            _,size,_=_render_page_text(text,self.font,PageLayoutConfig())
            sizes[indices]=size/.085
        return torch.tensor([int(identities[i]) if i>=0 else -1 for i in part['target']]),torch.tensor([float(sizes[i]) if i>=0 else 1. for i in part['target']])

    @torch.inference_mode()
    def classify(self,curves,scales):
        value=(curves-curves.mean(-2,keepdim=True))/scales[...,None,None]
        shape=value.shape[:-2]
        value=value.reshape(-1,64)
        output=[]
        for i in range(0,len(value),128):
            distances=torch.cdist(value[i:i+128],self.reference)
            output.append(self.reference_labels[distances.argmin(-1)])
        return torch.cat(output).reshape(shape)
