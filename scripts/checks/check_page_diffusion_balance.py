"""用真实稳定段完整状态和归档文档，核对新阶段入口、数学、切块和中途恢复。"""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.st import train_page_diffusion as train
from src.st.formal_prefetch import PreparedUpdates
from src.st.formal_recipe import TASKS
from src.st.formal_stream import InputDocumentStream
from src.st.page_diffusion_balance import block_clock
from src.st.page_diffusion_data import losses, prepare_update
from src.st.page_diffusion_runtime import autocast, write_json
from src.st.page_diffusion_sampling import sample_block
from src.st.pipeline_pretrain import KIND_IDS


def check_geometry(spec):
    torch.manual_seed(805)
    prediction = torch.randn(8,5,48,requires_grad=True)
    target = torch.randn_like(prediction)
    residual = prediction-target
    axes = torch.tensor(spec["translation_axes"])
    model = torch.nn.Module()
    model.register_buffer("balance_axes",axes)
    times = torch.tensor([[.125,.3125,.5,.125,.125]]).expand(8,-1)
    body = torch.tensor([[True,True,True,False,False]]).expand(8,-1)
    control = torch.tensor([[False,False,False,True,False]]).expand(8,-1)
    synthetic = ~(body | control)
    kinds = torch.full((8,5),KIND_IDS["text"])
    kinds[TASKS.index("S1"),1] = KIND_IDS["sketch"]
    batch = dict(clean_size=0,times=times,real_body=body,real_control=control,synthetic=synthetic,
                 control_weight=control.float()+.25*synthetic,task_id=torch.arange(8),
                 targets=torch.arange(5).expand(8,-1),kind_id=kinds)
    weights = spec["target_shares"]
    counts = dict(body={task:3 for task in TASKS},control=8)
    recipe = dict(balance=spec,loss_weights=weights,control_loss_weight=.05)
    computed = losses(model,dict(prediction=prediction,hidden=None),target,batch,counts,recipe)["total"]
    raw = residual.square().mean(-1)
    perpendicular = (residual - (residual @ axes) @ axes.T).square().mean(-1)
    expected = .05*(raw[:,3]+.25*raw[:,4]).mean()
    for i,task in enumerate(TASKS):
        value = raw[i,:3].sum()
        if task in ("T1","S1"):
            value = value+.5*perpendicular[i,0]
            if task=="T1":
                value = value+.25*perpendicular[i,1]
        expected = expected+weights[task]*value/3
    torch.testing.assert_close(computed,expected)
    grad = torch.autograd.grad(computed,prediction)[0]
    reference = torch.autograd.grad(expected,prediction)[0]
    torch.testing.assert_close(grad,reference)
    return float((grad-reference).abs().max())


def check_targets(documents):
    signatures = {}
    sizes = {}
    for width in (16,8,4):
        counts = prepare_update(documents,torch.Generator().manual_seed(1105),width,supervision_block=16)
        keys = []
        for doc in documents:
            layout = doc.metadata["_page_diffusion"]
            sequence = doc.to_sequence()
            targets = []
            for part in layout["blocks"]:
                ids = [i for i in part["target"] if i>=0]
                assert ids and len(part["target"])==width
                assert ids == list(range(part["start"],part["start"]+len(ids)))
                assert part["local"]==layout["local"][ids[0]]
                assert all(sequence[i].page_id==part["page"] for i in ids)
                if len(ids)<width:
                    assert sequence[ids[-1]].kind=="eop" and part["target"][len(ids):]==[-1]*(width-len(ids))
                targets.extend(ids)
            keys.append(targets)
        if signatures:
            assert keys==signatures[16] and counts["body"]==sizes[16]["body"] and counts["control"]==sizes[16]["control"]
        signatures[width] = keys
        sizes[width] = counts
    return sizes


class ReplaySource(InputDocumentStream):
    """本机只替换原始数据读取；复用正式曝光调度与消费游标，用归档文档作短批检查。"""
    def __init__(self,pools,origin,recipe):
        super().__init__(SimpleNamespace(tasks=TASKS),{},recipe["input_shares"],workers=0,max_strokes=recipe["context"])
        self.pools,self.origin = pools,origin

    def _load(self,task):
        if task in self.windows:
            return
        offset = self.indices[task]-self.origin["indices"][task]
        source = self.pools[task][offset % len(self.pools[task])]
        doc = replace(source,metadata=dict(source.metadata))
        # 归档行已是训练窗口。起点T1位于原故事的中间，保留其窗口偏移。
        slots = 1+(self.origin["offsets"][task] if offset==0 else 0)
        self.windows[task] = [doc]*slots


class ReplayUpdates(PreparedUpdates):
    def take(self,budget):
        return super().take(min(budget,self.recipe["input_budget"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,required=True)
    parser.add_argument("--documents",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config = Path("configs/r_page_diffusion_balance_20261105.json")
    spec = json.loads(config.read_text(encoding="utf-8"))
    parent = torch.load(args.parent,map_location="cpu",mmap=True,weights_only=False)
    documents = torch.load(args.documents,map_location="cpu",weights_only=False)["documents"]
    torch.set_num_threads(4)
    gradient_error = check_geometry(spec)
    sizes = check_targets(documents)
    pools = defaultdict(list)
    for doc in documents:
        pools[doc.metadata["task"]].append(doc)
    origin = parent["stream"]

    def create_stream(data_args,recipe,state,*,new_exposure):
        stream = ReplaySource(pools,origin,recipe)
        if new_exposure:
            stream.continue_positions(state)
        else:
            stream.load_state_dict(state)
        return ReplayUpdates(stream,{**recipe,"input_budget":2048},pin=True,prepare_microbatches=False)

    class Manifest:
        def manifest(self):
            return parent["source"]["data_manifest"]

    output = args.output/"training"
    start = spec["parent_step"]
    common = ["check","--mode","train","--device","cuda","--output_dir",str(output)]
    # 同一真实入口，先跑半个块长周期，再从磁盘的完整状态恢复跑完。
    with patch.object(train,"create_stream",create_stream), patch.object(train,"build_document_dataset",return_value=Manifest()):
        first = ["--balance_from",str(args.parent),"--balance_config",str(config),
                 "--runtime_padded_stroke_budget","2048","--runtime_microbatch_documents","8",
                 "--no-runtime_adaptive_microbatch","--max_steps",str(start+2)]
        with patch.object(sys,"argv",common+first):
            train.main()
        saved = torch.load(output/"last.pt",map_location="cpu",mmap=True,weights_only=False)
        clock = saved["block_clock"]
        weights = saved["recipe"]["loss_weights"]
        assert clock==block_clock(saved["recipe"],start+2) and saved["scheduler"]["inputs"]>spec["parent_inputs"]
        del saved
        with patch.object(sys,"argv",common+["--resume",str(output/"last.pt"),"--max_steps",str(start+4)]):
            train.main()
    state = torch.load(output/"last.pt",map_location="cpu",mmap=True,weights_only=False)
    report = json.loads((output/"balance_calibration.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output/"train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["native_block"] for r in rows]==[block_clock(state["recipe"],start+i)["next_width"] for i in range(4)]
    assert sorted(r["native_block"] for r in rows)==[4,8,8,16] and rows[2]["native_block"]==clock["next_width"]
    assert state["recipe"]["loss_weights"]==weights==report["loss_weights"]
    assert report["committed_inputs"]==report["optimizer_updates"]==0 and report["cursor_restored"]
    assert state["step"]==start+4 and all(int(s["step"])==start+4 for s in state["optimizer"]["state"].values())
    assert sum(state["stream"]["inputs"].values())==sum(sum(row["input_counts"].values()) for row in rows)
    assert state["scheduler"]["inputs"]==spec["parent_inputs"]+sum(state["stream"]["inputs"].values())
    assert state["model_cfg"]["block"]==8 and state["model"].keys()==parent["model"].keys()==state["ema"].keys()
    assert not any("aux" in k for k in state["model"])
    for row in rows:
        assert abs(sum(row["body_share"].values())-1.)<1e-6 and row["weighted_control"]>=0
    from src.st.page_diffusion import PageDiffusionConfig,PageDiffusionR
    model = PageDiffusionR(PageDiffusionConfig(**state["model_cfg"])).cuda().eval()
    model.load_state_dict(state["ema"])
    with autocast(torch.device("cuda")):
        for width in (4,8,16):
            result,_,local = sample_block(model,None,0,3,torch.randn(1,width,48,device="cuda"),steps=2)
            assert torch.isfinite(result).all() and local[0].tolist()==list(range(3,3+width))
    summary = dict(documents=len(documents),targets=sizes,max_loss_gradient_error=gradient_error,
        calibration_measured_inputs=report["measured_inputs"],heldout_share=report["heldout_share"],
        widths=[r["native_block"] for r in rows],heldout_updates=report["check_updates"],
        local_update_inputs=sum(state["stream"]["inputs"].values()),adam_steps=state["step"],
        next_width=state["block_clock"]["next_width"],formal_budget_consumed=0,
        finite_sampling_widths=[4,8,16],oom_retries=sum(r["oom_retries"] for r in rows))
    write_json(args.output/"checks.json",summary)
    print(json.dumps(summary))


if __name__=="__main__":
    main()
