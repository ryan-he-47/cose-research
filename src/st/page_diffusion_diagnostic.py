"""仅CPU导出一批真实训练文档，用于本机同批性能诊断；不加载R或更新参数。"""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from .formal_stream import InputDocumentStream, collect_documents
from .page_diffusion_data import prepare_update
from .page_diffusion_runtime import write_json
from .pipeline_pretrain import build_document_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_manifest", type=Path, required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if os.getenv("COSE_REQUIRE_SLURM") == "1" and not os.getenv("SLURM_JOB_ID"):
        p.error("服务器数据构造须经Slurm计算节点")
    if args.out.exists():
        p.error("诊断输出必须是新文件")
    torch.set_num_threads(1)
    started = time.monotonic()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    data = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    data.update(split="train", object_sampling="affine_pass_v1", paired_scene_share=0.)
    dataset = build_document_dataset(**data)
    stream = InputDocumentStream(dataset, data, recipe["input_shares"],
                                 max_strokes=recipe["context"], workers=0)
    try:
        documents, inputs, _, _ = collect_documents(stream, recipe["input_budget"])
        rng = torch.Generator().manual_seed(recipe["seed"]+100003)
        counts = prepare_update(documents, rng, recipe["model"]["block"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(documents=documents, counts=counts, recipe=recipe, inputs=inputs,
            source=dict(commit=os.getenv("COSE_GIT_RUNTIME_COMMIT"), job=os.getenv("SLURM_JOB_ID"),
                        cursor="fresh_first_update", data=data)), args.out)
        report = dict(documents=len(documents), inputs=sum(inputs.values()), counts=counts,
            raw_lengths=[d.stroke_count for d in documents],
            noisy_lengths=[len(d.metadata["_page_diffusion"]["blocks"])*16 for d in documents],
            seconds=time.monotonic()-started, bytes=args.out.stat().st_size)
        write_json(args.out.with_suffix(".json"), report)
        print(json.dumps({k:v for k,v in report.items() if not k.endswith("lengths")}))
    finally:
        stream.close()


if __name__ == "__main__":
    main()
