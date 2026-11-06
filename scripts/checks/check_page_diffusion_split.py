"""双流注意力与旧完整掩码的有效输出、输入梯度和参数梯度核对。"""
from copy import deepcopy
import json
from pathlib import Path

import torch

from scripts.checks.check_page_diffusion import document
from scripts.checks.check_page_diffusion_performance import load_baseline
from src.st.page_diffusion import PageDiffusionConfig, PageDiffusionR
from src.st.page_diffusion_data import prepare_update, pack_documents, diffusion_inputs, diffusion_groups


def main():
    torch.set_num_threads(4)
    torch.manual_seed(112)
    old = load_baseline(Path("tmp_smoke/r_page_diffusion_runtime_v2_20261102/baseline_v1"), "page_diffusion")
    cfg = dict(width=64, layers=2, heads=4, ffn=128, condition=32, activation_checkpoint=False)
    baseline = old.PageDiffusionR(old.PageDiffusionConfig(**cfg))
    with torch.no_grad():
        for layer in baseline.layers:
            layer.modulation.weight.normal_(0., .02)
            layer.modulation.bias.normal_(0., .1)
    recipe = dict(context=1024, model={**baseline.configuration()}, encoder_length_buckets=[8,16,32,48,76],
        padding_group_weight=.25, padded_stroke_budget=128, microbatch_documents=8)
    docs = [document(34), document(9), document(58)]
    counts = prepare_update(docs, torch.Generator().manual_seed(15))
    dense = pack_documents(docs, recipe)
    compact = pack_documents(docs, {**recipe, "attention_layout":"split"})
    clean = dense["clean_size"]
    torch.testing.assert_close(dense["allowed"][..., clean:, :], compact["allowed"], atol=0, rtol=0)
    embeddings = torch.randn(*dense["stroke_pad"].shape, 48)
    values, target = diffusion_inputs(baseline, embeddings, dense)
    valid = dense["real_body"] | dense["real_control"] | dense["synthetic"]
    expected = None
    report = {}
    for label, layout, batch, checkpoint in [("old", "dense", dense, False),
            ("new_dense", "dense", dense, False), ("aligned", "aligned", dense, False),
            ("split_full", "split", dense, False), ("split_compact", "split", compact, False),
            ("split_checkpoint", "split", compact, True)]:
        model = baseline if label == "old" else PageDiffusionR(PageDiffusionConfig(**{**cfg, "activation_checkpoint":checkpoint}))
        if model is not baseline:
            # 旧分类头从未参与扩散预测；只对照保留的主干及扩散输出。
            model.load_state_dict({k:v for k,v in baseline.state_dict().items() if not k.startswith("text_aux.")}, strict=True)
        model.zero_grad(set_to_none=True)
        incoming = values.detach().clone().requires_grad_()
        kwargs = {} if label == "old" else dict(clean_size=clean, attention_layout=layout)
        output = model(incoming, batch["pages"], batch["local"], batch["roles"], batch["times"], batch["allowed"], **kwargs)
        prediction = output["prediction"][:, clean:][valid]
        loss = (prediction-target[valid]).square().mean()
        loss.backward()
        gradients = {n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
        if expected is None:
            expected = (prediction.detach().clone(), incoming.grad.clone(), gradients)
        else:
            torch.testing.assert_close(prediction, expected[0], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(incoming.grad, expected[1], atol=2e-6, rtol=2e-5)
            maximum = 0.
            for n, gradient in gradients.items():
                if gradient is None:
                    assert expected[2][n] is None
                    continue
                torch.testing.assert_close(gradient, expected[2][n], atol=2e-6, rtol=2e-4)
                maximum = max(maximum, float((gradient-expected[2][n]).abs().max()))
            report[label] = dict(output_max=float((prediction.detach()-expected[0]).abs().max()), gradient_max=maximum)
    original_noise = [d.metadata["_page_diffusion"]["noise"].clone() for d in docs]
    grouped = list(diffusion_groups(docs, {**recipe, "packing":"cost"}))
    assert sorted(map(id, docs)) == sorted(id(d) for group in grouped for d in group)
    for d, noise in zip(docs, original_noise):
        torch.testing.assert_close(d.metadata["_page_diffusion"]["noise"], noise, atol=0, rtol=0)
    print(json.dumps(dict(equivalence=report, packing_preserves_documents_and_noise=True)))


if __name__ == "__main__":
    main()
