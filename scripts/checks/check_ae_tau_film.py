"""2026-10-06：连续τ/FiLM条件接线、旧模型兼容、FP16恢复和冻结加载短检查。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hpc.train_ae_text5 import configure_sources
from src.st.ae_direct import ArcAECollator, DirectAEStream
from src.st.pretrain_model import FrozenSketchLM, config_from_ae_checkpoint
from src.st.train_ae import StrokeAutoencoder, config_from_checkpoint, load_ae_weights
from src.st.train_ae_direct import direct_ae_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', type=Path, default=ROOT / 'tmp_smoke/ae_tau_film_check_20261006')
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    recipe = json.loads((ROOT / 'configs/pretrain_ae_lambda48_dct_tau_film_recipe.json').read_text())
    baseline = json.loads((ROOT / 'configs/pretrain_ae_lambda48_dct_recipe.json').read_text())
    normalized = {key: value for key, value in recipe.items() if key not in ('experiment', 'ae_config')}
    assert normalized == {key: value for key, value in baseline.items() if key not in ('experiment', 'ae_config')}
    assert {key: value for key, value in recipe['ae_config'].items() if not key.startswith('dec_tau_')} == baseline['ae_config']
    torch.manual_seed(42)
    model = StrokeAutoencoder(direct_ae_config(recipe))
    torch.manual_seed(42)
    old = StrokeAutoencoder(direct_ae_config(baseline))
    for key, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, old.encoder.state_dict()[key], rtol=0, atol=0)
    parameters = sum(value.numel() for value in model.parameters())
    delta = parameters - sum(value.numel() for value in old.parameters())
    assert parameters == 6521446 and delta == 531456
    assert model.decoder.mlp[0].in_features == 48 and len(model.decoder.film) == 4
    tau = torch.tensor([[0., .12345, .5, 1.]], requires_grad=True)
    code = model.decoder.tau_encoding(tau)
    assert code.shape == (1, 4, 64) and code.dtype == torch.float32
    torch.testing.assert_close(code[0, 0, 0::2], torch.zeros(32), rtol=0, atol=0)
    torch.testing.assert_close(code[0, 0, 1::2], torch.ones(32), rtol=0, atol=0)
    assert not torch.allclose(code[:, 0], code[:, -1])
    assert float((model.decoder.tau_encoding(tau + 1e-5) - code).abs().max().detach()) < .01
    embedding = torch.randn(1, 48)
    with patch('torch.cat', side_effect=AssertionError('FiLM查询不能拼接τ和λ')):
        mu, sigma, pi = model.decoder(embedding, tau)
    assert not torch.allclose(mu[:, 0], mu[:, -1], rtol=1e-5, atol=1e-9)
    mu.square().sum().backward()
    assert tau.grad is not None and torch.isfinite(tau.grad).all() and tau.grad.abs().sum() > 0
    # 验证条件确实按通道同时缩放、偏移；去掉FiLM后τ依赖消失。
    with torch.no_grad():
        for layer in model.decoder.film:
            layer.weight.zero_()
            layer.bias.zero_()
        plain = model.decoder.head(model.decoder.mlp(embedding[:, None].expand(1, 4, 48)))
        output = model.decoder(embedding, tau.detach())
        for expected, actual in zip(plain, output):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    del model
    torch.manual_seed(42)
    model = StrokeAutoencoder(direct_ae_config(recipe))
    configure_sources(recipe)
    fixture = ROOT / 'tmp_smoke/ae_direct_check_20261004'
    stream = DirectAEStream(json.loads((fixture / 'data_args.json').read_text()), batch_size=64, workers=0)
    records = stream.next_batch()
    stream.close()
    batch = ArcAECollator(model.cfg, 42)(records)
    model(batch)['loss'].backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())
    old_state = torch.load(ROOT / 'tmp_smoke/ae_lambda48_continue_review_20261006/run/final.pt', map_location='cpu', weights_only=False)
    legacy = StrokeAutoencoder(config_from_checkpoint(old_state)).eval()
    assert legacy.cfg.dec_tau_mode == 'concat' and set(legacy.state_dict()) == set(old_state['model'])
    load_ae_weights(legacy, old_state)
    with torch.no_grad():
        embedded = legacy.encoder(batch['ink'], batch['stroke_len'])
        expected = legacy.decoder.head(legacy.decoder.mlp(torch.cat(
            (batch['tau'][..., None], embedded[:, None].expand(-1, 16, -1)), dim=-1)))
        actual = legacy.decoder(embedded, batch['tau'])
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    checks = dict(parameters=parameters, added_parameters=delta, pe_dimension=64,
                  frequency_cycles=[.5, 32.], recipe_only_tau_design_changed=True,
                  same_seed_encoder_unchanged=True, continuous_tau_gradient=True,
                  endpoints_distinct=True, film_without_concat=True, zero_film_is_lambda_mlp=True,
                  all_ae_and_film_gradients_finite=True, legacy_checkpoint_exact=True)
    del model, old, legacy, old_state, batch, mu, sigma, pi, code
    gc.collect()
    if args.fp16:
        smoke_recipe = dict(recipe, batch_size=64, microbatch_strokes=16,
                            padded_point_budget=1024, workers=2, warmup=2, validate_steps=2, save_steps=1)
        recipe_path = args.out_dir / 'recipe.json'
        recipe_path.write_text(json.dumps(smoke_recipe), encoding='utf-8')
        for name, steps, resume in (('first', 2, None), ('resumed', 3, args.out_dir / 'first/final.pt'), ('continuous', 3, None)):
            command = [sys.executable, '-u', 'scripts/hpc/train_ae_text5.py', '--manifest', str(fixture / 'data_args.json'),
                       '--validation_samples', str(fixture / 'samples.npz'), '--recipe', str(recipe_path),
                       '--output_dir', str(args.out_dir / name), '--steps', str(steps), '--max_seconds', '0', '--log_every', '1']
            if resume:
                command.extend(('--resume', str(resume)))
            with (args.out_dir / f'{name}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        resumed = torch.load(args.out_dir / 'resumed/final.pt', map_location='cpu', weights_only=False)
        continuous = torch.load(args.out_dir / 'continuous/final.pt', map_location='cpu', weights_only=False)
        assert resumed['step'] == 3 and len(resumed['optimizer']['state']) == 124
        assert all(int(value['step']) == 3 for value in resumed['optimizer']['state'].values())
        for key, value in resumed['model'].items():
            torch.testing.assert_close(value, continuous['model'][key], rtol=0, atol=0)
        for key, state in resumed['optimizer']['state'].items():
            for field, value in state.items():
                torch.testing.assert_close(value, continuous['optimizer']['state'][key][field], rtol=0, atol=0)
        for key in ('data', 'collator', 'scaler', 'stroke_counts'):
            assert resumed[key] == continuous[key], key
        frozen = FrozenSketchLM(config_from_ae_checkpoint(resumed))
        frozen.load_ae(resumed)
        assert frozen.cfg.dec_tau_mode == 'film'
        for key, value in resumed['model'].items():
            torch.testing.assert_close(value, frozen.state_dict()[key], rtol=0, atol=0)
        assert all(not value.requires_grad for name, value in frozen.named_parameters()
                   if name.startswith(('encoder.', 'decoder.', 'dct_head.')))
        checks.update(fp16_steps=[2, 3], optimizer_state_entries=124,
                      resumed_model_adam_data_scaler_exact=True, frozen_interface_film_load=True)
    (args.out_dir / 'checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == '__main__':
    main()
