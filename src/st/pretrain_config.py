"""2026-10-02：已批准的新页协议预训练配置，旧 A1' 默认值保持原样。"""
from __future__ import annotations

from .config_st import STConfig


# 2026-10-14用户批准：13层R、GMM/形状/空格均从h线性读出，从头训练。
R_ARCHITECTURE_KEYS = ('r_layers', 'r_d_model', 'r_heads', 'r_dff', 'r_head_mlp',
    'r_head_extra', 'r_text_aux_enabled', 'r_text_aux_classes', 'r_text_aux_hidden')


def apply_r_architecture(cfg: STConfig, recipe: dict, *, fresh: bool) -> STConfig:
    """显式新R配置；旧checkpoint按自身cfg恢复，不能静默改变架构。"""
    spec = recipe.get('r_architecture', {})
    unknown = set(spec)-set(R_ARCHITECTURE_KEYS)
    if unknown:
        raise ValueError(f'未支持的R结构字段：{sorted(unknown)}')
    for name, value in spec.items():
        if name in ('r_head_mlp', 'r_text_aux_enabled'):
            if not isinstance(value, bool): raise ValueError(f'{name}须为bool')
        elif not isinstance(value, int) or isinstance(value, bool) or value < (
                0 if name in ('r_head_extra', 'r_text_aux_hidden') else 1):
            raise ValueError(f'{name}的结构取值无效')
        if fresh:
            setattr(cfg, name, value)
        elif getattr(cfg, name)!=value:
            raise ValueError(f'已有R的{name}与配方不同；架构更改须从AE起新R')
    if cfg.r_d_model % cfg.r_heads:
        raise ValueError('R宽度须整除头数')
    return cfg


def check_chinchilla_budget(model, recipe: dict):
    """项目沿用20×可训参数的成功笔画输入口径，冻结AE不计入。"""
    spec = recipe.get('chinchilla')
    if spec is None: return None
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual!=spec['trainable_parameters'] or recipe['total_inputs']!=actual*spec['inputs_per_parameter']:
        raise ValueError(f'实际可训参数{actual}与已声明的Chinchilla预算不一致')
    return dict(trainable_parameters=actual, inputs_per_parameter=spec['inputs_per_parameter'],
                total_inputs=recipe['total_inputs'], unit='successful_input_strokes', excludes_frozen_AE=True)


def gpt2_small_pretrain_config() -> STConfig:
    """冻结新分布 AE 后训练 R；R 尺寸沿用 GPT-2 small，输入输出沿用 CoSE。"""
    return STConfig(
        # 沿用现有统一 AE 的结构、坐标通道与重建设置。
        coord_dim=2,
        latent_units=16,
        endpoint_loss_w=0.0,
        temporal_resampling_factor=2,
        unified_emb=True,
        dct_coeffs=8,
        dct_dc=True,
        sigma_eps_r=1e-4,
        gmm_head_fp32=True,
        w_first=1.0,
        # 只照搬 GPT-2 small 尺寸，保留原笔画 GMM 缓冲头。
        r_layers=12,
        r_d_model=768,
        r_heads=12,
        r_dff=3072,
        gmm_r=32,
        r_head_mlp=True,
        r_pe_mode="page",
        slice_pe_dim=32,
        max_context_strokes=1024,
        # 同一冻结 E 编码上下文与目标；保留确定性/源笔顺。
        freeze_ed=True,
        stop_predictive_grad=True,
        free_input_encoder=False,
        p_shuffle_graphic=0.0,
        p_shuffle_text=0.0,
        first_stroke_prior=False,
        use_terminator=False,
        amp=True,
        amp_dtype="fp16",
    )
