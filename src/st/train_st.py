"""时空架构训练入口（src/st/train_st.py）。

从零重构 —— 契约见 docs/22。LR 复用 src/train.py 的 TransformerLR（不手搓调度器，
2026-08-20 教训）；AMP 按 config（bf16 / fp16+GradScaler / off）。

用法（冒烟）：
    python src/st/train_st.py --steps 2000 --batch 32 --max_diagrams 20000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.st.config_st import STConfig
from src.st.model import STModel
from src.st.pipeline_st import build_st_loader
from src.st.lr import TransformerLR


def _clip_grouped(model: STModel, max_norm: float = 1.0):
    """分组梯度裁剪：E / D / R（双塔时 R_pred|R_pos 各自）独立 clip。

    A1 崩塌事故教训（docs/22 §13）：decoder 的 GMM μ 锐化梯度（~(x-μ)/σ²）范数
    可达 encoder 的 ~40 倍，E+D 同组裁剪会把 encoder 的有效步长压到 ~2%——
    E 必须独占一组。旧 train.py 的 E+D|R 两分法在本架构下不够用。
    """
    mods = [model.encoder, model.decoder]
    if model.dct_head is not None:
        mods.append(model.dct_head)      # 与优化器分组一致（2026-09-24 修复）
    if getattr(model, "input_encoder", None) is not None:
        mods.append(model.input_encoder)  # 预测侧（2026-09-27）
    for m in (model.r_pred, model.r_pos):
        if m is not None:
            mods.append(m)
    for m in mods:
        params = [p for p in m.parameters() if p.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="datasets/quick_draw/.cache")
    ap.add_argument("--task", default="graphic", choices=["graphic", "word_graphic"],
                    help="graphic=单片基线；word_graphic=A1' 两片（词条件画图）")
    ap.add_argument("--letter_db", default="datasets/handwritten_letters/letter_database.npz")
    ap.add_argument("--amp", default=None, choices=["bf16", "fp16", "off"],
                    help="覆盖 config 的 amp_dtype（V100 用 off 跑 fp32）")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader 进程数（服务器建议 4）")
    ap.add_argument("--max_diagrams", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--lr_scale", type=float, default=1.0,
                    help="LR 线性法则校正系数（v4=B=80，B=24 → 0.3）")
    ap.add_argument("--ed_lr_div", type=float, default=1.0,
                    help="AE 组 LR 分压系数（旧口径，已被 r_lr_mul 取代）")
    ap.add_argument("--r_lr_mul", type=float, default=1.0,
                    help="R 组 LR 放大系数（docs/24 阶段3 终版：32/Ḡ≈4.29，"
                         "模拟原管线 32 子集过采样的信号效率；基准调度按 AE 视角"
                         "的 batch 等效 80/24 计算）")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt", default=None, help="保存路径（默认 tmp_outputs/st_smoke.pt）")
    ap.add_argument("--output_dir", default=None,
                    help="周期 checkpoint 目录（ckpt_XXXXXX.pt，服务器长训用）")
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--resume", default=None, help="从 checkpoint 恢复（含步数）")
    ap.add_argument("--reset_step", action="store_true",
                    help="配合 --resume：加载权重但步数归零（分阶段训练时给 R 全新 warmup）")
    ap.add_argument("--ablate", default="full", choices=["full", "recon_only"],
                    help="recon_only：只用重建损失训 E/D（崩塌诊断消融）")
    ap.add_argument("--no_terminator", action="store_true",
                    help="关闭终止符 e（崩塌诊断消融）")
    ap.add_argument("--r_pe", default=None,
                    choices=["rope", "none", "concat", "rope_stroke"],
                    help="R 位置编码模式（D27 对照实验；默认取 config=concat）")
    ap.add_argument("--shuffle", type=float, default=None,
                    help="覆盖 riffle 触发率（对照实验设为 0）")
    ap.add_argument("--sdpa", default="auto", choices=["auto", "math"],
                    help="math=禁用 flash/mem-efficient 内核，纯参考后端（排障用）")
    ap.add_argument("--repro_a2", action="store_true",
                    help="A2 对齐复现模式：2D 坐标 + 无端点监督 + 长笔抽稀×2 + "
                         "r_pe=none + w_first=1 + σ_r 下限 1e-4 + 无 e。"
                         "唯一保留差异=shuffle 扁平因果（替代 32 子集 hybrid）")
    ap.add_argument("--sigma_eps_r", type=float, default=None,
                    help="覆盖 R 头 σ 下限（过锐机理检验用，如 0.05）")
    ap.add_argument("--r_dual_head", action="store_true",
                    help="D28 方案1：单 trunk 双头（位置+嵌入并联，无查询注入），"
                         "默认配 d192/dff768（单塔 1.44× 容量）")
    ap.add_argument("--r_d", type=int, default=None, help="覆盖 R d_model")
    ap.add_argument("--r_dff", type=int, default=None, help="覆盖 R dff")
    ap.add_argument("--r_layers", type=int, default=None, help="覆盖 R 层数")
    ap.add_argument("--r_heads", type=int, default=None, help="覆盖 R 注意力头数")
    ap.add_argument("--gmm_r", type=int, default=None, help="覆盖 R 头 GMM 分量数")
    ap.add_argument("--lr_ref_d", type=int, default=None,
                    help="LR 调度的 d_model 参考值（缺省=r_d_model；扩容实验时钉死 "
                         "160 保持与已验证配置同一 LR 轨迹，单一变量=容量）")
    ap.add_argument("--freeze_ed", action="store_true",
                    help="冻结 E/D 只训 R（decoder 前向跳过）——移动靶诊断探针")
    ap.add_argument("--first_stroke_prior", action="store_true",
                    help="首笔先验：50%% 最长笔/50%% 真实首笔固定图形序列首位"
                         "（豁免打乱），治开局模式坍塌")
    ap.add_argument("--free_input_encoder", action="store_true",
                    help="预测专用输入编码器：随机初始化同架构编码器接 R 输入、"
                         "不截梯度端到端训练（冻结编码器只产 λ 目标）")
    ap.add_argument("--init_ae", default=None,
                    help="从指定 ckpt 只加载 encoder/decoder/dct_head 权重"
                         "（R 保持随机初始化；配 --freeze_ed 做冻结嵌入训练）")
    ap.add_argument("--r_head_mlp", action="store_true",
                    help="GMM 头前加缓冲 MLP(512,256)+ReLU+LN（还原 v4 输出通路）")
    ap.add_argument("--r_mid_head_layer", type=int, default=None,
                    help="阶段1错层头：位置头读第 N 层输出，嵌入头仍读最终层（缺省=不错层）")
    ap.add_argument("--late_query_layer", type=int, default=None,
                    help="阶段2高层注入：R_pred 目标起点查询在第 K 层 Block 前加性注入"
                         "（缺省=旧行为注入输入层）")
    ap.add_argument("--r_skip_pos", action="store_true",
                    help="阶段2修正案：只训嵌入塔 R_pred，R_pos 前向与位置损失整个跳过")
    ap.add_argument("--r_head_query", action="store_true",
                    help="D29：目标起点拼到查询塔 GMM 头输入，transformer 侧零注入")
    ap.add_argument("--r_head_extra", type=int, default=None,
                    help="D29：缓冲 MLP 追加的 512 宽隐层数（加深 GMM 条件解码头）")
    ap.add_argument("--unified_emb", action="store_true",
                    help="计划1 统一嵌入：AE 保留绝对画布坐标（位置进 λ），配 DCT 纯形状头")
    ap.add_argument("--dct_coeffs", type=int, default=None,
                    help="DCT 纯形状辅助目标系数个数（每通道 k=1..K，如 8）")
    ap.add_argument("--dct_weight", type=float, default=None,
                    help="DCT MSE 损失权重（默认 0.1）")
    ap.add_argument("--dct_dc", action="store_true",
                    help="B 臂：DCT 目标追加质心 (x̄,ȳ)（直流=位置，显式解耦监督）")
    ap.add_argument("--pos_loss_w", type=float, default=None,
                    help="覆盖位置损失权重（淹没嫌疑检验，如 0.005）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = STConfig()
    if args.no_terminator:
        cfg.use_terminator = False
    if args.r_pe is not None:
        cfg.r_pe_mode = args.r_pe
    if args.repro_a2:
        # A2 正典对齐复现（2026-08-22 用户裁定）：先复现过去成功的结果，
        # 再逐项加架构更改。保留差异仅 shuffle 扁平因果。
        cfg.coord_dim = 2
        cfg.endpoint_loss_w = 0.0
        cfg.temporal_resampling_factor = 2
        cfg.r_pe_mode = "none"
        cfg.w_first = 1.0
        cfg.sigma_eps_r = 1e-4
        cfg.use_terminator = False
    if args.sigma_eps_r is not None:
        cfg.sigma_eps_r = args.sigma_eps_r
    if args.r_dual_head:
        cfg.r_dual_head = True
        cfg.r_d_model = args.r_d or 192      # (192/160)²≈1.44× 单塔容量
        cfg.r_dff = args.r_dff or 768
    else:
        if args.r_d is not None:
            cfg.r_d_model = args.r_d
        if args.r_dff is not None:
            cfg.r_dff = args.r_dff
    if args.r_layers is not None:
        cfg.r_layers = args.r_layers
    if args.r_heads is not None:
        cfg.r_heads = args.r_heads
    if args.gmm_r is not None:
        cfg.gmm_r = args.gmm_r
    if args.shuffle is not None:
        cfg.p_shuffle_graphic = args.shuffle
    if args.amp == "off":
        cfg.amp = False
    elif args.amp is not None:
        cfg.amp_dtype = args.amp
    if args.freeze_ed:
        cfg.freeze_ed = True
    if args.first_stroke_prior:
        cfg.first_stroke_prior = True
    if args.free_input_encoder:
        cfg.free_input_encoder = True
    if args.r_head_mlp:
        cfg.r_head_mlp = True
    if args.r_mid_head_layer is not None:
        cfg.r_mid_head_layer = args.r_mid_head_layer
    if args.late_query_layer is not None:
        cfg.late_query_layer = args.late_query_layer
    if args.r_skip_pos:
        cfg.r_skip_pos = True
    if args.r_head_query:
        cfg.r_head_query = True
    if args.r_head_extra is not None:
        cfg.r_head_extra = args.r_head_extra
    if args.unified_emb:
        cfg.unified_emb = True
    if args.dct_coeffs is not None:
        cfg.dct_coeffs = args.dct_coeffs
    if args.dct_weight is not None:
        cfg.dct_weight = args.dct_weight
    if args.dct_dc:
        cfg.dct_dc = True
    if args.pos_loss_w is not None:
        cfg.pos_loss_w = args.pos_loss_w
    if args.sdpa == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[train_st] SDPA 强制 math 后端（flash/mem-efficient 禁用）")
    device = torch.device(args.device)
    model = STModel(cfg).to(device)
    if cfg.freeze_ed:
        for p in list(model.encoder.parameters()) + list(model.decoder.parameters()):
            p.requires_grad_(False)
        if model.dct_head is not None:
            for p in model.dct_head.parameters():
                p.requires_grad_(False)
        print("[train_st] E/D(+dct_head) 已冻结（只训 R）")
    if args.init_ae:
        # 只移植 AE 侧权重（R 保持随机）——冻结嵌入训练臂（2026-09-25）
        st = torch.load(args.init_ae, map_location=device)
        mstate = st["model"] if "model" in st else st
        model.encoder.load_state_dict(
            {k[len("encoder."):]: v for k, v in mstate.items()
             if k.startswith("encoder.")})
        model.decoder.load_state_dict(
            {k[len("decoder."):]: v for k, v in mstate.items()
             if k.startswith("decoder.")})
        if model.dct_head is not None:
            model.dct_head.load_state_dict(
                {k[len("dct_head."):]: v for k, v in mstate.items()
                 if k.startswith("dct_head.")})
        # 冻结在 init 之后再做（init 不依赖 requires_grad，顺序无关，但明确起见）
        for p in list(model.encoder.parameters()) + list(model.decoder.parameters()):
            p.requires_grad_(False)
        if model.dct_head is not None:
            for p in model.dct_head.parameters():
                p.requires_grad_(False)
        print(f"[train_st] AE 权重已加载: {args.init_ae}（E/D/dct_head 已冻结）")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_st] params: {n_params/1e6:.2f}M | device={device} | "
          f"steps={args.steps} batch={args.batch}")

    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        step = int(state.get("step", 0))
        print(f"[train_st] resume: {args.resume} @ step {step}")
    if args.reset_step:
        step = 0  # 分阶段训练（docs/24 §3 阶段2-c）：权重继承、调度器从头 warmup

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    loader = build_st_loader(args.cache, "train", cfg, args.batch,
                             max_diagrams=args.max_diagrams, task=args.task,
                             letter_db_path=args.letter_db,
                             num_workers=args.workers)
    # 分组优化器：组0=E/D，组1=R（双塔时两塔同组）——配合 --ed_lr_div 实现
    # AE 慢速联合训练（docs/24 阶段3；原管线信号比 1:32 的 LR 近似）
    # 2026-09-24 修复：dct_head 此前不在任何参数组 → 从未被优化（dct_mse 恒=目标
    # 方差 0.25 的根因；dct 梯度进 E 但头本身冻结在随机初始化）
    ed_mods = [m for m in (model.encoder, model.decoder, model.dct_head)
               if m is not None]
    ed_params = [p for m in ed_mods
                 for p in m.parameters() if p.requires_grad]
    # 2026-09-27 free_input_encoder：输入编码器是预测侧模块，随 R 组同 LR
    r_mods = [m for m in (model.r_pred, model.r_pos,
                          getattr(model, "input_encoder", None))
              if m is not None]
    r_params = [p for m in r_mods for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam([{"params": ed_params}, {"params": r_params}], lr=0.0)
    # 分组调度：组0=E/D，组1=R，共享同一 schedule 形状；R 按 r_lr_mul 放大
    # （docs/24 阶段3 终版：放大 R 模拟 32 子集信号效率，基准按 AE 视角 80/24）
    gs = None
    if args.ed_lr_div > 1.0 or args.r_lr_mul != 1.0:
        gs = [1.0 / args.ed_lr_div, args.r_lr_mul]
    sched = TransformerLR(opt, d_model=(args.lr_ref_d or cfg.r_d_model),
                          warmup_steps=args.warmup,
                          lr_scale=args.lr_scale, group_scales=gs)
    sched._step = step  # resume 时同步调度器步数（LR 是 _step 的纯函数，直接对齐）

    # AMP：bf16（A100+/CPU 支持）或 fp16+GradScaler（V100）
    use_amp = cfg.amp and args.device == "cuda"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp_dtype)
    scaler = torch.amp.GradScaler() if (use_amp and cfg.amp_dtype == "fp16") else None
    print(f"[train_st] AMP: {cfg.amp_dtype if use_amp else 'off'}")

    model.train()
    t0 = time.time()
    running = {}
    nan_streak = 0
    while step < args.steps:
        for batch in loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(batch)
            if args.ablate == "recon_only":
                loss = (out["recon_nll"] + cfg.endpoint_loss_w * out["endpoint_nll"]
                        + cfg.dct_weight * out["dct_mse"])
            else:
                loss = out["loss"]
            if not torch.isfinite(loss):
                # NaN/Inf 跳批（对齐旧 train.py 行为；连续 50 次视为真故障才中止）
                nan_streak += 1
                print(f"[train_st] NaN/Inf at step {step}，跳过本批 "
                      f"(连续 {nan_streak} 次)")
                if nan_streak >= 50:
                    print("[train_st] 连续 NaN 过多，中止")
                    return
                opt.zero_grad()
                continue
            nan_streak = 0
            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                _clip_grouped(model)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                _clip_grouped(model)
                opt.step()
            sched.step()

            log_keys = ("loss", "recon_nll", "pred_nll", "pos_nll", "endpoint_nll",
                        "pred_e_nll", "pred_g_nll", "end_bce", "dct_mse")
            for k in log_keys:
                if k in out:
                    running[k] = running.get(k, 0.0) + out[k].item()
            if step % args.log_every == 0:
                el = time.time() - t0

                def _r(k):
                    v = running.get(k)
                    return f"{v/args.log_every:+.3f}" if v is not None else "  n/a "

                pred_str = (f"pred {_r('pred_nll')} (e/g "
                            f"{_r('pred_e_nll')}/{_r('pred_g_nll')})")
                print(f"step {step:6d} | recon {_r('recon_nll')} | "
                      f"{pred_str} | "
                      f"pos {_r('pos_nll')} | endpoint {_r('endpoint_nll')} | "
                      f"dct {_r('dct_mse')} | end_bce {_r('end_bce')} | "
                      f"lr {sched.get_lr():.2e} | {el/step:.3f}s/it")
                running = {}
            if args.output_dir and step % args.save_every == 0:
                p = Path(args.output_dir) / f"ckpt_{step:06d}.pt"
                torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                            "step": step}, p)
                print(f"[train_st] saved {p}")
            if step >= args.steps:
                break

    ckpt = args.ckpt or "tmp_outputs/st_smoke.pt"
    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step}, ckpt)
    print(f"[train_st] saved {ckpt}")


if __name__ == "__main__":
    main()
