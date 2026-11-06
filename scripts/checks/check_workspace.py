"""2026-10-01：Codex 接管环境核验；只读，不训练、不安装依赖。"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", action="store_true", help="检查本机数据和基准权重是否存在")
    parser.add_argument("--legacy-assets", action="store_true", help="显式核对旧A1缓存/字库/权重")
    parser.add_argument("--require-cuda", action="store_true", help="要求 CUDA 可用")
    parser.add_argument("--checkpoint", type=Path, help="读取可信本地 checkpoint 的配置")
    args = parser.parse_args()
    errors = []
    print(f"解释器：{sys.executable}\nPython：{sys.version.split()[0]}\n仓库：{ROOT}")
    modules = {}
    for module, distribution in (
        ("torch", "torch"), ("numpy", "numpy"), ("scipy", "scipy"),
        ("matplotlib", "matplotlib"), ("sklearn", "scikit-learn"),
        ("tqdm", "tqdm"), ("flask", "flask"), ("pyarrow", "pyarrow"),
    ):
        try:
            modules[module] = importlib.import_module(module)
            print(f"依赖：{module} {version(distribution)}")
        except Exception as exc:
            errors.append(f"依赖 {module}：{exc}")
    try:
        importlib.import_module("tkinter")
        print("GUI：tkinter 可导入（窗口交互需另行目检）")
    except ImportError:
        print("GUI：tkinter 不可用；无显示服务器可忽略，本地 GUI 需修复")
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py"))
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('scripts.'):
                    if importlib.util.find_spec(node.module) is None:
                        errors.append(f"脚本导入不存在：{path.relative_to(ROOT)} → {node.module}")
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"语法 {path.relative_to(ROOT)}：{exc}")
    print(f"活跃 Python 语法：{len(files)} 个文件")
    for module in ("src.st.model", "src.st.pipeline_st", "src.st.dct", "src.st.lr",
                   "src.st.pipeline_pretrain", "src.st.pretrain_model", "src.st.train_ae"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"活跃栈导入 {module}：{exc}")
    torch = modules.get("torch")
    if torch is not None:
        available = torch.cuda.is_available()
        print(f"CUDA：{available}，torch CUDA={torch.version.cuda}")
        if available:
            print(f"GPU：{torch.cuda.get_device_name(0)}")
        elif args.require_cuda:
            errors.append("CUDA 不可用")
    if args.assets:
        paths = ["datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg",
                 "tmp_smoke/r_dual_review_assets_20261009/ae_final.pt"]
        for name in paths:
            path = ROOT/name
            if path.is_file() and path.stat().st_size > 0:
                print(f"当前资产：{name} ({path.stat().st_size:,} 字节)")
            else:
                errors.append(f"缺少当前资产：{name}")
    if args.legacy_assets:
        paths = [
            "datasets/quick_draw/.cache_9cls_resampled/.cache/meta.json",
            *[f"datasets/quick_draw/.cache_9cls_resampled/.cache/precached_{s}.npz"
              for s in ("train", "valid", "test")],
            "datasets/handwritten_letters/letter_database.npz",
            "tmp_outputs/st_uni_dc/ckpt_270000.pt",
            "tmp_outputs/st_uni_frozen2/final.pt",
        ]
        for name in paths:
            path = ROOT / name
            if path.is_file() and path.stat().st_size > 0:
                print(f"资产：{name} ({path.stat().st_size:,} 字节)")
            else:
                errors.append(f"缺少资产：{name}（worktree 不会自动复制 gitignored 文件）")
    if args.checkpoint:
        if torch is None:
            errors.append("读取 checkpoint 需要 torch")
        else:
            try:
                path = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
                state = torch.load(path, map_location="cpu", weights_only=False)
                cfg = state["cfg"]
                print(f"检查点：{path}，step={state.get('step')}；"
                      f"stage={state.get('stage', 'legacy')}；优化器状态={'optimizer' in state}")
                for key, default in (("unified_emb", False), ("dct_coeffs", 0),
                                     ("dct_dc", False), ("first_stroke_prior", False),
                                     ("freeze_ed", False), ("free_input_encoder", False)):
                    if state.get('phase') == 'ae_direct_v1' and key in ('first_stroke_prior','freeze_ed','free_input_encoder'):
                        continue
                    print(f"  {key}={cfg.get(key, default)}")
                if state.get('stage') == 'r_pretrain' or state.get('phase') == 'ae_direct_v1':
                    keys = ['latent_units','enc_d_model','enc_layers','enc_heads','dec_hidden','dec_tau_mode','dec_tau_pe_dim']
                    if state.get('stage') == 'r_pretrain':
                        keys += ['r_layers','r_d_model','r_heads','sigma_eps_r','r_sigma_floor_mode','max_context_strokes']
                    else:
                        print('  当前读取AE来源快照；R实际规格以R checkpoint.cfg为准')
                    for key in keys:
                        print(f"  {key}={cfg.get(key)}")
                else:
                    prior = "加 --first_stroke_prior" if cfg.get("first_stroke_prior", False) else "不加 --first_stroke_prior"
                    print(f"旧tf_groups先验参数：{prior}")
            except Exception as exc:
                errors.append(f"checkpoint：{exc}")
    for error in errors:
        print(f"失败：{error}", file=sys.stderr)
    print("核验通过（未训练、未评估模型质量）" if not errors else f"核验失败：{len(errors)} 项")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
