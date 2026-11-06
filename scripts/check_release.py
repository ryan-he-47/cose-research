"""CPU packaging check; the detailed mask/cache check is preserved separately."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.st.page_diffusion import PageDiffusionConfig, PageDiffusionR

torch.set_num_threads(2)
torch.manual_seed(42)
cfg = PageDiffusionConfig(width=64, layers=2, heads=4, ffn=128,
                          condition=16, activation_checkpoint=False)
model = PageDiffusionR(cfg).eval()
x = torch.randn(1, 8, 48)
pages = torch.zeros(1, 8, dtype=torch.long)
local = torch.arange(8)[None]
roles = torch.ones_like(local)
times = torch.full((1, 8), .5)
allowed = torch.ones(1, 1, 8, 8, dtype=torch.bool)
with torch.inference_mode():
    out = model(x, pages, local, roles, times, allowed)['prediction']
assert out.shape == x.shape and torch.isfinite(out).all()
print('PASS: imports and finite CPU page-diffusion forward; no data or weights required')
