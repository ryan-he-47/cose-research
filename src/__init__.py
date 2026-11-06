"""CoSE 时空笔画项目（src/ 公共 API，2026-09-30 重组后）。

活跃栈 = src/st/（时空统一嵌入架构）。本层只保留共享基础件：
composite（排版/画布）、letter_database（手写字母库）、precache（离线预缓存）。
"""
from src.composite import TextLayoutConfig, typeset_text, fit_graphic_to_box
from src.letter_database import LetterDatabase
from src.precache import build_quickdraw_precache

__all__ = [
    "TextLayoutConfig",
    "typeset_text",
    "fit_graphic_to_box",
    "LetterDatabase",
    "build_quickdraw_precache",
]
