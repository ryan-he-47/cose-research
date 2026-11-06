"""TinyStories → 单线字体基础 ASCII 的有记录规范化（2026-10-02）。

先局部修复 UTF-8 被 cp1252/latin1 误解码的 2–4 字符片段，最多两轮；
再改写标点、空白和可分解的 Latin 重音。未知字符保留并列入 unsupported，
调用方据此隔离样本；本模块不默默删除未知字符。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import unicodedata


NORMALIZATION_VERSION = "2026-10-02-v1"

_PUNCTUATION = {
    **dict.fromkeys("‘’‚‛", "'"),
    **dict.fromkeys("“”„‟", '"'),
    **dict.fromkeys("‐‑‒–—―−", "-"),
    "…": "...",
}


@dataclass(frozen=True)
class TextChange:
    """position 是该修复轮/规范化步骤输入中的字符索引。

    before/after 保留每次替换；round 为乱码修复轮次（1/2），规范化为 0。
    """

    kind: str
    position: int
    before: str
    after: str
    round: int = 0
    encoding: str | None = None


@dataclass(frozen=True)
class NormalizedText:
    text: str
    original_sha256: str
    repairs: tuple[TextChange, ...]
    changes: tuple[TextChange, ...]
    unsupported: tuple[str, ...]

    @property
    def is_supported(self) -> bool:
        return not self.unsupported

    @property
    def change_counts(self) -> dict[str, int]:
        return dict(Counter(change.kind for change in (*self.repairs, *self.changes)))


def _legacy_bytes(fragment: str, encoding: str) -> bytes | None:
    try:
        if encoding == "cp1252+latin1":
            # 部分 cp1252 解码器会把未定义的 C1 字节原样保留为控制字符。
            values = []
            for character in fragment:
                try:
                    values.append(character.encode("cp1252"))
                except UnicodeEncodeError:
                    values.append(character.encode("latin1"))
            return b"".join(values)
        return fragment.encode(encoding)
    except UnicodeEncodeError:
        return None


def _repair_round(text: str, round_number: int) -> tuple[str, list[TextChange]]:
    result, changes = [], []
    index = 0
    while index < len(text):
        if ord(text[index]) < 128:
            result.append(text[index])
            index += 1
            continue
        replacement = None
        for encoding in ("cp1252", "latin1", "cp1252+latin1"):
            leading = _legacy_bytes(text[index], encoding)
            if leading is None or len(leading) != 1:
                continue
            byte = leading[0]
            length = 2 if 0xC2 <= byte <= 0xDF else 3 if 0xE0 <= byte <= 0xEF else 4 if 0xF0 <= byte <= 0xF4 else 0
            if not length or index + length > len(text):
                continue
            fragment = text[index:index + length]
            encoded = _legacy_bytes(fragment, encoding)
            if encoded is None or len(encoded) != length:
                continue
            try:
                decoded = encoded.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if len(decoded) == 1 and decoded.encode("utf-8") == encoded:
                replacement = decoded
                changes.append(TextChange("mojibake", index, fragment, decoded,
                                          round_number, encoding))
                index += length
                break
        if replacement is None:
            result.append(text[index])
            index += 1
        else:
            result.append(replacement)
    return "".join(result), changes


def normalize_text(text: str, *, repair_rounds: int = 2) -> NormalizedText:
    """返回规范化内容及记录；允许 ASCII 32–126 和换行。

    CRLF/CR/Unicode 行分隔符统一为 LF；tab 与 Zs 空格统一为一个普通空格，
    不合并连续空白。é/ñ/ü 等 NFD 后为 Latin 字母 + combining mark 的字符
    转成基础字母，并移除其紧随的组合重音；œ/ø 等不可分解字母继续隔离。
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是 str")
    if not isinstance(repair_rounds, int) or not 0 <= repair_rounds <= 2:
        raise ValueError("repair_rounds 只允许 0、1、2")
    original_sha256 = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
    repaired = text
    repairs = []
    for round_number in range(1, repair_rounds + 1):
        repaired, new_repairs = _repair_round(repaired, round_number)
        repairs.extend(new_repairs)
        if not new_repairs:
            break

    result, changes = [], []
    index = 0
    while index < len(repaired):
        character = repaired[index]
        before, after, kind = character, character, None
        if character == "\r":
            before = "\r\n" if repaired[index:index + 2] == "\r\n" else character
            after, kind = "\n", "newline"
        elif character in ("\u2028", "\u2029"):
            after, kind = "\n", "newline"
        elif character == "\t" or (ord(character) >= 128 and unicodedata.category(character) == "Zs"):
            after, kind = " ", "space"
        elif character in _PUNCTUATION:
            after, kind = _PUNCTUATION[character], "punctuation"
        elif ord(character) >= 128 and repaired[index:index + 2] != "â€":
            # 本地数据有丢失收尾字节的 â€。保留残片，交给 unsupported 隔离；
            # 不把其中的 â 当普通 Latin 重音改写成 a。
            decomposed = unicodedata.normalize("NFD", character)
            if (len(decomposed) > 1 and decomposed[0].isascii() and decomposed[0].isalpha()
                    and all(unicodedata.category(mark) == "Mn" for mark in decomposed[1:])):
                after, kind = decomposed[0], "latin_diacritic"
            elif (unicodedata.category(character) == "Mn" and result
                    and result[-1] and result[-1][-1].isascii() and result[-1][-1].isalpha()):
                # 已分解的 e + combining acute 与预组合 é 使用相同策略。
                after, kind = "", "latin_diacritic"
        if kind:
            changes.append(TextChange(kind, index, before, after))
        if after:
            result.append(after)
        index += len(before)
    normalized = "".join(result)
    unsupported = tuple(sorted({character for character in normalized
                                if character != "\n" and not 32 <= ord(character) <= 126}))
    return NormalizedText(normalized, original_sha256, tuple(repairs), tuple(changes), unsupported)
