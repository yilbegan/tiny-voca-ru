"""Pure-Python WordPiece tokenizer and input helpers for the INT8 tiny2.1 homograph model."""

from __future__ import annotations

import gzip
import re
import unicodedata
from collections.abc import Sequence
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SEP_ID = 3
W_OPEN_ID = 30522
W_CLOSE_ID = 30523

MAX_LENGTH = 512
MAX_WORD_CHARS = 100

# Pinned upstream source for reproducible quantization of tiny2.1 FP32 -> INT8.
RUACCENT_REPO = "ruaccent/accentuator"
RUACCENT_REVISION = "b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855"
RUACCENT_MODEL_FILE = "nn/nn_omograph/tiny2.1/model.onnx"

_SENTENCE_BREAK = re.compile(r"""(?<=[.!?…])(?:["')\]»“])?\s+|\n+""")


def _is_whitespace(char: str) -> bool:
    return char in (" ", "\t", "\n", "\r") or unicodedata.category(char) == "Zs"


def _is_control(char: str) -> bool:
    if char in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(char).startswith("C")


def _is_punctuation(char: str) -> bool:
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(char).startswith("P")


def _is_chinese_char(cp: int) -> bool:
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


def _basic_tokens(text: str) -> list[str]:
    """Match HuggingFace ``BertNormalizer(lowercase=False)`` + ``BertPreTokenizer``."""
    cleaned: list[str] = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            cleaned.append(" ")
        elif _is_chinese_char(cp):
            cleaned.extend((" ", char, " "))
        else:
            cleaned.append(char)

    tokens: list[str] = []
    for piece in "".join(cleaned).split():
        buf: list[str] = []
        for char in piece:
            if _is_punctuation(char):
                if buf:
                    tokens.append("".join(buf))
                    buf.clear()
                tokens.append(char)
            else:
                buf.append(char)
        if buf:
            tokens.append("".join(buf))
    return tokens


@cache
def _vocab() -> dict[str, int]:
    assert __package__ is not None
    raw = (
        resources.files(__package__).joinpath("data").joinpath("omograph_vocab.txt.gz").read_bytes()
    )
    lines = gzip.decompress(raw).decode("utf-8").splitlines()
    return {token: idx for idx, token in enumerate(lines)}


def _wordpiece(token: str, vocab: dict[str, int]) -> list[int]:
    if len(token) > MAX_WORD_CHARS:
        return [UNK_ID]
    ids: list[int] = []
    start = 0
    end_limit = len(token)
    while start < end_limit:
        end = end_limit
        found: int | None = None
        while start < end:
            substr = token[start:end] if start == 0 else "##" + token[start:end]
            token_id = vocab.get(substr)
            if token_id is not None:
                found = token_id
                break
            end -= 1
        if found is None:
            return [UNK_ID]
        ids.append(found)
        start = end
    return ids


def wordpiece_ids(text: str) -> list[int]:
    """Encode ``text`` into WordPiece token IDs using the bundled ``tiny2.1`` vocabulary."""
    vocab = _vocab()
    ids: list[int] = []
    for token in _basic_tokens(text):
        ids.extend(_wordpiece(token, vocab))
    return ids


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Return ``(sent_start, sent_end)`` bounds for the sentence around ``text[start:end]``."""
    sent_start = 0
    for match in _SENTENCE_BREAK.finditer(text, 0, start):
        sent_start = match.end()
    after = _SENTENCE_BREAK.search(text, end)
    sent_end = after.start() if after is not None else len(text)
    return sent_start, sent_end


def _trim_context(
    left: list[int],
    right: list[int],
    budget: int,
) -> tuple[list[int], list[int]]:
    if budget <= 0:
        return [], []
    if len(left) + len(right) <= budget:
        return left, right
    half = budget // 2
    if len(left) <= half:
        return left, right[: budget - len(left)]
    if len(right) <= budget - half:
        return left[-(budget - len(right)) :], right
    return left[-half:], right[: budget - half]


def omograph_input_ids(
    text: str,
    start: int,
    end: int,
    candidates: Sequence[str],
) -> list[list[int]]:
    """Build ``tiny2.1`` ``input_ids`` for each candidate reading of ``text[start:end]``."""
    sent_start, sent_end = _sentence_bounds(text, start, end)
    left_ids = wordpiece_ids(text[sent_start:start].replace("+", "").lower())
    word_ids = wordpiece_ids(text[start:end].replace("+", "").lower())
    right_ids = wordpiece_ids(text[end:sent_end].replace("+", "").lower())

    sequences: list[list[int]] = []
    for candidate in candidates:
        cand_ids = wordpiece_ids(candidate.lower())
        # Overhead: [CLS], <w>, </w>, [SEP], [SEP] = 5 tokens.
        budget = MAX_LENGTH - len(word_ids) - len(cand_ids) - 5
        if budget >= 0:
            kept_left, kept_right = _trim_context(left_ids, right_ids, budget)
            seq_word = word_ids
            seq_cand = cand_ids
        else:
            kept_left, kept_right = [], []
            cap = MAX_LENGTH - 5
            half = cap // 2
            seq_word = word_ids[:half]
            seq_cand = cand_ids[: cap - len(seq_word)]
        sequences.append(
            [
                CLS_ID,
                *kept_left,
                W_OPEN_ID,
                *seq_word,
                W_CLOSE_ID,
                *kept_right,
                SEP_ID,
                *seq_cand,
                SEP_ID,
            ]
        )
    return sequences


def choose_reading(candidates: Sequence[str], logits: Sequence[Sequence[float]]) -> str:
    """Select the candidate reading with highest positive class score ``logit[1] - logit[0]``."""
    if not candidates or len(candidates) != len(logits):
        raise ValueError("candidates and logits must be non-empty and have the same length")
    best_index = 0
    best_score = float(logits[0][1]) - float(logits[0][0])
    for index in range(1, len(candidates)):
        score = float(logits[index][1]) - float(logits[index][0])
        if score > best_score:
            best_score = score
            best_index = index
    return candidates[best_index]


def export_omograph_onnx(
    output_path: str | Path,
    *,
    source_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Quantize ``ruaccent/accentuator`` ``tiny2.1`` (FP32) into INT8 ``omograph.onnx``."""
    from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore[import-untyped]

    if source_path is None:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=RUACCENT_REPO,
            filename=RUACCENT_MODEL_FILE,
            revision=RUACCENT_REVISION,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        src = Path(downloaded)
    else:
        src = Path(source_path)

    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_fn: Any = quantize_dynamic
    quantize_fn(
        str(src),
        str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gather"],
    )
    return dst
