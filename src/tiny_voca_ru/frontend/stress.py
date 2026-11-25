"""Russian lexical stress placement (normalized text -> stressed text)."""

from __future__ import annotations

import gzip
import math
import re
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from .omograph import choose_reading, omograph_input_ids

DEFAULT_OMOGRAPH_REPO = "egnch/tiny-voca-ru"
DEFAULT_OMOGRAPH_FILENAME = "onnx/omograph.onnx"
VOWELS = "аеёиоуыэюя"

# Function words normally unstressed (ambiguous words like да, что, то, надо stay out).
CLITICS: frozenset[str] = frozenset(
    [
        "без",
        "безо",
        "в",
        "во",
        "для",
        "до",
        "за",
        "из",
        "изо",
        "к",
        "ко",
        "на",
        "над",
        "о",
        "об",
        "обо",
        "от",
        "ото",
        "по",
        "под",
        "подо",
        "пред",
        "предо",
        "при",
        "про",
        "с",
        "со",
        "у",
        "и",
        "а",
        "но",
        "ни",
        "не",
        "же",
        "ж",
        "ли",
        "ль",
        "бы",
        "б",
    ]
)
ENCLITICS: frozenset[str] = frozenset(["то", "ка", "либо", "нибудь", "таки"])  # кто-то, давай-ка
PROCLITICS: frozenset[str] = frozenset(["кое", "кой"])  # кое-кто
NE_STRESSED: frozenset[str] = frozenset(["был", "было", "были"])  # н+е был, н+е было

_PATRONYMICS: dict[str, str] = {
    w.replace("+", ""): w
    for w in [
        "андр+еевич",
        "алекс+еевна",
        "констант+инович",
        "ром+анович",
        "ив+анович",
        "п+авлович",
        "+игоревич",
        "дм+итриевич",
        "гл+ебович",
    ]
}

# Preferred homograph readings where the first dictionary entry is the rare sense.
PREFERRED: dict[str, str] = {
    w.replace("+", "").replace("ё", "е"): w
    for w in [
        "п+осле",
        "г+ода",
        "год+у",
        "д+ома",
        "вр+емени",
        "вод+ы",
        "г+ости",
        "с+амого",
        "с+амом",
        "с+амой",
        "пот+ом",
        "пор+а",
        "сад+у",
        "берег+у",
        "в+ечера",
        "моём",
        "нём",
        "с+емью",
        "лес+у",
        "семь+и",
        "объём",
        "подъём",
        "п+ятой",
        "пол+у",
        "ед+ы",
        "в+ека",
        "так+ая",
        "том+у",
        "должн+о",
        "м+ало",
        "н+ачал",
        "п+ередал",
        "светл+о",
        "широк+о",
        "ч+асти",
        "с+оли",
        "б+оли",
        "к+ости",
        "ж+еня",
    ]
} | _PATRONYMICS

WORD = re.compile(r"[а-яё+]+(?:-[а-яё+]+)*", re.IGNORECASE)

# Character vocabulary of ruaccent's nn_accent ONNX model in id order.
ACCENT_VOCAB: tuple[str, ...] = (
    "[pad]",
    "[unk]",
    "[bos]",
    "[eos]",
    "'",
    "-",
    ".",
    "?",
    "`",
    "c",
    "e",
    "\u0301",
    *"абвгдежзийклмнопрстуфхцчшщъыьэюяё",
)
_ACCENT_IDS: dict[str, int] = {token: i for i, token in enumerate(ACCENT_VOCAB)}
_ACCENT_MAX = 58  # max_position_embeddings (60) minus [bos], [eos]


def fold(word: str) -> str:
    """Return lowercase dictionary key without ``+`` and with ``ё`` folded to ``е``."""
    return word.lower().replace("+", "").replace("ё", "е")


def canonical(stressed: str) -> str:
    """Remove redundant ``+`` before ``ё``."""
    return stressed.replace("+ё", "ё").replace("+Ё", "Ё")


def implied(word: str) -> str | None:
    """Return implied stress for words with ``ё`` or a single vowel, else ``None``."""
    word = word.lower()
    if "ё" in word:
        return word
    vowels = [i for i, c in enumerate(word) if c in VOWELS]
    if len(vowels) != 1:
        return None
    return word[: vowels[0]] + "+" + word[vowels[0] :]


def apply(word: str, reading: str) -> str:
    """Copy stress marks and ``ё`` from lowercase ``reading`` onto ``word``, preserving case."""
    out: list[str] = []
    letters = iter(word)
    for c in reading:
        out.append(c if c == "+" else c.upper() if next(letters).isupper() else c)
    return "".join(out)


def accent_input_ids(word: str) -> list[int]:
    """Encode a word into input token ids for ``accent.onnx``."""
    return [2, *(_ACCENT_IDS.get(c, 1) for c in word.lower()), 3]


def stress_from_logits(word: str, logits: Sequence[Sequence[float]]) -> str:
    """Place ``+`` before the vowel with highest primary-stress probability."""
    best: int | None = None
    best_p = -1.0
    for i, c in enumerate(word.lower()):
        if c in VOWELS:
            row = [float(x) for x in logits[i + 1]]
            p = 1.0 / sum(math.exp(x - row[1]) for x in row)
            if p > best_p:
                best, best_p = i, p
    return word if best is None else word[:best] + "+" + word[best:]


class StressProvider:
    """Interface mapping normalized Russian text to ``+``-stressed text."""

    name: str = ""

    def accentuate(self, text: str) -> str:
        raise NotImplementedError


class NoStress(StressProvider):
    """Pass-through provider that keeps existing stress marks without adding new ones."""

    name: str = "none"

    def accentuate(self, text: str) -> str:
        return text


class LiteStress(StressProvider):
    """Bundled stress dictionary plus ``accent.onnx`` character model for OOV words."""

    name: str = "lite"

    def __init__(self, predict: Callable[[str], str | None] | None = None) -> None:
        self.extra: dict[str, str] = {}
        self._predict_fn = predict

    @property
    def lexicon(self) -> dict[str, tuple[str, ...]]:
        return _lexicon()

    def readings(self, word: str) -> list[str]:
        """Return stressed readings of ``word`` (most frequent first), or ``[]``."""
        word = word.lower().replace("+", "")
        if word in CLITICS:
            return []
        found: Sequence[str] = self.lexicon.get(fold(word), ())
        if "ё" in word:
            found = [r for r in found if r.replace("+", "") == word]
        if found:
            return list(found)
        stressed = implied(word)
        return [stressed] if stressed else []

    def homograph_readings(self, word: str) -> list[str]:
        """Return spelling-preserving ``+``-marked readings when ``word`` is a homograph."""
        lower = word.lower().replace("+", "")
        cands = [r for r in self.readings(lower) if "+" in r and r.replace("+", "") == lower]
        return cands if len(cands) > 1 else []

    def homographs(self, text: str) -> list[tuple[int, int, list[str]]]:
        """Return ``(start, end, candidates)`` for every homograph span in ``text``."""
        found: list[tuple[int, int, list[str]]] = []
        for match in WORD.finditer(text):
            word = match.group()
            if "+" in word:
                continue
            if "-" in word and self.lexicon.get(fold(word)):
                cands = self.homograph_readings(word)
                if cands:
                    found.append((match.start(), match.end(), cands))
                continue
            pos = match.start()
            for part, lower in zip(word.split("-"), _parts(word), strict=False):
                if lower:
                    cands = self.homograph_readings(lower)
                    if cands:
                        found.append((pos, pos + len(part), cands))
                pos += len(part) + 1
        return found

    def unknown_words(self, text: str) -> list[str]:
        """Return deduplicated OOV words that cannot be stressed without an external model."""
        if self._predict_fn is not None or _accent_session() is not None:
            return []
        unknown: dict[str, None] = {}
        for match in WORD.finditer(text):
            if "+" in match.group():
                continue
            for part in _parts(match.group()):
                if (
                    part
                    and len(part) <= _ACCENT_MAX
                    and any(c in VOWELS for c in part)
                    and self._known(part) is None
                ):
                    unknown[part] = None
        return list(unknown)

    def accentuate(self, text: str) -> str:
        return self.accentuate_with(text, {})

    def accentuate_with(self, text: str, overrides: Mapping[tuple[int, int], str]) -> str:
        """Stress ``text``, using ``overrides[(start, end)]`` for disambiguated homographs."""
        return _rewrite(text, lambda start, word: self._stress_span(start, word, overrides))

    def _known(self, part: str) -> str | None:
        found = self.readings(part)
        if found:
            return found[0]
        return self.extra.get(part) or self.extra.get(fold(part))

    def _predict_word(self, word: str) -> str | None:
        if self._predict_fn is not None:
            return self._predict_fn(word)
        return _predict(word)

    def _stress_span(self, start: int, word: str, overrides: Mapping[tuple[int, int], str]) -> str:
        if "+" in word:
            return word
        full = self.lexicon.get(fold(word)) if "-" in word else None
        if full:
            chosen = overrides.get((start, start + len(word))) or full[0]
            return apply(word, chosen)
        out: list[str] = []
        pos = start
        for part, lower in zip(word.split("-"), _parts(word), strict=False):
            chosen = overrides.get((pos, pos + len(part))) if lower else None
            stressed = (
                (chosen or self._known(lower) or self._predict_word(lower)) if lower else None
            )
            out.append(apply(part, stressed) if stressed and fold(stressed) == fold(part) else part)
            pos += len(part) + 1
        return "-".join(out)


class FullStress(LiteStress):
    """Bundled dictionary + ``accent.onnx`` plus INT8 ``tiny2.1`` homograph resolver."""

    name: str = "full"
    REPO: str = DEFAULT_OMOGRAPH_REPO
    FILENAME: str = DEFAULT_OMOGRAPH_FILENAME

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        predict: Callable[[str], str | None] | None = None,
        score: Callable[[Sequence[Sequence[int]]], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        super().__init__(predict=predict)
        self._score_fn = score
        self._session: Any = None
        if score is None:
            self._session = _load_omograph_session(
                model_path=model_path,
                repo_id=repo_id or self.REPO,
                filename=filename or self.FILENAME,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

    def _score(self, sequences: Sequence[Sequence[int]]) -> list[list[float]]:
        if self._score_fn is not None:
            return [[float(x) for x in row] for row in self._score_fn(sequences)]
        import numpy as np

        logits: list[list[float]] = []
        for seq in sequences:
            ids = np.array([seq], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
            row = self._session.run(["logits"], feed)[0][0]
            logits.append([float(row[0]), float(row[1])])
        return logits

    def accentuate(self, text: str) -> str:
        queries = self.homographs(text)
        if not queries:
            return super().accentuate(text)
        overrides: dict[tuple[int, int], str] = {}
        for start, end, candidates in queries:
            seqs = omograph_input_ids(text, start, end, candidates)
            overrides[(start, end)] = choose_reading(candidates, self._score(seqs))
        return self.accentuate_with(text, overrides)


def _parts(word: str) -> list[str]:
    """Return lowercase hyphen parts, or ``""`` for clitic parts."""
    parts = word.lower().split("-")
    return [
        p
        if p not in CLITICS
        and not (i and p in ENCLITICS)
        and not (i < len(parts) - 1 and p in PROCLITICS)
        else ""
        for i, p in enumerate(parts)
    ]


def _finish(word: str) -> str:
    """Apply output policy: clitics unmarked, no ``+`` before ``ё``."""
    lowers = _parts(word.replace("+", ""))
    return "-".join(
        canonical(p) if lower else p.replace("+", "")
        for p, lower in zip(word.split("-"), lowers, strict=False)
    )


def _rewrite(text: str, stress: Callable[[int, str], str]) -> str:
    """Rewrite each word with ``stress(start, word)`` and apply the ``не был`` phrase rule."""
    matches = list(WORD.finditer(text))
    words = [_finish(stress(m.start(), m.group())) for m in matches]
    for i in range(1, len(words)):
        between = text[matches[i - 1].end() : matches[i].start()]
        if (
            fold(words[i - 1]) == "не"
            and fold(words[i]) in NE_STRESSED
            and between.isspace()
            and "+" not in matches[i - 1].group() + matches[i].group()
        ):
            words[i - 1] = apply(matches[i - 1].group(), "н+е")
            words[i] = words[i].replace("+", "")
    out: list[str] = []
    end = 0
    for match, word in zip(matches, words, strict=False):
        out.extend((text[end : match.start()], word))
        end = match.end()
    out.append(text[end:])
    return "".join(out)


@cache
def _lexicon() -> dict[str, tuple[str, ...]]:
    assert __package__ is not None
    data = resources.files(__package__).joinpath("data").joinpath("stress.txt.gz").read_bytes()
    lexicon: dict[str, tuple[str, ...]] = {}
    for line in gzip.decompress(data).decode("utf-8").splitlines():
        readings = tuple(line.split(" "))
        key = fold(readings[0])
        preferred = PREFERRED.get(key)
        if preferred is not None and preferred in readings:
            readings = (preferred,) + tuple(r for r in readings if r != preferred)
        elif key in _PATRONYMICS and preferred is not None:
            readings = (preferred,)
        lexicon[key] = readings
    return lexicon


@cache
def _accent_session() -> Any:
    try:
        import onnxruntime
    except ImportError:
        return None
    assert __package__ is not None
    model = resources.files(__package__).joinpath("data").joinpath("accent.onnx")
    if not model.is_file():
        return None
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        model.read_bytes(), options, providers=["CPUExecutionProvider"]
    )


@cache
def _predict(word: str) -> str | None:
    session = _accent_session()
    if session is None or len(word) > _ACCENT_MAX or not any(c in VOWELS for c in word):
        return None
    import numpy as np

    ids = np.array([accent_input_ids(word)], dtype=np.int64)
    feed = {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "token_type_ids": np.zeros_like(ids),
    }
    logits = session.run(["logits"], feed)[0][0]
    return stress_from_logits(word, logits)


def _load_omograph_session(
    *,
    model_path: str | Path | None,
    repo_id: str,
    filename: str,
    revision: str | None,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> Any:
    try:
        import onnxruntime
    except ImportError as err:
        raise ImportError(
            "onnxruntime is required for stress='full'; "
            'install with `pip install "tiny-voca-ru[onnx]"`'
        ) from err

    if model_path is not None:
        resolved = Path(model_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"omograph ONNX model not found: {resolved}")
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as err:
            raise ImportError(
                "huggingface-hub is required to download omograph.onnx for stress='full'; "
                'install with `pip install "tiny-voca-ru[onnx]"`'
            ) from err
        resolved = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=local_files_only,
            )
        )

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(str(resolved), options, providers=["CPUExecutionProvider"])


_PROVIDERS: dict[str, type[StressProvider]] = {
    "lite": LiteStress,
    "full": FullStress,
    "none": NoStress,
}


@cache
def get_provider(name: str = "lite") -> StressProvider:
    """Return a cached stress provider by name (``"lite"``, ``"full"``, or ``"none"``)."""
    if name not in _PROVIDERS:
        raise ValueError(f"unknown stress provider {name!r}; expected one of {sorted(_PROVIDERS)}")
    return _PROVIDERS[name]()
