"""Russian phoneme and token inventory (61 symbols).

One token is one IPA phone; lexical stress is a parallel 0/1/2 feature.
Order is part of the checkpoint contract and must not change.
"""

from __future__ import annotations

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
SP = "<sp>"
UNK = "<unk>"

SPECIALS: tuple[str, ...] = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION: tuple[str, ...] = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
PAUSABLE: tuple[str, ...] = SPECIALS + PUNCTUATION

CONSONANTS: tuple[str, ...] = (
    # stops (hard / soft)
    "p",
    "pʲ",
    "b",
    "bʲ",
    "t",
    "tʲ",
    "d",
    "dʲ",
    "k",
    "kʲ",
    "ɡ",
    "ɡʲ",
    # fricatives (hard / soft)
    "f",
    "fʲ",
    "v",
    "vʲ",
    "s",
    "sʲ",
    "z",
    "zʲ",
    # ш ж (always hard), щ, х
    "ʂ",
    "ʐ",
    "ɕ",
    "x",
    "xʲ",
    # ц (always hard), ч (always soft)
    "ts",
    "tɕ",
    # sonorants
    "m",
    "mʲ",
    "n",
    "nʲ",
    "l",
    "lʲ",
    "r",
    "rʲ",
    "j",
)

# Full vowels plus reduced ɐ (first pretonic / word-initial) and ə (other unstressed а/о).
VOWELS: tuple[str, ...] = ("a", "o", "e", "i", "u", "ɨ", "ɐ", "ə")
PHONES: tuple[str, ...] = CONSONANTS + VOWELS
SYMBOLS: tuple[str, ...] = SPECIALS + PUNCTUATION + PHONES

SYMBOL_TO_ID: dict[str, int] = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_SET: frozenset[str] = frozenset(PHONES)
VOWEL_SET: frozenset[str] = frozenset(VOWELS)
PHONE_IDS: frozenset[int] = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}
VOWEL_IDS: frozenset[int] = frozenset(SYMBOL_TO_ID[v] for v in VOWELS)

# Paired hard/soft consonants (ж ш ц are always hard; ч щ й are always soft).
SOFT: dict[str, str] = {
    hard: hard + "ʲ"
    for hard in ("p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "x", "m", "n", "l", "r")
}
HARD: dict[str, str] = {soft: hard for hard, soft in SOFT.items()}

VOICELESS: frozenset[str] = frozenset(
    ("p", "pʲ", "t", "tʲ", "k", "kʲ", "f", "fʲ", "s", "sʲ", "ʂ", "ɕ", "x", "xʲ", "ts", "tɕ")
)
# х, ц, ч, щ have no voiced partner in the inventory and stay voiceless.
VOICED_PARTNER: dict[str, str] = {
    "p": "b",
    "pʲ": "bʲ",
    "t": "d",
    "tʲ": "dʲ",
    "k": "ɡ",
    "kʲ": "ɡʲ",
    "f": "v",
    "fʲ": "vʲ",
    "s": "z",
    "sʲ": "zʲ",
    "ʂ": "ʐ",
}
VOICELESS_PARTNER: dict[str, str] = {v: k for k, v in VOICED_PARTNER.items()}
OBSTRUENTS: frozenset[str] = VOICELESS | frozenset(VOICELESS_PARTNER)
SONORANTS: frozenset[str] = frozenset(("m", "mʲ", "n", "nʲ", "l", "lʲ", "r", "rʲ", "j"))

if len(set(SYMBOLS)) != len(SYMBOLS) or len(SYMBOLS) != 61:
    raise AssertionError("the inventory must contain 61 distinct symbols")
