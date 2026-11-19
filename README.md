# tiny-voca-ru

A lightweight 4.5M-parameter Russian text-to-speech model with a pure-Python Russian text frontend, safetensors checkpoints, ONNX Runtime inference, custom-voice fine-tuning, and a client-side browser demo.

- **Architecture:** Non-autoregressive rotary transformer with duration/prosody predictors and a pitch-driven harmonic-plus-noise source-filter vocoder (4,534,411 parameters, 24 kHz mono).
- **Voices:** Two built-in synthetic voices (`F` and `M`), plus single-command fine-tuning to adapt to a custom Russian speaker.
- **Zero-Torch Frontend:** `tiny_voca_ru.frontend` has no PyTorch dependency (only stdlib + optional `onnxruntime` for OOV stress), so it runs cleanly in lightweight environments and Pyodide.

## Installation

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/) (or `pip`).

```bash
# Minimal Russian text frontend only
uv pip install -e .

# Lightweight ONNX Runtime inference (no PyTorch)
uv pip install -e ".[onnx]"

# PyTorch inference + safetensors + ONNX export
uv pip install -e ".[torch]"

# Full training & custom-voice fine-tuning stack
uv pip install -e ".[train]"

# Optional full ruaccent homograph disambiguation
uv pip install -e ".[ruaccent]"
```

## Quick Start: Inference

`Synthesizer` automatically selects the PyTorch backend for safetensors checkpoint directories (`model.safetensors` + `config.json`) and the ONNX Runtime backend for exported ONNX directories (`model.onnx` + `model.json`).

### Python API

```python
from tiny_voca_ru import Synthesizer

# Works with either a safetensors directory or an ONNX directory
synth = Synthesizer("checkpoints/tiny-voca-ru")
# synth = Synthesizer("checkpoints/tiny-voca-ru-onnx")

result = synth.save_wav(
    "Здравствуйте! Встречу перенесли на 15:30, 14 марта.",
    "hello.wav",
    voice="M",  # "F" or "M"
    speed=1.0,  # 0.25 .. 4.0
    pitch_shift=0.0,  # log-F0 offset (0.058 ≈ +1 semitone)
    pitch_range=1.0,  # 0.0 .. 3.0
    stress="lite",  # "lite" | "ruaccent" | "none"
    pause_seconds=0.0,  # optional silence gap between sentences
)
```

### CLI

```bash
tiny-voca-ru-speak \
  --model checkpoints/tiny-voca-ru \
  --voice F \
  --text "Привет! Билет стоит 1250 рублей." \
  --output out.wav

# Same CLI with lightweight ONNX Runtime backend
tiny-voca-ru-speak \
  --model checkpoints/tiny-voca-ru-onnx \
  --voice M \
  --text "Привет! Билет стоит 1250 рублей." \
  --output out_onnx.wav
```

## Russian Text Frontend

`tiny_voca_ru.frontend` normalizes Russian text (numbers, dates, times, currencies, units, Roman numerals, abbreviations, acronyms, Latin words), places lexical stress, and produces the 61-symbol IPA phoneme sequence.

```python
from tiny_voca_ru import accentuate, normalize, phonemize, transcribe

normalize("В 1999 году билет стоил 12,50 руб.")
accentuate("Я открыл зам+ок ключом.")  # explicit '+' marks are preserved
transcribe("Здравствуйте!")  # readable IPA with ˈ
seq = phonemize("Привет, мир!", stress="lite")  # PhoneSequence(symbols, ids, stress, unknown)
```

### Stress Providers

| Provider | Dependencies | Notes |
| --- | --- | --- |
| `lite` (default) | Bundled `stress.txt.gz` + `accent.onnx` (`onnxruntime`) | Fast, offline, works in browser via JS `accent.onnx` |
| `ruaccent` | `.[ruaccent]` | Full `ruaccent` (`turbo3.1`) neural homograph resolver |
| `none` | None | Leaves existing `+` stress marks untouched |

## Custom-Voice Fine-Tuning

### 1. Prepare a dataset

Create a `metadata.csv` (or `metadata.jsonl`) with columns `audio`, `text`, and optional `speaker`, `split`, `id`:

```csv
audio,text,speaker
wavs/001.wav,Добрый день! Как ваши дела?,alice
wavs/002.wav,Сегодня отличная погода.,alice
```

Run feature extraction (24 kHz mono resampling, mel, `pyin` pitch/voicing, RMS) and `wav2vec2` CTC forced alignment:

```bash
tiny-voca-ru-prepare \
  --metadata data/metadata.csv \
  --audio-root data \
  --output work/alice-prepared \
  --val-ratio 0.05
```

### 2. Fine-tune from a pretrained checkpoint

When fine-tuning onto a new speaker list (e.g., pretrained `("F", "M")` -> custom `("alice",)`), `tiny-voca-ru-train` copies all compatible weights and initializes the new speaker embeddings from `--init-voice` (or the mean of pretrained voices if omitted):

```bash
tiny-voca-ru-train \
  --data-dir work/alice-prepared \
  --init-from checkpoints/tiny-voca-ru \
  --init-voice F \
  --output work/alice-run \
  --steps 5000 \
  --learning-rate 1e-4 \
  --save-every 500
```

Add `--freeze-encoder` for small datasets (< 30 min) to keep the phoneme encoder fixed while adapting duration, prosody, and acoustics. Resume an interrupted run with `--resume work/alice-run/last.pt`. Clean safetensors checkpoints are written to `work/alice-run/inference/` and `work/alice-run/best/`.

## ONNX Export & Legacy Checkpoint Conversion

```bash
# Export safetensors checkpoint -> model.onnx + model.json
tiny-voca-ru-export \
  --checkpoint checkpoints/tiny-voca-ru \
  --output checkpoints/tiny-voca-ru-onnx

# Convert a legacy .pt checkpoint -> model.safetensors + config.json
tiny-voca-ru-convert \
  --source path/to/tiny-voca-ru.pt \
  --output checkpoints/tiny-voca-ru
```

## Browser Demo

The static demo in `web/public/` runs `tiny_voca_ru.frontend` inside Pyodide and executes `model.onnx` + `accent.onnx` via ONNX Runtime Web completely offline in the browser:

```bash
bash web/build_assets.sh checkpoints/tiny-voca-ru-onnx
python3 -m http.server --directory web/public 8000
```

## Development

```bash
uv sync --python 3.12 --extra torch --extra onnx --extra train --group dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
```

## License & Attribution

Code is licensed under the Apache License 2.0 ([LICENSE](LICENSE)).
See [NOTICE](NOTICE) for attribution to `tiny-voca` (Muhammed Nazeem, Apache-2.0) and `ruaccent` (MIT License, bundled stress dictionary and character accent model in `src/tiny_voca_ru/frontend/data/`).
