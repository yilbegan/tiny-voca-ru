# tiny-voca-ru

English | [Русский](https://github.com/yilbegan/tiny-voca-ru/blob/main/README.ru.md)

A Russian adaptation of the 4.5M-parameter [`tiny-voca`](https://github.com/n4ze3m/tiny-voca) text-to-speech model with female and male voices, fast CPU inference, and client-side browser support.

- **Model:** [`egnch/tiny-voca-ru`](https://huggingface.co/egnch/tiny-voca-ru)
- **Dataset:** [`egnch/tiny-voca-ru-dataset`](https://huggingface.co/datasets/egnch/tiny-voca-ru-dataset)
- **Demo:** [`egnch/tiny-voca-ru-demo`](https://huggingface.co/spaces/egnch/tiny-voca-ru-demo)
- **PyPI:** [`tiny-voca-ru`](https://pypi.org/project/tiny-voca-ru/)

## Installation

Requires Python 3.10+. Works with `pip` or [`uv`](https://docs.astral.sh/uv/).

```bash
# Text frontend only
pip install tiny-voca-ru

# ONNX Runtime inference (no PyTorch)
pip install "tiny-voca-ru[onnx]"

# PyTorch inference + ONNX export
pip install "tiny-voca-ru[torch]"

# Training & custom-voice fine-tuning
pip install "tiny-voca-ru[train]"
```

## Quick Start

`Synthesizer` loads models from Hugging Face Hub or a local directory, selecting PyTorch or ONNX Runtime automatically.

### Python API

```python
from tiny_voca_ru import Synthesizer

# Load from Hugging Face Hub
synth = Synthesizer.from_pretrained("egnch/tiny-voca-ru")
# ONNX Runtime backend
# synth = Synthesizer.from_pretrained("egnch/tiny-voca-ru", backend="onnx")
# Local directory
# synth = Synthesizer("checkpoints/tiny-voca-ru")

result = synth.save_wav(
    "Здравствуйте! Встречу перенесли на 15:30, 14 марта.",
    "hello.wav",
    voice="M",  # "F" or "M"
    speed=1.0,  # 0.25 .. 4.0
    pitch_shift=0.0,  # log-F0 offset (0.058 ≈ +1 semitone)
    pitch_range=1.0,  # 0.0 .. 3.0
    stress="lite",  # "lite" | "full" | "none"
    pause_seconds=0.0,  # pause between sentences (seconds)
)
```

### CLI

```bash
tiny-voca-ru-speak \
  --voice F \
  --text "Привет! Билет стоит 1250 рублей." \
  --output out.wav

# ONNX Runtime backend
tiny-voca-ru-speak \
  --model egnch/tiny-voca-ru \
  --backend onnx \
  --voice M \
  --text "Привет! Билет стоит 1250 рублей." \
  --output out_onnx.wav

# Local checkpoint
tiny-voca-ru-speak \
  --model checkpoints/tiny-voca-ru \
  --voice F \
  --text "Привет! Билет стоит 1250 рублей." \
  --output out_local.wav
```

## Russian Text Frontend

`tiny_voca_ru.frontend` is a pure-Python Russian frontend for text normalization (numbers, dates, times, currencies, units, abbreviations, Latin words), stress placement, and conversion to a 61-symbol IPA sequence.

```python
from tiny_voca_ru import accentuate, normalize, phonemize, transcribe

normalize("В 1999 году билет стоил 12,50 руб.")
accentuate("Я открыл зам+ок ключом.")  # explicit '+' marks are preserved
transcribe("Здравствуйте!")  # readable IPA with ˈ
seq = phonemize("Привет, мир!", stress="lite")
```

| Stress Provider | Dependencies | Notes |
| --- | --- | --- |
| `lite` (default) | Bundled `stress.txt.gz` + `accent.onnx` (`onnxruntime`) | Fast, offline |
| `full` | `tiny-voca-ru[onnx]` (downloads 10 MB INT8 `omograph.onnx`) | Contextual homograph disambiguation (`tiny2.1` INT8) |
| `none` | None | Keeps existing `+` marks only |

## Custom-Voice Fine-Tuning

### 1. Prepare a dataset

Create `metadata.csv` (or `metadata.jsonl`) with `audio`, `text`, and optional `speaker`, `split`, `id`:

```csv
audio,text,speaker
wavs/001.wav,Добрый день! Как ваши дела?,alice
wavs/002.wav,Сегодня отличная погода.,alice
```

Extract 24 kHz features and `wav2vec2` CTC alignments:

```bash
tiny-voca-ru-prepare \
  --metadata data/metadata.csv \
  --audio-root data \
  --output work/alice-prepared \
  --val-ratio 0.05
```

### 2. Fine-tune

`tiny-voca-ru-train` copies compatible weights from a pretrained checkpoint and initializes new speaker embeddings from `--init-voice` (or the mean of pretrained voices):

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

Use `--freeze-encoder` for small datasets (< 30 min) and `--resume work/alice-run/last.pt` to resume. Safetensors checkpoints are written to `work/alice-run/inference/` and `work/alice-run/best/`.

## ONNX Export & Checkpoint Conversion

```bash
# Safetensors -> ONNX
tiny-voca-ru-export \
  --checkpoint checkpoints/tiny-voca-ru \
  --output checkpoints/tiny-voca-ru-onnx

# Legacy .pt -> safetensors
tiny-voca-ru-convert \
  --source path/to/tiny-voca-ru.pt \
  --output checkpoints/tiny-voca-ru
```

## Browser Demo

The static demo in `web/public/` runs entirely in the browser via Pyodide (`tiny_voca_ru.frontend`) and ONNX Runtime Web (`model.onnx` + `accent.onnx` + `omograph.onnx`):

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

## Citation

Metadata is also available in [CITATION.cff](https://github.com/yilbegan/tiny-voca-ru/blob/main/CITATION.cff).

```bibtex
@misc{chetvertkov2026tinyvocaru,
  author = {Chetvertkov, Evgeniy},
  title  = {tiny-voca-ru: A 4.5M-Parameter Russian Text-to-Speech Model},
  year   = {2026},
  url    = {https://huggingface.co/egnch/tiny-voca-ru},
  note   = {ORCID: https://orcid.org/0009-0006-6158-2624}
}
```

Please also cite the original [`tiny-voca`](https://huggingface.co/n4ze3m/tiny-voca) by Muhammed Nazeem (2026).

## License & Attribution

Code and weights: Apache-2.0 ([LICENSE](https://github.com/yilbegan/tiny-voca-ru/blob/main/LICENSE)).
See [NOTICE](https://github.com/yilbegan/tiny-voca-ru/blob/main/NOTICE) for attribution to `tiny-voca` (Apache-2.0) and `ruaccent` (MIT, bundled `stress.txt.gz`, `accent.onnx`, `omograph_vocab.txt.gz`, and quantized `omograph.onnx`).
