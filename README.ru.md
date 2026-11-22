# tiny-voca-ru

[English](README.md) | Русский

Русская адаптация компактной модели синтеза речи [`tiny-voca`](https://github.com/n4ze3m/tiny-voca) на 4,5 млн параметров: женский и мужской голоса, быстрый инференс на CPU и работа прямо в браузере.

- **Модель:** [`egnch/tiny-voca-ru`](https://huggingface.co/egnch/tiny-voca-ru)
- **Датасет:** [`egnch/tiny-voca-ru-dataset`](https://huggingface.co/datasets/egnch/tiny-voca-ru-dataset)
- **Демо:** [`egnch/tiny-voca-ru-demo`](https://huggingface.co/spaces/egnch/tiny-voca-ru-demo)
- **PyPI:** [`tiny-voca-ru`](https://pypi.org/project/tiny-voca-ru/)

## Установка

Требуется Python 3.11+. Поддерживаются `pip` и [`uv`](https://docs.astral.sh/uv/).

```bash
# Только текстовый фронтенд
pip install tiny-voca-ru

# Инференс на ONNX Runtime (без PyTorch)
pip install "tiny-voca-ru[onnx]"

# Инференс на PyTorch + экспорт в ONNX
pip install "tiny-voca-ru[torch]"

# Обучение и дообучение на свой голос
pip install "tiny-voca-ru[train]"
```

## Быстрый старт

`Synthesizer` загружает модели с Hugging Face Hub или из локальной папки и автоматически выбирает PyTorch или ONNX Runtime.

### Python API

```python
from tiny_voca_ru import Synthesizer

# Загрузка с Hugging Face Hub
synth = Synthesizer.from_pretrained("egnch/tiny-voca-ru")
# Версия на ONNX Runtime
# synth = Synthesizer.from_pretrained("egnch/tiny-voca-ru", backend="onnx")
# Загрузка из локальной папки
# synth = Synthesizer("checkpoints/tiny-voca-ru")

result = synth.save_wav(
    "Здравствуйте! Встречу перенесли на 15:30, 14 марта.",
    "hello.wav",
    voice="M",  # "F" или "M"
    speed=1.0,  # 0.25 .. 4.0
    pitch_shift=0.0,  # сдвиг log-F0 (0.058 ≈ +1 полутон)
    pitch_range=1.0,  # 0.0 .. 3.0
    stress="lite",  # "lite" | "full" | "none"
    pause_seconds=0.0,  # пауза между предложениями (в секундах)
)
```

### CLI

```bash
tiny-voca-ru-speak \
  --voice F \
  --text "Привет! Билет стоит 1250 рублей." \
  --output out.wav

# Версия на ONNX Runtime
tiny-voca-ru-speak \
  --model egnch/tiny-voca-ru \
  --backend onnx \
  --voice M \
  --text "Привет! Билет стоит 1250 рублей." \
  --output out_onnx.wav

# Локальный чекпоинт
tiny-voca-ru-speak \
  --model checkpoints/tiny-voca-ru \
  --voice F \
  --text "Привет! Билет стоит 1250 рублей." \
  --output out_local.wav
```

## Русский текстовый фронтенд

`tiny_voca_ru.frontend` — русский фронтенд на чистом Python для нормализации текста (числа, даты, время, валюты, единицы измерения, сокращения, латиница), расстановки ударений и преобразования текста в последовательность из 61 IPA-символа.

```python
from tiny_voca_ru import accentuate, normalize, phonemize, transcribe

normalize("В 1999 году билет стоил 12,50 руб.")
accentuate("Я открыл зам+ок ключом.")  # явные '+' сохраняются
transcribe("Здравствуйте!")  # читаемая транскрипция IPA с ˈ
seq = phonemize("Привет, мир!", stress="lite")
```

| Провайдер ударений | Зависимости | Примечания |
| --- | --- | --- |
| `lite` (по умолчанию) | Встроенные `stress.txt.gz` + `accent.onnx` (`onnxruntime`) | Быстрый, полностью оффлайн |
| `full` | `tiny-voca-ru[onnx]` (скачивает 10 МБ INT8 `omograph.onnx`) | Контекстное разрешение омографов (`tiny2.1` INT8) |
| `none` | Нет | Оставляет только проставленные знаки `+` |

## Дообучение на свой голос

### 1. Подготовка датасета

Создайте `metadata.csv` (или `metadata.jsonl`) с колонками `audio`, `text` и опциональными `speaker`, `split`, `id`:

```csv
audio,text,speaker
wavs/001.wav,Добрый день! Как ваши дела?,alice
wavs/002.wav,Сегодня отличная погода.,alice
```

Извлеките признаки (24 кГц) и CTC-выравнивание через `wav2vec2`:

```bash
tiny-voca-ru-prepare \
  --metadata data/metadata.csv \
  --audio-root data \
  --output work/alice-prepared \
  --val-ratio 0.05
```

### 2. Дообучение

`tiny-voca-ru-train` копирует совместимые веса из предобученного чекпоинта и инициализирует эмбеддинги новых дикторов из `--init-voice` (или среднего по предобученным голосам):

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

Для небольших датасетов (< 30 мин) добавьте `--freeze-encoder`, для продолжения обучения — `--resume work/alice-run/last.pt`. Чекпоинты в формате safetensors сохраняются в `work/alice-run/inference/` и `work/alice-run/best/`.

## Экспорт в ONNX и конвертация чекпоинтов

```bash
# Safetensors -> ONNX
tiny-voca-ru-export \
  --checkpoint checkpoints/tiny-voca-ru \
  --output checkpoints/tiny-voca-ru-onnx

# Старый .pt -> safetensors
tiny-voca-ru-convert \
  --source path/to/tiny-voca-ru.pt \
  --output checkpoints/tiny-voca-ru
```

## Браузерное демо

Статическое демо в `web/public/` работает полностью в браузере через Pyodide (`tiny_voca_ru.frontend`) и ONNX Runtime Web (`model.onnx` + `accent.onnx` + `omograph.onnx`):

```bash
bash web/build_assets.sh checkpoints/tiny-voca-ru-onnx
python3 -m http.server --directory web/public 8000
```

## Разработка

```bash
uv sync --python 3.12 --extra torch --extra onnx --extra train --group dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
```

## Цитирование

Метаданные цитирования также доступны в [CITATION.cff](CITATION.cff).

```bibtex
@misc{chetvertkov2026tinyvocaru,
  author = {Chetvertkov, Evgeniy},
  title  = {tiny-voca-ru: A 4.5M-Parameter Russian Text-to-Speech Model},
  year   = {2026},
  url    = {https://huggingface.co/egnch/tiny-voca-ru},
  note   = {ORCID: https://orcid.org/0009-0006-6158-2624}
}
```

При использовании также рекомендуем цитировать оригинальный [`tiny-voca`](https://huggingface.co/n4ze3m/tiny-voca) (Muhammed Nazeem, 2026).

## Лицензия и атрибуция

Код и веса: Apache-2.0 ([LICENSE](LICENSE)).
См. [NOTICE](NOTICE) для атрибуции `tiny-voca` (Apache-2.0) и `ruaccent` (MIT, встроенные `stress.txt.gz`, `accent.onnx`, `omograph_vocab.txt.gz` и квантованная `omograph.onnx`).
