#!/usr/bin/env bash
# Build static demo assets into public/assets (or a custom target directory):
#
#   bash web/build_assets.sh checkpoints/tiny-voca-ru-onnx [output_dir]
#
# Packages model.onnx, config.json, accent.onnx, omograph.onnx, and phonemizer.zip
# (tiny_voca_ru.frontend + stress.txt.gz + omograph_vocab.txt.gz for Pyodide, without .onnx files).
set -euo pipefail

INPUT="$(realpath "${1:?path to ONNX directory (model.onnx + model.json) or model.onnx file}")"
if [ -d "$INPUT" ]; then
  MODEL="$INPUT/model.onnx"
  META="$INPUT/model.json"
  SRC_DIR="$INPUT"
else
  MODEL="$INPUT"
  META="${INPUT%.onnx}.json"
  SRC_DIR="$(dirname "$INPUT")"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${2:-$SCRIPT_DIR/public/assets}"
mkdir -p "$OUT_DIR"

cp "$MODEL" "$OUT_DIR/model.onnx"
cp "$SCRIPT_DIR/../src/tiny_voca_ru/frontend/data/accent.onnx" "$OUT_DIR/accent.onnx"

if [ -f "$SRC_DIR/omograph.onnx" ]; then
  cp "$SRC_DIR/omograph.onnx" "$OUT_DIR/omograph.onnx"
elif [ ! -f "$OUT_DIR/omograph.onnx" ]; then
  PYTHON="${PYTHON:-python3}"
  if [ -x "$SCRIPT_DIR/../.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/../.venv/bin/python"
  fi
  PYTHONPATH="$SCRIPT_DIR/../src" "$PYTHON" - "$OUT_DIR/omograph.onnx" <<'PY'
import sys
from tiny_voca_ru.frontend.omograph import export_omograph_onnx

export_omograph_onnx(sys.argv[1])
PY
fi
python3 - "$META" "$OUT_DIR/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    meta = json.load(handle)
meta["model"] = {"max_tokens": meta["max_tokens"], "max_frames": meta["max_frames"]}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    handle.write(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
PY

python3 - "$SCRIPT_DIR/../src" "$OUT_DIR/phonemizer.zip" <<'PY'
import pathlib
import sys
import zipfile

src = pathlib.Path(sys.argv[1])
pkg = src / "tiny_voca_ru"
with zipfile.ZipFile(sys.argv[2], "w", zipfile.ZIP_DEFLATED) as archive:
    for top in ("__init__.py", "config.py"):
        archive.write(pkg / top, f"tiny_voca_ru/{top}")
    for path in sorted((pkg / "frontend").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".onnx"):
            archive.write(path, path.relative_to(src).as_posix())
PY

ls -lh "$OUT_DIR"
