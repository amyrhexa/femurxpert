#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null 2>&1 || { echo "Error: 'uv' not found in PATH." >&2; exit 1; }

if [ ! -s "best.onnx" ]; then
  echo "Error: 'best.onnx' is missing or empty." >&2
  exit 1
fi

size=$(wc -c < "best.onnx" | tr -d '[:space:]')
if [ "$size" -lt 1000000 ]; then
  echo "Error: 'best.onnx' is only $size bytes. Run 'git lfs pull'." >&2
  exit 1
fi

uv sync --frozen

extra_flags=()
if [ -f "app.ico" ]; then
  extra_flags+=(--include-data-files=app.ico=app.ico)
fi

uv run python -m nuitka \
  --onefile \
  --enable-plugin=pyside6 \
  --include-package-data=onnxruntime \
  --include-package=cv2 \
  --include-data-files=best.onnx=best.onnx \
  --nofollow-import-to=torch,sympy \
  --low-memory \
  --assume-yes-for-downloads \
  --output-dir=dist \
  --output-filename=FemurXpert \
  ${extra_flags[@]+"${extra_flags[@]}"} \
  femurxpert.py

test -x "dist/FemurXpert" || { echo "Error: dist/FemurXpert was not created." >&2; exit 1; }
echo "Built dist/FemurXpert"
