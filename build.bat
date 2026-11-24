@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
  echo Error: 'uv' not found in PATH.
  exit /b 1
)

if not exist "best.onnx" (
  echo Error: 'best.onnx' is missing.
  exit /b 1
)

rem shortcut: cmd 32-bit int comparison assumes best.onnx is under 2GB
for %%A in ("best.onnx") do set "SIZE=%%~zA"
if %SIZE% LSS 1000000 (
  echo Error: 'best.onnx' is only %SIZE% bytes. Run 'git lfs pull'.
  exit /b 1
)

uv sync --frozen || exit /b 1

set "EXTRA_FLAGS="
if exist "app.ico" set "EXTRA_FLAGS=--include-data-files=app.ico=app.ico --windows-icon-from-ico=app.ico"

uv run python -m nuitka ^
  --onefile ^
  --enable-plugin=pyside6 ^
  --include-package-data=onnxruntime ^
  --include-package=cv2 ^
  --include-data-files=best.onnx=best.onnx ^
  --nofollow-import-to=torch,sympy ^
  --low-memory ^
  --windows-console-mode=disable ^
  --assume-yes-for-downloads ^
  --output-dir=dist ^
  --output-filename=FemurXpert.exe ^
  %EXTRA_FLAGS% ^
  femurxpert.py
if errorlevel 1 exit /b 1

if not exist "dist\FemurXpert.exe" (
  echo Error: dist\FemurXpert.exe was not created.
  exit /b 1
)
echo Built dist\FemurXpert.exe
