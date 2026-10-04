# FemurXpert

FemurXpert is a standalone desktop workstation for detecting and segmenting proximal femur fractures on plain radiographs (X-rays). Built with PySide6, OpenCV, and ONNX Runtime, it provides real-time deep learning inference with an interactive radiograph viewer.

---

## Features

- **Real-Time Instance Segmentation:** Detects proximal femoral fracture boundaries and renders polygon contour overlays.
- **Hardware-Accelerated Inference:** Prioritizes DirectML (`DmlExecutionProvider`) on Windows with fallback to CPU execution (`CPUExecutionProvider`)
- **Radiograph Viewer:**
  - Real-time windowing and leveling (contrast and brightness adjustment) using 8-bit lookup tables.
  - Mouse-anchored smooth zooming and panning.
  - Drag-and-drop loading for `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, and `.bmp` images.
  - Dynamic legend panel highlighting detected fracture classes or healthy status.
  - Visibility toggle for segmentation masks and classification labels.
- **PE Resource Embedding:** Supports loading model weights from external files or directly from Windows PE binary resources (`.rsrc`).

---

## Supported Fracture Classes

| ID    | Class Name                    | Color Key                   |
| ----- | ----------------------------- | --------------------------- |
| **0** | Greater Trochanteric Fracture | `#88c0d0` (Frost Ice Blue)  |
| **1** | Intertrochanteric Fracture    | `#bf616a` (Crimson / Coral) |
| **2** | Lesser Trochanteric Fracture  | `#ebcb8b` (Amber Gold)      |
| **3** | Femoral Neck Fracture         | `#d08770` (Orange / Coral)  |
| **4** | Subtrochanteric Fracture      | `#b48ead` (Soft Violet)     |
| **—** | Healthy (No Fracture)         | `#a3be8c` (Sage Green)      |

---

## Project Structure

```text
├── .github/workflows/
│   └── build.yml          # GitHub Actions cross-platform build workflow
├── app.ico                # Application icon
├── best.onnx              # Trained ONNX segmentation model weights
├── build.bat              # Windows Nuitka build script
├── build.sh               # Linux Nuitka build script
├── femurxpert.py          # Application entry point, UI, and inference engine
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Installation & Local Development

### Prerequisites

- Python 3.10+ (tested on Python 3.12)

- Git LFS (for pulling `best.onnx`)

### Setup

```bash
# Clone the repository
git clone https://github.com/amyrhexa/FemurXpert.git
cd FemurXpert

# Ensure LFS assets are downloaded
git lfs pull

# Create and activate a virtual environment
python -m venv .venv
# Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run from Source

Place `best.onnx` and `app.ico` in the root directory alongside `femurxpert.py`:

```bash
python femurxpert.py
```

---

## Standalone Compilation (Nuitka)

The repository provides automated build scripts for standalone packaging.

### Linux Build

Requires `patchelf` and system OpenGL libraries:

```bash
chmod +x build.sh
./build.sh
```

The resulting standalone distribution is placed in `femurxpert.dist/FemurXpert`.

### Windows Build

Run the batch script:

```cmd
build.bat
```

The compiled standalone executable is placed in `femurxpert.dist\FemurXpert.exe`.

### Embed Model into Windows PE Resources (Optional)

Inject `best.onnx` directly into the compiled executable's Win32 resource section (`XRAYMODEL`/`MODEL`) using the built-in CLI command:

```cmd
python femurxpert.py --embed femurxpert.dist\FemurXpert.exe best.onnx
```

---

## Viewer Controls

- **Left-Click + Drag:** Adjust Contrast (Window Width).

- **Right-Click + Drag:** Adjust Brightness (Window Level).

- **Middle-Click + Drag:** Pan image.

- **Scroll Wheel:** Zoom in / out anchored under cursor.

- **Reset Button (Bottom-Left Circular Arrow):** Reset zoom, pan, and contrast/brightness levels.

- **Eye Button (Bottom-Left):** Toggle segmentation masks and legend visibility.

- **Drag-and-Drop:** Drop any supported radiograph image onto the window to run inference.
