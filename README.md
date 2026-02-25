# TinyML Time-Series Classification on Edge Devices

> **Reproducible TinyML time-series research pipeline — benchmarks, model compression, and on-device evaluation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TFLite](https://img.shields.io/badge/TensorFlow_Lite-Edge_Inference-orange)](https://www.tensorflow.org/lite)

---

## Overview

`tinyml-timeseries` is an end-to-end research repository for **time-series classification** on **resource-constrained microcontrollers** such as the **ESP32** and **ARM Cortex-M** family. The project prioritizes full reproducibility across every stage of the ML lifecycle — from raw data ingestion to on-device firmware export.

```
data → preprocessing → training → evaluation → quantization → edge validation → firmware export
```

The goal is to provide a clean, reproducible baseline that researchers and engineers can use to benchmark different compression strategies and evaluate real-world TinyML performance on embedded hardware.

---

## Key Features

- **Reproducible pipeline** — every experiment is version-controlled and parametrized
- **Model compression** — post-training quantization (INT8 / FP16) and optional pruning
- **Edge validation** — TFLite inference tested against float baseline before deployment
- **Firmware export** — C header array generation for direct use in microcontroller firmware
- **Benchmarking utilities** — accuracy, latency, and model-size trade-off comparisons
- **Jupyter notebooks** — interactive exploration and visualization for each pipeline stage

---

## Repository Structure

```
tinyml-timeseries/
├── data/               # Raw and preprocessed datasets
├── notebooks/          # Jupyter notebooks for exploration and experiments
├── scripts/            # CLI utility scripts (training, quantization, export)
├── src/                # Core library source code (models, preprocessing, evaluation)
├── __init__.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Directory Details

| Directory | Description |
|-----------|-------------|
| `data/` | Houses raw time-series datasets and any preprocessed/cached outputs |
| `notebooks/` | Step-by-step Jupyter notebooks covering data exploration, model training, quantization, and benchmarking |
| `scripts/` | Standalone runnable scripts for each pipeline stage (useful for CI and reproducible runs) |
| `src/` | Importable Python modules — data loaders, model architectures, compression utilities, and evaluation helpers |

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip / virtual environment manager
- (Optional) A supported microcontroller: **ESP32**, **Arduino Nano 33 BLE Sense**, or any **ARM Cortex-M** board

### Installation

```bash
# Clone the repository
git clone https://github.com/TolgaReis/tinyml-timeseries.git
cd tinyml-timeseries

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as a package in editable mode
pip install -e .
```

---

## Pipeline Walkthrough

### 1. Data Preparation

Place your raw time-series CSV/NumPy files in the `data/` directory. The preprocessing utilities in `src/` handle windowing, normalization, and train/val/test splitting.

### 2. Model Training

```bash
python scripts/train.py --config configs/baseline.yaml
```

Supported architectures include lightweight CNNs and RNN variants optimized for small memory footprints.

### 3. Evaluation

```bash
python scripts/evaluate.py --model outputs/model.h5
```

Reports accuracy, F1-score, confusion matrix, and inference latency estimates.

### 4. Quantization

```bash
python scripts/quantize.py --model outputs/model.h5 --mode int8
```

Supports:
- **INT8 post-training quantization** (full integer — recommended for MCUs)
- **FP16 quantization** (for boards with float16 support)

### 5. Edge Validation

The quantized TFLite model is benchmarked against the float baseline to confirm accuracy is preserved within acceptable bounds before hardware deployment.

### 6. Firmware Export

```bash
python scripts/export_to_c.py --tflite outputs/model_int8.tflite
```

Generates a `model_data.h` C header file containing the model as a byte array, ready to be included in an Arduino / ESP-IDF / Zephyr RTOS project.

---

## Notebooks

The `notebooks/` directory contains interactive Jupyter notebooks that walk through each stage.

---

## Target Hardware

| Board | Architecture | Flash | RAM | Supported |
|-------|-------------|-------|-----|-----------|
| ESP32 | Xtensa LX6 (240 MHz) | 4 MB | 520 KB | ✅ |

---

## Dependencies

Core dependencies (see `requirements.txt` for full list):

- `tensorflow` / `tensorflow-lite` — model training and quantization
- `numpy`, `pandas` — data processing
- `scikit-learn` — preprocessing and evaluation metrics
- `matplotlib`, `seaborn` — visualization
- `jupyter` — interactive notebooks

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

Please make sure new experiments are reproducible and documented.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

*Maintained by [TolgaReis](https://github.com/TolgaReis)*