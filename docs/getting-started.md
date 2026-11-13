# Getting Started & Pipeline Execution

This guide walks through configuring the runtime environment, downloading models from Hugging Face, converting them to ONNX, quantizing, validating, and generating Android packages.

---

## 1. Prerequisites

### System Requirements
- Linux (x86_64 or aarch64) or macOS.
- At least 8 GB of RAM.
- At least 5 GB of free disk space.
- Git & Git LFS installed.

> [!IMPORTANT]
> **Use Python 3.11**. Modern Linux distributions (such as Arch or Ubuntu 25+) may ship with Python 3.13 or 3.14. PyTorch and Hugging Face `tokenizers` (which depend on PyO3) do not have binary wheels for Python 3.14 and will fail to compile from source.

---

## 2. Setting Up the Environment

### Option A: Using `mise` (Recommended)
If `mise` is installed:
```bash
# Install Python 3.11.18
mise install python@3.11.18

# Create a virtual environment using Python 3.11
rm -rf venv
$(mise which python@3.11.18) -m venv venv
source venv/bin/activate
```

### Option B: Using standard Python 3.11
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Install Pinned Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify that the environment loads correctly:
```bash
./venv/bin/python -c "import torch, transformers, onnx, onnxruntime, librosa, jiwer; print('✓ All ML dependencies verified!')"
```

---

## 3. Quickstart: One-Command Pipeline

To execute all steps (download, inspection, FP32 export, FP16 export, INT8 quantization, validation, benchmarking, and Android packaging) in a single run:

```bash
./venv/bin/python ethio-asr-android-builder/scripts/run_pipeline.py --model ethio-94m --precision all
```

Expected output directory structure upon completion:
```text
output/
├── android/
│   ├── EthioASRRecognizer.kt        # Android Kotlin wrapper
│   ├── ethio-94m-int8/             # Ready-to-deploy INT8 bundle
│   ├── ethio-94m-fp16/             # FP16 bundle
│   └── ethio-94m-fp32/             # FP32 bundle
└── reports/
    ├── model-inspection.json       # Extracted architecture metadata
    ├── validation_ethio-94m.json   # Multi-language numerical & WER validation
    ├── benchmark_ethio-94m.csv     # Detailed timing & memory benchmark
    └── benchmark_ethio-94m.json    # JSON formatted benchmark data
```

---

## 4. Step-by-Step Script Walkthrough

If you want granular control over individual stages, run the scripts independently:

### Step 1: Download Original Hugging Face Model
Downloads repository weights and generates `download_meta.json` with commit SHAs and license info:
```bash
./venv/bin/python ethio-asr-android-builder/scripts/download_models.py --model ethio-94m
```

### Step 2: Inspect Model Architecture
Inspects layer count, feature encoder strides, tensor names, and CTC parameters:
```bash
./venv/bin/python ethio-asr-android-builder/scripts/inspect_models.py --model ethio-94m
```

### Step 3: Export FP32 ONNX
Converts PyTorch `HubertForCTC` to an ONNX graph with dynamic axes and opset 17:
```bash
./venv/bin/python ethio-asr-android-builder/scripts/export_onnx.py --model ethio-94m
```
*Output: `models/onnx/ethio-94m/model.fp32.onnx` (361.45 MB)*

### Step 4: Export FP16 ONNX
Exports directly from PyTorch half-precision weights:
```bash
./venv/bin/python ethio-asr-android-builder/scripts/convert_fp16.py --model ethio-94m
```
*Output: `models/onnx/ethio-94m/model.fp16.onnx` (180.84 MB)*

### Step 5: Quantize to INT8
Applies dynamic integer quantization on matrix multiplication operations:
```bash
./venv/bin/python ethio-asr-android-builder/scripts/quantize_int8.py --model ethio-94m
```
*Output: `models/quantized/ethio-94m/model.int8.onnx` (117.00 MB)*

### Step 6: Validate Against Multi-Language Dataset
Compares output logits and decoded text across Amharic, Afaan Oromo, Tigrinya, Sidama, and Wolaytta:
```bash
./venv/bin/python ethio-asr-android-builder/scripts/validate_onnx.py --model ethio-94m
```

### Step 7: Benchmark Inference Latency & RAM
Measures load time, median inference latency, RTF, CPU usage, and RAM for 1s, 3s, and 10s audio samples:
```bash
./venv/bin/python ethio-asr-android-builder/scripts/benchmark.py --model ethio-94m
```

### Step 8: Build Android Deployment Packages
Copies the `.onnx` model, vocabularies, preprocessor configurations, and generates `manifest.json` and cryptographic `SHA256SUMS`:
```bash
./venv/bin/python ethio-asr-android-builder/scripts/generate_manifest.py --model ethio-94m --precision all
```
