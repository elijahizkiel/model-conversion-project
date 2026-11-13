# Ethio-ASR Documentation: Index & Overview

Welcome to the **Ethio-ASR Model Conversion & Android Deployment Toolchain** documentation.

This project delivers an automated, reproducible machine learning pipeline designed to bridge pretrained multilingual Ethiopian speech recognition models from Hugging Face into optimized, on-device ONNX runtimes for Android.

---

## 🧭 How to Read This Documentation

This documentation suite is structured into dedicated modules tailored to different engineering needs:

```text
docs/
├── index.md                 # (This document) Project overview & documentation navigation
├── architecture.md          # Model topology, tensor contracts, ONNX conversion & quantization
├── getting-started.md       # Environment configuration, toolchain setup, and step-by-step pipeline execution
└── android-integration.md   # Android (Kotlin & C++) embedding, audio preprocessing, CTC decoding, and performance tuning
```

### Choose Your Reading Path:
- **If you are setting up or reproducing the model conversion on Linux/macOS:**  
  Start with [getting-started.md](getting-started.md) to configure Python 3.11, install pinned requirements, and run the automated orchestrator.
- **If you are integrating the converted models into an Android application (Kotlin / NDK):**  
  Jump straight to [android-integration.md](android-integration.md) for code snippets, memory considerations, audio contracts, and asset placement.
- **If you need in-depth ML engineering details (HubertForCTC graphs, quantization tradeoffs, dynamic axes):**  
  Read [architecture.md](architecture.md) for exact tensor inputs, mathematical normalization, CTC blank token handling, and benchmark evaluations.

---

## 🌟 What This Project Solves

Deploying state-of-the-art Automatic Speech Recognition (ASR) models on mobile devices in Africa poses unique challenges:
1. **Multilingual Complexity**: Ethiopia has over 80 languages. Most off-the-shelf ASR solutions either ignore non-Latin scripts (Ge'ez / Ethiopic Unicode) or only support single languages.
2. **Resource Constraints**: High-end ASR models (e.g. Wav2Vec2-BERT, Whisper) require gigabytes of RAM and powerful GPUs, making them impractical for standard mid-range Android devices with 3–6 GB of RAM.
3. **Network Independence**: Rural and low-connectivity environments require completely offline, self-contained inference without cloud round-trips.

This toolchain addresses these issues by:
- Targeting `badrex/Ethio-ASR-multilingual-94M` (AfriHuBERT base) covering **Amharic (`amh`)**, **Afaan Oromo (`orm`)**, **Tigrinya (`tir`)**, **Sidama (`sid`)**, and **Wolaytta (`wal`)**.
- Exporting dynamic-length computation graphs to **ONNX opset 17**.
- Employing **dynamic INT8 quantization** to shrink the model from **361.45 MB down to 117.00 MB** (a 67.6% reduction) while preserving full acoustic transcription fidelity.
- Demonstrating a CPU Real-Time Factor (RTF) of **0.290** (processing 10s of audio in 2.9s on a low-power laptop CPU, easily runnable on modern mobile ARM cores).

---

## 📋 System Requirements

| Component | Minimum Specification | Recommended |
| :--- | :--- | :--- |
| **Operating System** | Linux (Ubuntu 20.04+, Arch, Debian) / macOS | Linux x86_64 or aarch64 |
| **Python Version** | **Python 3.11** *(Critical: avoid 3.14 due to missing PyTorch/ONNX binary wheels)* | Python 3.11.18 via `mise` or `pyenv` |
| **RAM** | 8 GB | 12 GB+ |
| **Disk Space** | 5 GB free disk space | 15 GB+ (for downloading all comparison models) |
| **Target Device** | Android 8.0+ (API Level 26+) | Android 10.0+ with 4+ CPU cores |

---

## 🔗 Quick Links & Assets

- **Source Code Repository**: [github.com/elijahizkiel/model-conversion-project](https://github.com/elijahizkiel/model-conversion-project)
- **GitHub Release (Pre-built Binaries)**: [v1.0.0 Release Assets](https://github.com/elijahizkiel/model-conversion-project/releases/tag/v1.0.0)
- **Hugging Face Model Card**: [badrex/Ethio-ASR-multilingual-94M](https://huggingface.co/badrex/Ethio-ASR-multilingual-94M)
