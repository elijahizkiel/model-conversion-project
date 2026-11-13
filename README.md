# Ethio-ASR Android Builder & ONNX Toolchain

An automated, reproducible ML engineering toolchain for converting, quantizing, validating, and deploying multilingual Ethiopian speech recognition models (`badrex/Ethio-ASR-multilingual-94M`) for on-device Android execution.

---

## 🌍 Supported Languages
The converted models support end-to-end Automatic Speech Recognition (ASR) across five Ethiopian languages:
- **Amharic (`amh`)**
- **Afaan Oromo (`orm`)**
- **Tigrinya (`tir`)**
- **Sidama (`sid`)**
- **Wolaytta (`wal`)**

---

## 📊 Model Variants & Benchmark Results

Benchmarks measured on CPU (Intel Core i3-10110U, 4 threads @ 4.10 GHz, 12 GB RAM) using ONNX Runtime with multi-threaded intra-op execution:

| Variant | Precision | Model Size | Size Reduction | RTF (1s Audio) | RTF (3s Audio) | RTF (10s Audio) | Peak RAM | Max Abs Diff vs PyTorch |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32-bit Float | 361.45 MB | Baseline | 0.650 | 0.811 | 0.493 | ~776 MB | 0.000016 |
| **FP16** | 16-bit Float | 180.84 MB | -50.0% | 1.155 | 0.759 | 0.570 | ~850 MB | 0.007307 |
| **INT8 (Primary)** | 8-bit Dynamic | **117.00 MB** | **-67.6%** | **0.331** | **0.354** | **0.290** | **~726 MB** | 0.282805 |

> **Real-Time Factor (RTF)** = `Inference Time / Audio Duration`.  
> An RTF of `0.290` means 10 seconds of spoken audio is processed in **~2.9 seconds** on CPU.

---

## 🚀 Repository Structure

```text
model-conversion-project/
├── configs/
│   └── models.yaml              # Registered model repository IDs and aliases
├── ethio-asr-android-builder/
│   ├── scripts/
│   │   ├── download_models.py   # Resumable download from Hugging Face Hub
│   │   ├── inspect_models.py    # Architecture & tensor shape inspector
│   │   ├── export_onnx.py       # FP32 ONNX export with dynamic axes
│   │   ├── convert_fp16.py      # Direct PyTorch half-precision FP16 ONNX export
│   │   ├── quantize_int8.py     # Dynamic INT8 MatMul/Gemm quantization
│   │   ├── validate_onnx.py     # Numerical & multi-language transcription validation
│   │   ├── benchmark.py         # Latency, RTF, CPU, and RAM profiler
│   │   ├── generate_manifest.py # Standalone Android package & SHA256 generator
│   │   └── run_pipeline.py      # End-to-end automated orchestrator
│   └── src/
│       └── ethio_asr_builder/
│           ├── audio.py         # Audio preprocessing (16kHz mono, zero-mean unit-variance)
│           └── decoder.py       # Greedy CTC decoder (consecutive token collapse, blank removal)
├── output/
│   ├── android/
│   │   ├── EthioASRRecognizer.kt # Ready-to-use Android Kotlin wrapper
│   │   ├── ethio-94m-int8/      # Android deployment bundle (INT8, 117 MB)
│   │   ├── ethio-94m-fp16/      # Android deployment bundle (FP16, 181 MB)
│   │   └── ethio-94m-fp32/      # Android deployment bundle (FP32, 362 MB)
│   └── reports/
│       ├── model-inspection.json
│       ├── validation_ethio-94m.json
│       └── benchmark_ethio-94m.csv
└── requirements.txt
```

---

## 🛠️ Environment Setup & Installation

The toolchain requires **Python 3.11** (due to binary compatibility with ONNX Runtime and PyTorch CPU):

```bash
# Clone the repository
git clone https://github.com/elijahizkiel/model-conversion-project.git
cd model-conversion-project

# Create and activate Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚡ Running the Pipeline

Run the complete pipeline end-to-end with a single command:

```bash
./venv/bin/python ethio-asr-android-builder/scripts/run_pipeline.py --model ethio-94m --precision all
```

Or run individual steps sequentially:

```bash
# 1. Download model from Hugging Face
./venv/bin/python ethio-asr-android-builder/scripts/download_models.py --model ethio-94m

# 2. Inspect architecture
./venv/bin/python ethio-asr-android-builder/scripts/inspect_models.py --model ethio-94m

# 3. Export FP32 ONNX
./venv/bin/python ethio-asr-android-builder/scripts/export_onnx.py --model ethio-94m

# 4. Export FP16 ONNX
./venv/bin/python ethio-asr-android-builder/scripts/convert_fp16.py --model ethio-94m

# 5. Quantize to INT8
./venv/bin/python ethio-asr-android-builder/scripts/quantize_int8.py --model ethio-94m

# 6. Validate numerical accuracy and transcription
./venv/bin/python ethio-asr-android-builder/scripts/validate_onnx.py --model ethio-94m

# 7. Benchmark performance
./venv/bin/python ethio-asr-android-builder/scripts/benchmark.py --model ethio-94m

# 8. Generate Android asset packages
./venv/bin/python ethio-asr-android-builder/scripts/generate_manifest.py --model ethio-94m --precision all
```

---

## 📱 Android Integration (Kotlin)

### 1. Add ONNX Runtime Mobile dependency
In your app's `build.gradle.kts`:
```kotlin
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
}
```

### 2. Copy Model Assets
Copy the files from `output/android/ethio-94m-int8/` to your project's `app/src/main/assets/asr/`:
- `model.onnx`
- `vocab.json`
- `manifest.json`

### 3. Use the Recognizer Wrapper
Copy [output/android/EthioASRRecognizer.kt](output/android/EthioASRRecognizer.kt) into your Android codebase:

```kotlin
import com.ethioasr.recognizer.EthioASRRecognizer

// Initialize recognizer (loads ONNX model and vocab from assets/asr/)
val recognizer = EthioASRRecognizer(context, "asr/model.onnx")

// Transcribe 16kHz mono audio (FloatArray)
val audioSamples: FloatArray = getAudioBuffer() // raw PCM samples
val transcription: String = recognizer.transcribe(audioSamples)

println("Detected transcription: $transcription")

// Clean up resources when done
recognizer.close()
```

---

## 📦 GitHub Releases & Downloading Pre-built Models

The `.onnx` models and their Android asset archives are hosted on GitHub Releases:

- **INT8 Bundle**: `ethio-asr-94m-int8-android.tar.gz` (~104 MB compressed)
- **FP16 Bundle**: `ethio-asr-94m-fp16-android.tar.gz` (~168 MB compressed)
- **FP32 Bundle**: `ethio-asr-94m-fp32-android.tar.gz` (~336 MB compressed)

Download and extract them directly into your Android assets folder:
```bash
tar -xzvf ethio-asr-94m-int8-android.tar.gz -C app/src/main/assets/asr/
```

---

## 📄 License
The underlying models are developed by the Ethio-ASR project team. The conversion scripts, tools, and Android integration code in this repository are distributed under the Apache-2.0 License.
