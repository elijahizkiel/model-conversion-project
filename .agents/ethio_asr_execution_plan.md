## Execution Plan for Ethio‑ASR Android Builder

**Hardware**: Intel Core i3‑10110U (4 threads), 12 GB RAM, CPU‑only.

**Models**: `badrex/Ethio-ASR-multilingual-94M`, `badrex/Ethio-ASR-multilingual-300M`, `badrex/Ethio-ASR-multilingual-600M`, `b1n1yam/shook-tiny-amharic-600hr`.

**Goal**: Produce **FP32**, **FP16**, and **INT8** ONNX models for **all five languages**, generate synthetic test audio, validate, benchmark, and create an Android‑ready asset bundle. Kotlin will be used for the Android app; C++ can be added later for acceleration.

### Steps
1. **Environment setup** – venv, install `requirements.txt`.
2. **Project scaffolding** – create directory tree, scripts, src package.
3. **Synthetic audio generation** – 5 × 3 s wav files (one per language).
4. **Model download** – `scripts/download_models.py`.
5. **Inspection** – `scripts/inspect_models.py` → `output/reports/model_inspection.json`.
6. **ONNX export (FP32)** – `scripts/export_onnx.py`.
7. **FP16 conversion** – `scripts/convert_fp16.py`.
8. **INT8 quantization** – `scripts/quantize_int8.py`.
9. **Validation** – `scripts/validate_onnx.py` (graph check + inference comparison).
10. **Benchmark** – `scripts/benchmark.py` (loading time, inference time, RTF, memory).
11. **Manifest generation** – `scripts/generate_manifest.py`.
12. **Android packaging** – copy assets to `output/android/...`.
13. **Run pipeline** – `scripts/run_pipeline.py --model ethio-94m`.
14. **Final report** – `output/reports/final_report.md`.

All scripts are idempotent and support a `--model` argument (`ethio-94m`, `ethio-300m`, `ethio-600m`, `amharic-tiny`).

The pipeline will first process the **94 M** model (fits comfortably) and then optionally the 300 M model if resources permit.
