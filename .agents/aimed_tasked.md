# Task: Build an Automated Ethio-ASR Model Download, ONNX Conversion, Quantization and Validation Pipeline

You are an autonomous ML engineering agent. Your task is to build and execute a reproducible model preparation pipeline for deploying Ethiopian multilingual speech recognition models on Android.

Do not merely describe the process or generate scripts without executing them. Research the model repositories, download the model artifacts, implement the conversion pipeline, execute it, validate the outputs, and produce a final deployment report.

## 1. Project objectives

Build a Python-based model preparation toolchain that:

1. Downloads the original pretrained Ethio-ASR models from Hugging Face.
2. Inspects their architectures, configurations, processors and tokenizers.
3. Exports compatible models to ONNX.
4. Produces FP32, FP16 where supported, and INT8 variants.
5. Validates numerical equivalence and transcription consistency.
6. Benchmarks model size and inference performance.
7. Produces Android deployment artifacts and metadata.
8. Documents all limitations and failures.

Target languages:

- Amharic (amh)
- Afaan Oromo (orm)
- Tigrinya (tir)
- Sidama / Sidaama (sid)
- Wolaytta (wal)

## 2. Models to investigate

Use these Hugging Face model repositories as initial candidates:

### Primary model

`badrex/Ethio-ASR-multilingual-94M`

Expected role:
- Primary Android model.
- Five-language multilingual recognition.
- Initial INT8 deployment candidate.

### Larger comparison model

`badrex/Ethio-ASR-multilingual-300M`

Expected role:
- Accuracy comparison.
- Alternative for devices with more available memory.

### Larger accuracy reference

`badrex/Ethio-ASR-multilingual-600M`

Expected role:
- FP32/FP16 accuracy reference.
- Determine whether the larger model provides a meaningful accuracy improvement.

### Additional Amharic candidate

`b1n1yam/shook-tiny-amharic-600hr`

Expected role:
- Investigate a specialized lightweight Amharic model.
- Do not treat it as a replacement for the five-language multilingual model.

IMPORTANT:
Before downloading anything, verify that these repositories exist and that the model IDs are correct.

Do not invent model architectures, evaluation metrics, supported languages or license information.

Inspect the model cards, configuration files, repository contents and licenses.

If a repository is unavailable, record the failure and investigate whether an official replacement exists.

## 3. Repository structure

Create the following project:

```text
ethio-asr-android-builder/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── models.yaml
│   └── benchmark.yaml
│
├── scripts/
│   ├── download_models.py
│   ├── inspect_models.py
│   ├── export_onnx.py
│   ├── convert_fp16.py
│   ├── quantize_int8.py
│   ├── validate_onnx.py
│   ├── benchmark.py
│   ├── generate_manifest.py
│   └── run_pipeline.py
│
├── src/
│   └── ethio_asr_builder/
│       ├── __init__.py
│       ├── downloader.py
│       ├── model_inspector.py
│       ├── exporter.py
│       ├── quantizer.py
│       ├── validator.py
│       ├── decoder.py
│       ├── audio.py
│       ├── benchmark.py
│       └── manifest.py
│
├── tests/
│   ├── test_downloader.py
│   ├── test_exporter.py
│   ├── test_quantizer.py
│   ├── test_decoder.py
│   └── test_validation.py
│
├── models/
│   ├── original/
│   ├── onnx/
│   └── quantized/
│
├── datasets/
│   └── README.md
│
├── output/
│   ├── reports/
│   ├── manifests/
│   └── android/
│
└── logs/
```

Do not commit downloaded model weights or generated ONNX files to Git.

## 4. Environment setup

Create an isolated Python virtual environment.

Install compatible versions of:

- PyTorch
- Transformers
- Hugging Face Hub
- Torchaudio
- ONNX
- ONNX Runtime
- ONNX Runtime quantization utilities
- Optimum
- NumPy
- SoundFile
- Librosa
- JiWER
- PyYAML
- pytest
- psutil

Pin compatible versions in requirements.txt or pyproject.toml.

Do not blindly install the latest versions if they introduce incompatible APIs.

Check the installed Python version, available RAM, available storage, CPU architecture and CUDA availability before starting downloads.

If the machine has insufficient storage or memory for a model, do not crash or silently skip it. Record the limitation and continue with feasible tasks.

## 5. Model downloading

Implement download_models.py.

Requirements:

- Use huggingface_hub.
- Support resumable downloads.
- Preserve the original repository structure.
- Record repository ID and exact revision/commit SHA.
- Verify downloaded files.
- Support selecting individual models.
- Avoid downloading the same model multiple times.
- Record model licenses and model-card URLs.
- Never upload credentials or tokens into logs.

Provide commands such as:

```bash
python scripts/download_models.py --model ethio-94m
python scripts/download_models.py --model ethio-300m
python scripts/download_models.py --model all
```

Use the actual model repository IDs discovered during research.

## 6. Architecture inspection

Before conversion, inspect each model.

Determine:

- Model architecture.
- Number of parameters.
- Input tensor names and shapes.
- Required sample rate.
- Expected audio normalization.
- Whether attention masks are required.
- Output tensor names and shapes.
- Vocabulary size.
- CTC blank token ID.
- Padding token ID.
- Processor implementation.
- Tokenizer implementation.
- Language-specific configuration requirements.

Save the results in:

```text
output/reports/model-inspection.json
```

Do not assume every Ethio-ASR model has identical forward signatures.

Implement architecture-specific exporters when necessary.

## 7. ONNX export

Implement export_onnx.py.

Requirements:

- Load each model from its local downloaded directory.
- Load the original Hugging Face processor.
- Put models into evaluation mode.
- Disable gradient calculation.
- Export model logits.
- Preserve the model's actual input requirements.
- Support variable-length audio.
- Use dynamic axes where supported.
- Use an appropriate ONNX opset.
- Verify the exported graph with ONNX checker.
- Verify that ONNX Runtime can load the exported graph.

Use the actual model forward signature.

Do not blindly apply a generic export wrapper if the architecture requires additional inputs.

If a standard Optimum exporter supports the model correctly, prefer it over a custom exporter.

If it does not, implement and document a model-specific wrapper.

Save artifacts as:

```text
models/onnx/<model-id>/model.fp32.onnx
```

Also save all processor and tokenizer files required for inference.

## 8. FP16 conversion

Implement convert_fp16.py.

Use ONNX conversion utilities where supported.

Requirements:

- Preserve dynamic input dimensions.
- Validate graph integrity.
- Test whether ONNX Runtime can execute the converted model.
- Record unsupported operators.
- Do not assume FP16 will be faster on Android CPU.
- Keep FP32 as the reference artifact.

Save:

```text
models/onnx/<model-id>/model.fp16.onnx
```

If conversion is not supported or causes incorrect outputs, record the failure instead of producing an invalid artifact.

## 9. INT8 quantization

Implement quantize_int8.py.

Start with dynamic quantization using ONNX Runtime.

Example approach:

```python
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType
)

quantize_dynamic(
    model_input="model.fp32.onnx",
    model_output="model.int8.onnx",
    weight_type=QuantType.QInt8
)
```

Adapt the quantization configuration to the actual model operators.

Requirements:

- Preserve required model inputs and outputs.
- Verify ONNX graph validity.
- Verify that ONNX Runtime can execute the quantized model.
- Record quantization warnings.
- Compare quantized outputs with FP32 outputs.
- Measure actual model file size.
- Record the percentage of model weights quantized.

Investigate static quantization only if appropriate calibration data and supported operators are available.

Do not automatically apply INT8 quantization to operators that cannot safely support it.

Save:

```text
models/quantized/<model-id>/model.int8.onnx
```

## 10. Audio preprocessing

Implement a reusable audio preprocessing module.

Requirements:

- Load WAV and supported audio formats.
- Convert stereo to mono.
- Resample to the exact sample rate expected by each model.
- Apply the exact normalization expected by the original processor.
- Support variable-duration recordings.
- Handle silence and invalid audio.
- Avoid arbitrary truncation.

The original Hugging Face processor must be the source of truth for preprocessing.

Do not assume that every model uses identical normalization.

## 11. CTC decoder

Implement a CTC greedy decoder.

Requirements:

1. Receive logits from ONNX Runtime.
2. Compute the highest-probability token for each frame.
3. Collapse repeated consecutive tokens.
4. Remove the blank token.
5. Decode tokens using the original tokenizer.
6. Preserve Ethiopic Unicode text.
7. Handle padding and special tokens correctly.

Validate the decoder against the original Hugging Face processor's decoding implementation.

Do not create a custom Ethiopic transliteration or character mapping.

Where a model uses a different decoding method, implement the correct method instead of forcing CTC decoding.

## 12. Validation

Implement validate_onnx.py.

Validation must include:

### Graph validation

- ONNX checker.
- ONNX Runtime loading.
- Correct input/output names.
- Correct output dimensions.
- Variable-length audio.

### Numerical validation

Compare:

- PyTorch FP32.
- ONNX FP32.
- ONNX FP16.
- ONNX INT8.

Calculate:

- Maximum absolute difference.
- Mean absolute difference.
- Relative error where meaningful.

### Transcription validation

Compare transcriptions produced by:

- Original PyTorch model.
- FP32 ONNX.
- FP16 ONNX.
- INT8 ONNX.

Use the same audio samples and preprocessing.

Calculate Word Error Rate using JiWER.

Evaluate all five target languages separately.

Do not report model accuracy without a valid reference transcription.

## 13. Benchmarking

Implement benchmark.py.

Record for each model variant:

- Original parameter count.
- Model file size.
- Model loading time.
- Audio duration.
- Inference time.
- Real-time factor.
- Peak process memory where measurable.
- CPU utilization where measurable.
- WER.
- Language.

Real-time factor:

```text
RTF = inference_time_seconds / audio_duration_seconds
```

RTF below 1 means inference completed faster than the audio duration.

Benchmark using:

- Short audio samples.
- Medium audio samples.
- Longer audio samples.

Run multiple repetitions and report median and percentile timings.

Clearly distinguish PC benchmarks from actual Android benchmarks.

Do not claim Android performance based on desktop CPU measurements.

Save results as CSV and JSON.

## 14. Android deployment package

Generate a deployment directory for every successfully validated model.

Example:

```text
output/android/ethio-asr-94m-int8/
├── model.onnx
├── vocab.json
├── tokenizer_config.json
├── preprocessor_config.json
├── model_config.json
├── manifest.json
├── SHA256SUMS
└── README.md
```

The manifest must contain:

- Model ID.
- Original repository.
- Original revision.
- Architecture.
- Supported languages.
- Sample rate.
- Input tensor name.
- Output tensor name.
- Vocabulary size.
- CTC blank token ID where applicable.
- Quantization type.
- ONNX opset.
- Model version.
- Model license.
- SHA-256 checksum.
- Required runtime information.

Generate SHA-256 checksums for all artifacts.

Do not claim that a model is Android-ready unless ONNX Runtime Mobile compatibility has been tested.

## 15. Automated pipeline

Implement run_pipeline.py.

It should support:

```bash
python scripts/run_pipeline.py --model ethio-94m
python scripts/run_pipeline.py --model ethio-94m --precision fp32
python scripts/run_pipeline.py --model ethio-94m --precision int8
python scripts/run_pipeline.py --model all
```

The pipeline should:

1. Check environment.
2. Download selected models.
3. Inspect architecture.
4. Export ONNX.
5. Generate FP16 where supported.
6. Generate INT8.
7. Validate artifacts.
8. Run available audio tests.
9. Benchmark variants.
10. Generate deployment packages.
11. Produce a final report.

Support resuming after failures without repeating completed steps unnecessarily.

Use clear logging and meaningful exit codes.

## 16. Testing requirements

Write automated tests for:

- Model configuration inspection.
- Download integrity.
- Audio preprocessing.
- ONNX graph loading.
- Quantized graph loading.
- CTC decoding.
- Manifest generation.
- SHA-256 verification.

Use small synthetic test inputs for unit tests.

Do not require downloading all model weights to run the basic test suite.

## 17. Final deliverables

Produce:

1. Complete Python source code.
2. requirements.txt or pyproject.toml.
3. Model inspection reports.
4. Original downloaded model artifacts where downloads succeed.
5. FP32 ONNX exports.
6. FP16 exports where supported.
7. INT8 ONNX exports.
8. Validation reports.
9. Benchmark CSV and JSON files.
10. Android deployment packages.
11. SHA-256 manifests.
12. README with reproducible commands.
13. Known limitations and failed conversion attempts.

## 18. Execution instructions

Work autonomously.

Do not stop after generating the scripts.

Actually execute the pipeline for the 94M model first.

After successfully validating it, proceed to the 300M model if system resources permit.

Use the 600M model as an accuracy reference only if sufficient storage and memory are available.

If the machine cannot complete a task, preserve all generated artifacts, explain the exact failure and continue with other feasible tasks.

Do not silently substitute a different model.

Do not fabricate benchmark results.

Do not report successful conversion unless the resulting model has been loaded and executed by ONNX Runtime.

Do not report successful transcription validation unless actual audio samples have been processed and compared against the original implementation.

At completion, provide a concise engineering report containing:

- Models successfully downloaded.
- Models successfully exported.
- Models successfully quantized.
- Models successfully validated.
- Model sizes.
- WER per language.
- Conversion failures.
- System limitations.
- Exact commands to reproduce the build.
- Exact paths to Android deployment artifacts.

The primary objective is a reproducible, validated offline ASR model package for Android, not merely an ONNX file.