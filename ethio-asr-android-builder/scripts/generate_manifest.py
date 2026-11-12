#!/usr/bin/env python3
"""generate_manifest.py
Build the Android deployment directory and metadata manifest:
output/android/<model-alias>-<precision>/
├── model.onnx
├── vocab.json
├── tokenizer_config.json
├── preprocessor_config.json
├── config.json
├── manifest.json
├── SHA256SUMS
└── README.md
"""
import argparse, sys, os, json, hashlib, shutil
from pathlib import Path
import yaml

CONFIG_PATH = Path("configs/models.yaml")

def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {m["alias"]: m["id"] for m in cfg["models"]}

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def package_android(alias, repo_id, precision="int8"):
    model_orig_dir = Path("models/original") / alias
    if precision == "int8":
        model_src = Path("models/quantized") / alias / "model.int8.onnx"
    elif precision == "fp16":
        model_src = Path("models/onnx") / alias / "model.fp16.onnx"
    else:
        model_src = Path("models/onnx") / alias / "model.fp32.onnx"

    if not model_src.exists():
        print(f"Error: Model file {model_src} does not exist.")
        sys.exit(1)

    target_dir = Path("output/android") / f"{alias}-{precision}"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Packaging Android bundle for {alias} ({precision}) into {target_dir} ...")

    # Copy ONNX model
    target_model = target_dir / "model.onnx"
    shutil.copy2(model_src, target_model)

    # Copy tokenizers and configs
    for fn in ["vocab.json", "tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json", "config.json"]:
        src = model_orig_dir / fn
        if src.exists():
            shutil.copy2(src, target_dir / fn)

    # Load download meta and inspection data
    download_meta = {}
    dl_meta_path = model_orig_dir / "download_meta.json"
    if dl_meta_path.exists():
        with open(dl_meta_path, "r", encoding="utf-8") as f:
            download_meta = json.load(f)

    inspection = {}
    insp_path = Path("output/reports/model-inspection.json")
    if insp_path.exists():
        with open(insp_path, "r", encoding="utf-8") as f:
            all_insp = json.load(f)
            inspection = all_insp.get(alias, {})

    # Generate Manifest
    manifest = {
        "model_id": alias,
        "original_repo": repo_id,
        "revision": download_meta.get("revision", "unknown"),
        "architecture": inspection.get("architecture", ["HubertForCTC"]),
        "precision": precision,
        "supported_languages": [
            {"code": "amh", "name": "Amharic"},
            {"code": "orm", "name": "Afaan Oromo"},
            {"code": "tir", "name": "Tigrinya"},
            {"code": "sid", "name": "Sidama"},
            {"code": "wal", "name": "Wolaytta"}
        ],
        "audio_requirements": {
            "sampling_rate": inspection.get("audio_requirements", {}).get("sampling_rate", 16000),
            "channels": 1,
            "normalization": "zero_mean_unit_variance"
        },
        "inputs": [
            {
                "name": "input_values",
                "dtype": "float32" if precision != "fp16" else "float16",
                "shape": ["batch_size", "sequence_length"]
            }
        ],
        "outputs": [
            {
                "name": "logits",
                "dtype": "float32" if precision != "fp16" else "float16",
                "shape": ["batch_size", "frame_length", 416]
            }
        ],
        "ctc": {
            "blank_token_id": 408,
            "pad_token_id": 408,
            "vocab_size": 416
        },
        "onnx_opset": 17,
        "model_license": download_meta.get("license", "Apache-2.0 / Open"),
        "android_runtime": {
            "min_sdk": 26,
            "recommended_runtime": "onnxruntime-android",
            "recommended_package": "com.microsoft.onnxruntime:onnxruntime-android:1.17.0"
        }
    }

    manifest_path = target_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Compute SHA256SUMS for all package files
    checksums = []
    for file_path in sorted(target_dir.iterdir()):
        if file_path.is_file() and file_path.name != "SHA256SUMS":
            sha = sha256_file(file_path)
            checksums.append(f"{sha}  {file_path.name}")

    sha_file = target_dir / "SHA256SUMS"
    with open(sha_file, "w", encoding="utf-8") as f:
        f.write("\n".join(checksums) + "\n")

    # Generate README for Android developers
    readme_content = f"""# {alias} ({precision.upper()}) Android Deployment Package

This directory contains deployment assets for offline speech recognition across 5 Ethiopian languages:
- Amharic (amh)
- Afaan Oromo (orm)
- Tigrinya (tir)
- Sidama (sid)
- Wolaytta (wal)

## Files
- `model.onnx`: {precision.upper()} ONNX model ({target_model.stat().st_size / (1024*1024):.2f} MB)
- `vocab.json`: CTC vocabulary mapping token IDs to characters
- `manifest.json`: Specification metadata (sample rate, inputs, CTC blank ID)
- `SHA256SUMS`: Cryptographic verification hashes

## Quick Android Integration (Kotlin)
1. Place the entire folder or files into `app/src/main/assets/asr/`.
2. Add dependency in `build.gradle.kts`:
   ```kotlin
   implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
   ```
3. Load the session:
   ```kotlin
   val env = OrtEnvironment.getEnvironment()
   val modelBytes = context.assets.open("asr/model.onnx").readBytes()
   val session = env.createSession(modelBytes)
   ```
4. Feed normalized 16kHz mono audio float buffer into tensor `input_values`.
5. Greedy CTC decode the output `logits` using `vocab.json` (collapse repeated tokens, filter ID 408).
"""
    with open(target_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"✅ Android package successfully created at: {target_dir}")
    return target_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model alias")
    parser.add_argument("--precision", default="int8", choices=["int8", "fp16", "fp32", "all"])
    args = parser.parse_args()

    cfg = load_cfg()
    if args.model not in cfg:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    if args.precision == "all":
        for p in ["int8", "fp16", "fp32"]:
            package_android(args.model, cfg[args.model], precision=p)
    else:
        package_android(args.model, cfg[args.model], precision=args.precision)

if __name__ == "__main__":
    main()
