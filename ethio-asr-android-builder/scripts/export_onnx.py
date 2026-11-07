#!/usr/bin/env python3
"""export_onnx.py
Export a downloaded Ethio‑ASR model to ONNX (FP32).
Usage:
  python scripts/export_onnx.py --model ethio-94m
"""
import argparse, json, os, sys
from pathlib import Path
import torch
from transformers import AutoModelForCTC, AutoProcessor
import onnx
import onnxruntime as ort

CONFIG_PATH = Path("configs/models.yaml")

def load_cfg():
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {m["alias"]: m["id"] for m in cfg["models"]}

def export(alias, repo_id):
    model_dir = Path("models/original") / alias
    if not model_dir.exists():
        print(f"Model {alias} not found under {model_dir}. Did you download it?")
        sys.exit(1)
    print(f"Loading model and processor from {model_dir} ...")
    model = AutoModelForCTC.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    model.eval()
    # Create a dummy input: 1‑second of silence at the expected sample rate
    sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
    dummy_wave = torch.zeros(1, int(sr))
    inputs = processor(dummy_wave, sampling_rate=sr, return_tensors="pt")
    # Export to ONNX
    onnx_dir = Path("models/onnx") / alias
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.fp32.onnx"
    print(f"Exporting to {onnx_path} ...")
    torch.onnx.export(
        model,
        tuple(inputs.values()),
        f=str(onnx_path),
        input_names=[k for k in inputs.keys()],
        output_names=["logits"],
        dynamic_axes={k: {0: "batch", 1: "time"} for k in inputs.keys()},
        opset_version=17,
    )
    # Verify ONNX model
    onnx.checker.check_model(str(onnx_path))
    ort.InferenceSession(str(onnx_path))
    print("✅ ONNX export succeeded and model loads correctly.")
    # Record meta
    meta = {
        "repo_id": repo_id,
        "alias": alias,
        "onnx_path": str(onnx_path),
    }
    (onnx_dir / "export_meta.json").write_text(json.dumps(meta, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model alias (e.g., ethio-94m)")
    args = parser.parse_args()
    cfg = load_cfg()
    if args.model not in cfg:
        print(f"Unknown model alias {args.model}")
        sys.exit(1)
    export(args.model, cfg[args.model])

if __name__ == "__main__":
    main()
