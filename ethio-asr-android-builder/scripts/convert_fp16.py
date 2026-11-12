#!/usr/bin/env python3
"""convert_fp16.py
Export / convert model to FP16 ONNX directly from PyTorch half-precision weights.
This avoids type-mismatch bugs introduced by naive post-export graph transformations.
Usage:
  python scripts/convert_fp16.py --model ethio-94m
"""
import argparse, sys, os, json
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from transformers import AutoModelForCTC, AutoProcessor
import onnx
import onnxruntime as ort
import numpy as np

CONFIG_PATH = Path("configs/models.yaml")

def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {m["alias"]: m["id"] for m in cfg["models"]}

class ModelWrapperFP16(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        outputs = self.model(input_values=input_values)
        return outputs.logits

def convert(alias, repo_id):
    model_dir = Path("models/original") / alias
    if not model_dir.exists():
        print(f"Error: Model dir {model_dir} not found.")
        sys.exit(1)

    onnx_dir = Path("models/onnx") / alias
    onnx_dir.mkdir(parents=True, exist_ok=True)
    fp16_path = onnx_dir / "model.fp16.onnx"

    print(f"Loading {alias} ({repo_id}) and casting to half (FP16)...")
    model = AutoModelForCTC.from_pretrained(model_dir).half()
    processor = AutoProcessor.from_pretrained(model_dir)
    model.eval()

    sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
    dummy_input = torch.randn(1, int(sr), dtype=torch.float16)

    wrapper = ModelWrapperFP16(model)
    wrapper.eval()

    print(f"Exporting FP16 ONNX to {fp16_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(fp16_path),
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "frame_length"}
            },
            opset_version=17,
            do_constant_folding=True
        )

    print("Checking ONNX model validity...")
    onnx_model = onnx.load(str(fp16_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model checker passed.")

    print("Verifying loading with ONNX Runtime...")
    sess = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input_values": dummy_input.numpy()})
    print(f"✓ ONNX Runtime verified. Output shape: {out[0].shape}")

    size_mb = fp16_path.stat().st_size / (1024 * 1024)
    print(f"✅ FP16 model successfully generated! File size: {size_mb:.2f} MB")

    meta = {
        "alias": alias,
        "repo_id": repo_id,
        "precision": "fp16",
        "onnx_path": str(fp16_path),
        "size_mb": round(size_mb, 2),
        "status": "success",
        "input_names": ["input_values"],
        "output_names": ["logits"],
        "opset_version": 17
    }
    with open(onnx_dir / "fp16_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model alias (e.g. ethio-94m)")
    args = parser.parse_args()

    cfg = load_cfg()
    if args.model not in cfg:
        print(f"Unknown model alias: {args.model}")
        sys.exit(1)
    convert(args.model, cfg[args.model])

if __name__ == "__main__":
    main()
