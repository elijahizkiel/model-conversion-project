#!/usr/bin/env python3
"""export_onnx.py
Export a downloaded Ethio‑ASR model to ONNX (FP32).
Usage:
  python scripts/export_onnx.py --model ethio-94m
"""
import argparse, json, os, sys, shutil
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from transformers import AutoModelForCTC, AutoProcessor
import onnx
import onnxruntime as ort

CONFIG_PATH = Path("configs/models.yaml")

def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {m["alias"]: m["id"] for m in cfg["models"]}

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        outputs = self.model(input_values=input_values)
        return outputs.logits

def export(alias, repo_id):
    model_dir = Path("models/original") / alias
    if not model_dir.exists():
        print(f"Error: Model {alias} not found under {model_dir}. Download first.")
        sys.exit(1)

    print(f"Loading {alias} ({repo_id}) from {model_dir} ...")
    model = AutoModelForCTC.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    model.eval()

    sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
    # Dummy input: 1 second audio waveform
    dummy_input = torch.randn(1, int(sr), dtype=torch.float32)

    onnx_dir = Path("models/onnx") / alias
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.fp32.onnx"

    print(f"Exporting FP32 ONNX to {onnx_path} ...")
    wrapper = ModelWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(onnx_path),
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
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model checker passed.")

    print("Verifying loading with ONNX Runtime...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_out = sess.run(None, {"input_values": dummy_input.numpy()})
    print(f"✓ ONNX Runtime verified. Output logits shape: {test_out[0].shape}")

    # Copy tokenizer and processor files to onnx directory for standalone deployment
    for fn in ["vocab.json", "tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json", "config.json"]:
        src = model_dir / fn
        if src.exists():
            shutil.copy2(src, onnx_dir / fn)

    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    meta = {
        "repo_id": repo_id,
        "alias": alias,
        "precision": "fp32",
        "onnx_path": str(onnx_path),
        "size_mb": round(file_size_mb, 2),
        "opset_version": 17,
        "input_names": ["input_values"],
        "output_names": ["logits"],
    }
    with open(onnx_dir / "export_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Export completed successfully! Model size: {file_size_mb:.2f} MB")
    return onnx_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model alias (e.g. ethio-94m)")
    args = parser.parse_args()

    cfg = load_cfg()
    if args.model not in cfg:
        print(f"Unknown model alias: {args.model}")
        sys.exit(1)
    export(args.model, cfg[args.model])

if __name__ == "__main__":
    main()
