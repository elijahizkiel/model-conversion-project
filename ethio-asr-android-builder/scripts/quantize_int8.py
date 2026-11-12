#!/usr/bin/env python3
"""quantize_int8.py
Dynamic INT8 quantization using ONNX Runtime.
Usage:
  python scripts/quantize_int8.py --model ethio-94m
"""
import argparse, sys, os, json
from pathlib import Path
import yaml
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np

CONFIG_PATH = Path("configs/models.yaml")

def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {m["alias"]: m["id"] for m in cfg["models"]}

def quantize(alias, repo_id):
    onnx_dir = Path("models/onnx") / alias
    fp32_path = onnx_dir / "model.fp32.onnx"
    quant_dir = Path("models/quantized") / alias
    quant_dir.mkdir(parents=True, exist_ok=True)
    int8_path = quant_dir / "model.int8.onnx"

    if not fp32_path.exists():
        print(f"Error: FP32 model not found at {fp32_path}. Export FP32 first.")
        sys.exit(1)

    print(f"Quantizing {fp32_path} to INT8 dynamic: {int8_path} ...")
    try:
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            op_types_to_quantize=["MatMul", "Gemm"]
        )
        print("✓ INT8 dynamic quantization completed.")

        print("Checking INT8 ONNX graph validity...")
        onnx_model = onnx.load(str(int8_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model checker passed.")

        print("Verifying loading and inference with ONNX Runtime...")
        sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        dummy_input = np.zeros((1, 16000), dtype=np.float32)
        out = sess.run(None, {"input_values": dummy_input})
        print(f"✓ ONNX Runtime verified. Output shape: {out[0].shape}")

        fp32_size = fp32_path.stat().st_size / (1024 * 1024)
        int8_size = int8_path.stat().st_size / (1024 * 1024)
        compression = (1.0 - (int8_size / fp32_size)) * 100.0

        print(f"✅ INT8 quantization succeeded!")
        print(f"   Original FP32 size: {fp32_size:.2f} MB")
        print(f"   Quantized INT8 size: {int8_size:.2f} MB ({compression:.1f}% reduction)")

        meta = {
            "alias": alias,
            "repo_id": repo_id,
            "precision": "int8",
            "onnx_path": str(int8_path),
            "size_mb": round(int8_size, 2),
            "fp32_size_mb": round(fp32_size, 2),
            "compression_percent": round(compression, 1),
            "quantization_type": "dynamic",
            "weight_type": "QInt8",
            "status": "success"
        }
        with open(quant_dir / "quant_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    except Exception as e:
        print(f"✖ INT8 quantization failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model alias (e.g. ethio-94m)")
    args = parser.parse_args()

    cfg = load_cfg()
    if args.model not in cfg:
        print(f"Unknown model alias: {args.model}")
        sys.exit(1)
    quantize(args.model, cfg[args.model])

if __name__ == "__main__":
    main()
