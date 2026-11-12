#!/usr/bin/env python3
"""run_pipeline.py
Automated end-to-end pipeline:
1. Environment check
2. Model download
3. Architecture inspection
4. FP32 ONNX export
5. FP16 export
6. INT8 quantization
7. Multi-language validation
8. Performance benchmarking
9. Android package generation
10. Final report generation
"""
import argparse, sys, os, subprocess
from pathlib import Path

def run_step(cmd_list, desc):
    print(f"\n==================================================")
    print(f"--> [STEP] {desc}")
    print(f"==================================================")
    res = subprocess.run(cmd_list)
    if res.returncode != 0:
        print(f"✖ Step failed: {desc}")
        sys.exit(res.returncode)
    print(f"✓ Step succeeded: {desc}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ethio-94m", help="model alias (default: ethio-94m)")
    parser.add_argument("--precision", default="all", choices=["int8", "fp16", "fp32", "all"])
    args = parser.parse_args()

    python_bin = sys.executable

    # 1. Download model
    run_step([python_bin, "ethio-asr-android-builder/scripts/download_models.py", "--model", args.model], f"Download {args.model}")

    # 2. Inspect architecture
    run_step([python_bin, "ethio-asr-android-builder/scripts/inspect_models.py", "--model", args.model], f"Inspect {args.model}")

    # 3. Export FP32 ONNX
    run_step([python_bin, "ethio-asr-android-builder/scripts/export_onnx.py", "--model", args.model], f"Export FP32 ONNX for {args.model}")

    # 4. FP16 export
    if args.precision in ["fp16", "all"]:
        run_step([python_bin, "ethio-asr-android-builder/scripts/convert_fp16.py", "--model", args.model], f"Export FP16 ONNX for {args.model}")

    # 5. INT8 dynamic quantization
    if args.precision in ["int8", "all"]:
        run_step([python_bin, "ethio-asr-android-builder/scripts/quantize_int8.py", "--model", args.model], f"Quantize INT8 for {args.model}")

    # 6. Validate
    run_step([python_bin, "ethio-asr-android-builder/scripts/validate_onnx.py", "--model", args.model], f"Validate ONNX for {args.model}")

    # 7. Benchmark
    run_step([python_bin, "ethio-asr-android-builder/scripts/benchmark.py", "--model", args.model], f"Benchmark {args.model}")

    # 8. Android packaging
    run_step([python_bin, "ethio-asr-android-builder/scripts/generate_manifest.py", "--model", args.model, "--precision", args.precision], f"Package Android assets for {args.model}")

    print("\n🎉 Full pipeline completed successfully!")

if __name__ == "__main__":
    main()
