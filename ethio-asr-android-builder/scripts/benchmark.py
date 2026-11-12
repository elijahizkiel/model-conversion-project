#!/usr/bin/env python3
"""benchmark.py
Benchmark inference speed, real-time factor (RTF), memory, and CPU usage
across model precisions (FP32, FP16, INT8).
Outputs CSV and JSON reports.
"""
import argparse, sys, os, json, time
from pathlib import Path
import yaml
import numpy as np
import onnxruntime as ort
import psutil
import csv

CONFIG_PATH = Path("configs/models.yaml")

def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {m["alias"]: m["id"] for m in cfg["models"]}

def run_benchmark(alias, repo_id):
    fp32_path = Path("models/onnx") / alias / "model.fp32.onnx"
    fp16_path = Path("models/onnx") / alias / "model.fp16.onnx"
    int8_path = Path("models/quantized") / alias / "model.int8.onnx"

    variants = []
    if fp32_path.exists():
        variants.append(("FP32", fp32_path, np.float32))
    if fp16_path.exists():
        variants.append(("FP16", fp16_path, np.float16))
    if int8_path.exists():
        variants.append(("INT8", int8_path, np.float32))

    durations = [1.0, 3.0, 10.0]  # short, medium, long in seconds
    sr = 16000
    repeats = 5

    benchmark_rows = []
    summary_data = {"alias": alias, "repo_id": repo_id, "benchmarks": []}

    print(f"Starting Benchmark for {alias} ...")

    for prec_name, model_path, dtype in variants:
        print(f"\nBenchmarking {prec_name} ({model_path}) ...")
        file_size_mb = model_path.stat().st_size / (1024 * 1024)

        # Measure load time
        t_load_start = time.perf_counter()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        load_time_sec = time.perf_counter() - t_load_start

        for dur in durations:
            num_samples = int(dur * sr)
            dummy_input = np.random.randn(1, num_samples).astype(dtype)

            # Warmup
            sess.run(None, {"input_values": dummy_input})

            timings = []
            process = psutil.Process()
            cpu_usages = []

            for _ in range(repeats):
                cpu_before = process.cpu_percent(interval=None)
                t_start = time.perf_counter()
                sess.run(None, {"input_values": dummy_input})
                t_end = time.perf_counter()
                timings.append(t_end - t_start)
                cpu_usages.append(process.cpu_percent(interval=None))

            median_time = float(np.median(timings))
            rtf = median_time / dur
            peak_memory_mb = process.memory_info().rss / (1024 * 1024)
            avg_cpu = float(np.mean(cpu_usages)) if cpu_usages else 0.0

            row = {
                "model": alias,
                "precision": prec_name,
                "file_size_mb": round(file_size_mb, 2),
                "load_time_sec": round(load_time_sec, 3),
                "audio_duration_sec": dur,
                "median_infer_sec": round(median_time, 4),
                "rtf": round(rtf, 4),
                "peak_ram_mb": round(peak_memory_mb, 2),
                "avg_cpu_percent": round(avg_cpu, 1)
            }
            benchmark_rows.append(row)
            summary_data["benchmarks"].append(row)
            print(f"  [{prec_name}] Duration: {dur}s | Infer: {median_time:.4f}s | RTF: {rtf:.3f} | RAM: {peak_memory_mb:.1f} MB")

    # Save to output/reports
    out_dir = Path("output/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / f"benchmark_{alias}.csv"
    json_file = out_dir / f"benchmark_{alias}.json"

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_rows)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n✅ Benchmark reports saved:")
    print(f"   CSV:  {csv_file}")
    print(f"   JSON: {json_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model alias (e.g. ethio-94m)")
    args = parser.parse_args()

    cfg = load_cfg()
    if args.model not in cfg:
        print(f"Unknown model alias: {args.model}")
        sys.exit(1)
    run_benchmark(args.model, cfg[args.model])

if __name__ == "__main__":
    main()
