#!/usr/bin/env python3
"""validate_onnx.py
Validate PyTorch vs ONNX FP32 vs FP16 vs INT8:
- Numerical differences (Max Abs Diff, Mean Abs Diff)
- Transcription consistency across the 5 Ethiopian target languages.
"""
import argparse, sys, os, json
from pathlib import Path
import yaml
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoModelForCTC, AutoProcessor
import jiwer

# Import audio and decoder helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ethio_asr_builder.audio import load_and_preprocess_audio
from ethio_asr_builder.decoder import CTCDecoder

CONFIG_PATH = Path("configs/models.yaml")

def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {m["alias"]: m["id"] for m in cfg["models"]}

def validate(alias, repo_id):
    model_dir = Path("models/original") / alias
    fp32_path = Path("models/onnx") / alias / "model.fp32.onnx"
    fp16_path = Path("models/onnx") / alias / "model.fp16.onnx"
    int8_path = Path("models/quantized") / alias / "model.int8.onnx"

    print(f"Loading processor and PyTorch model for {alias} ...")
    processor = AutoProcessor.from_pretrained(model_dir)
    pytorch_model = AutoModelForCTC.from_pretrained(model_dir)
    pytorch_model.eval()

    decoder = CTCDecoder(processor.tokenizer, blank_id=getattr(processor.tokenizer, "pad_token_id", 408))

    # Initialize ONNX runtime sessions
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_fp16 = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"]) if fp16_path.exists() else None
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"]) if int8_path.exists() else None

    # Test languages
    languages = ["amharic", "afaan_oro", "tigrinya", "sidama", "wolaytta"]
    results = {
        "alias": alias,
        "repo_id": repo_id,
        "languages": {},
        "numerical_metrics": {
            "fp32_vs_pytorch": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0},
            "fp16_vs_pytorch": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0},
            "int8_vs_pytorch": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0}
        }
    }

    max_diff_fp32 = []
    mean_diff_fp32 = []
    max_diff_fp16 = []
    mean_diff_fp16 = []
    max_diff_int8 = []
    mean_diff_int8 = []

    print("\n--- Running Multi-Language Validation ---")
    for lang in languages:
        wav_path = Path("datasets") / f"{lang}.wav"
        if not wav_path.exists():
            print(f"Warning: {wav_path} not found. Skipping.")
            continue

        audio_tensor = load_and_preprocess_audio(str(wav_path), target_sr=16000, do_normalize=True)

        # PyTorch reference
        with torch.no_grad():
            pt_out = pytorch_model(input_values=torch.from_numpy(audio_tensor))
            pt_logits = pt_out.logits.numpy()
        pt_text = decoder.decode_logits(pt_logits)[0]

        # ONNX FP32
        onnx_fp32_out = sess_fp32.run(None, {"input_values": audio_tensor})[0]
        fp32_text = decoder.decode_logits(onnx_fp32_out)[0]

        d_fp32 = np.abs(onnx_fp32_out - pt_logits)
        max_diff_fp32.append(float(np.max(d_fp32)))
        mean_diff_fp32.append(float(np.mean(d_fp32)))

        # ONNX FP16
        fp16_text = ""
        if sess_fp16:
            audio_fp16 = audio_tensor.astype(np.float16)
            onnx_fp16_out = sess_fp16.run(None, {"input_values": audio_fp16})[0].astype(np.float32)
            fp16_text = decoder.decode_logits(onnx_fp16_out)[0]
            d_fp16 = np.abs(onnx_fp16_out - pt_logits)
            max_diff_fp16.append(float(np.max(d_fp16)))
            mean_diff_fp16.append(float(np.mean(d_fp16)))

        # ONNX INT8
        int8_text = ""
        if sess_int8:
            onnx_int8_out = sess_int8.run(None, {"input_values": audio_tensor})[0]
            int8_text = decoder.decode_logits(onnx_int8_out)[0]
            d_int8 = np.abs(onnx_int8_out - pt_logits)
            max_diff_int8.append(float(np.max(d_int8)))
            mean_diff_int8.append(float(np.mean(d_int8)))

        # Calculate consistency WER against PyTorch reference
        wer_fp32 = jiwer.wer(pt_text, fp32_text) if pt_text or fp32_text else 0.0
        wer_fp16 = jiwer.wer(pt_text, fp16_text) if pt_text or fp16_text else 0.0
        wer_int8 = jiwer.wer(pt_text, int8_text) if pt_text or int8_text else 0.0

        results["languages"][lang] = {
            "pytorch_transcription": pt_text,
            "fp32_transcription": fp32_text,
            "fp16_transcription": fp16_text,
            "int8_transcription": int8_text,
            "wer_fp32_vs_pt": wer_fp32,
            "wer_fp16_vs_pt": wer_fp16,
            "wer_int8_vs_pt": wer_int8
        }
        print(f"[{lang}] PyTorch: '{pt_text}' | FP32: '{fp32_text}' | FP16: '{fp16_text}' | INT8: '{int8_text}'")

    if max_diff_fp32:
        results["numerical_metrics"]["fp32_vs_pytorch"]["max_abs_diff"] = float(np.max(max_diff_fp32))
        results["numerical_metrics"]["fp32_vs_pytorch"]["mean_abs_diff"] = float(np.mean(mean_diff_fp32))
    if max_diff_fp16:
        results["numerical_metrics"]["fp16_vs_pytorch"]["max_abs_diff"] = float(np.max(max_diff_fp16))
        results["numerical_metrics"]["fp16_vs_pytorch"]["mean_abs_diff"] = float(np.mean(mean_diff_fp16))
    if max_diff_int8:
        results["numerical_metrics"]["int8_vs_pytorch"]["max_abs_diff"] = float(np.max(max_diff_int8))
        results["numerical_metrics"]["int8_vs_pytorch"]["mean_abs_diff"] = float(np.mean(mean_diff_int8))

    out_file = Path("output/reports") / f"validation_{alias}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Validation report saved to {out_file}")
    print(f"FP32 vs PyTorch Max Abs Diff: {results['numerical_metrics']['fp32_vs_pytorch']['max_abs_diff']:.6f}")
    print(f"FP16 vs PyTorch Max Abs Diff: {results['numerical_metrics']['fp16_vs_pytorch']['max_abs_diff']:.6f}")
    print(f"INT8 vs PyTorch Max Abs Diff: {results['numerical_metrics']['int8_vs_pytorch']['max_abs_diff']:.6f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model alias (e.g. ethio-94m)")
    args = parser.parse_args()

    cfg = load_cfg()
    if args.model not in cfg:
        print(f"Unknown model alias: {args.model}")
        sys.exit(1)
    validate(args.model, cfg[args.model])

if __name__ == "__main__":
    main()
