#!/usr/bin/env python3
"""inspect_models.py
Inspect downloaded models and output detailed architecture specs to output/reports/model-inspection.json.
"""
import argparse, json, os, sys
from pathlib import Path
import yaml
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCTC

CONFIG_PATH = Path("configs/models.yaml")

def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {m["alias"]: m["id"] for m in cfg["models"]}

def inspect_model(alias, repo_id):
    model_dir = Path("models/original") / alias
    if not model_dir.exists():
        print(f"Error: Directory {model_dir} does not exist.")
        sys.exit(1)

    print(f"Inspecting {alias} ({repo_id})...")
    config = AutoConfig.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForCTC.from_pretrained(model_dir)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    feature_extractor = getattr(processor, "feature_extractor", None)
    tokenizer = getattr(processor, "tokenizer", None)

    sample_rate = getattr(feature_extractor, "sampling_rate", 16000)
    do_normalize = getattr(feature_extractor, "do_normalize", True)

    pad_token_id = getattr(config, "pad_token_id", None)
    if pad_token_id is None and tokenizer:
        pad_token_id = tokenizer.pad_token_id

    bos_token_id = getattr(config, "bos_token_id", None)
    eos_token_id = getattr(config, "eos_token_id", None)
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None and tokenizer:
        vocab_size = len(tokenizer)

    inspection_data = {
        "alias": alias,
        "repo_id": repo_id,
        "model_type": config.model_type,
        "architecture": config.architectures if hasattr(config, "architectures") else [model.__class__.__name__],
        "parameters": {
            "total": total_params,
            "trainable": trainable_params
        },
        "inputs": [
            {
                "name": "input_values",
                "shape": ["batch_size", "sequence_length"],
                "dtype": "float32"
            }
        ],
        "outputs": [
            {
                "name": "logits",
                "shape": ["batch_size", "frame_length", vocab_size],
                "dtype": "float32"
            }
        ],
        "audio_requirements": {
            "sampling_rate": sample_rate,
            "channels": 1,
            "normalization": "zero_mean_unit_variance" if do_normalize else "none"
        },
        "ctc_parameters": {
            "vocab_size": vocab_size,
            "pad_token_id": pad_token_id,
            "blank_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id
        },
        "processor_class": processor.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__ if tokenizer else None
    }

    out_dir = Path("output/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "model-inspection.json"

    # Merge with existing inspection file if it exists
    existing = {}
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    existing[alias] = inspection_data
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"✓ Inspection report saved to {report_file}")
    return inspection_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model alias or all")
    args = parser.parse_args()

    cfg = load_cfg()
    if args.model == "all":
        for a, r in cfg.items():
            inspect_model(a, r)
    else:
        if args.model not in cfg:
            print(f"Unknown model alias: {args.model}")
            sys.exit(1)
        inspect_model(args.model, cfg[args.model])

if __name__ == "__main__":
    main()
