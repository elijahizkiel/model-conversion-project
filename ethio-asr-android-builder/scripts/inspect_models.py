#!/usr/bin/env python3
"""inspect_models.py
Inspect a downloaded model and dump architecture details.
Usage:
  python scripts/inspect_models.py --model ethio-94m
"""
import argparse, json, os
from pathlib import Path
from transformers import AutoConfig, AutoProcessor
import torch
import yaml

CONFIG_PATH = Path("configs/models.yaml")

def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {m["alias"]: m["id"] for m in cfg["models"]}

def inspect(alias, repo_id):
    model_dir = Path("models/original") / alias
    if not model_dir.exists():
        print(f"Model directory {model_dir} not found. Did you download it?")
        return
    cfg = AutoConfig.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    info = {
        "model_id": repo_id,
        "alias": alias,
        "num_parameters": sum(p.numel() for p in cfg.to_dict().get("model_type", "unknown")) if hasattr(cfg, "to_dict") else "unknown",
        "model_type": cfg.model_type,
        "hidden_size": getattr(cfg, "hidden_size", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "max_length": getattr(cfg, "max_length", None),
        "sample_rate": getattr(processor.feature_extractor, "sampling_rate", None) if hasattr(processor, "feature_extractor") else None,
        "processor": processor.__class__.__name__,
    }
    out_path = Path("output/reports") / f"inspection_{alias}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Inspection written to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model alias")
    args = parser.parse_args()
    cfg = load_cfg()
    if args.model not in cfg:
        print(f"Unknown model alias {args.model}")
        return
    inspect(args.model, cfg[args.model])

if __name__ == "__main__":
    main()
