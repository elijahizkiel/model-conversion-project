#!/usr/bin/env python3
"""download_models.py
Download selected Ethio‑ASR models from Hugging Face.
Usage:
  python scripts/download_models.py --model ethio-94m|ethio-300m|ethio-600m|amharic-tiny|all
"""
import argparse, os, json, sys
from pathlib import Path
from huggingface_hub import snapshot_download, repo_info
import yaml

CONFIG_PATH = Path("configs/models.yaml")

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {m["alias"]: m["id"] for m in cfg["models"]}

def download(alias, repo_id):
    dest = Path("models/original") / alias
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} into {dest} ...")
    try:
        # snapshot_download returns the root directory of the repo
        local_path = snapshot_download(repo_id=repo_id, local_dir=dest, resume_download=True, allow_patterns=["*"], max_workers=4)
        # Record metadata
        info = repo_info(repo_id)
        meta = {
            "repo_id": repo_id,
            "local_path": str(local_path),
            "revision": info.sha,
            "license": info.license,
            "model_card": f"https://huggingface.co/{repo_id}",
        }
        meta_path = dest / "download_meta.json"
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
        print("✓ Download complete")
    except Exception as e:
        print(f"✖ Failed to download {repo_id}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model alias or 'all'")
    args = parser.parse_args()
    cfg = load_config()
    if args.model == "all":
        for alias, repo in cfg.items():
            download(alias, repo)
    else:
        if args.model not in cfg:
            print(f"Unknown model alias: {args.model}")
            sys.exit(1)
        download(args.model, cfg[args.model])

if __name__ == "__main__":
    main()
