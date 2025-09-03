#!/usr/bin/env python3
"""
Download models for DRM experiments.

Prerequisites: pip install -U huggingface_hub transformers
Usage: export HF_HOME=/scratch/gpfs/$USER/models2/hub && python download_working_models.py
"""

import getpass, os, pathlib, sys
from typing import List

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("Install dependencies: pip install -U huggingface_hub transformers")

# Models to download
MODELS: List[str] = [
    # Baselines and study models
    "facebook/opt-125m",
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen1.5-7B-Chat",
    "openchat/openchat-3.5-0106",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "codellama/CodeLlama-7b-Instruct-hf",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    # New models
    "openai/gpt-oss-20b",
    "zai-org/GLM-4.5",
]

# Set cache directory
cache_root = pathlib.Path(os.environ.get("HF_HOME", f"/scratch/gpfs/{getpass.getuser()}/models2"))
hub_root = cache_root / "hub"
hub_root.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(cache_root)

print(f"Cache root: {cache_root}\n")

# Download each model
for repo in MODELS:
    safe_name = repo.replace("/", "--")
    target = hub_root / f"models--{safe_name}"

    needs_fix = False
    if target.exists():
        snapshot_weights = list((target / "snapshots").glob("*/*.safetensors"))
        snapshot_bins = list((target / "snapshots").glob("*/*.bin"))
        complete = bool(snapshot_weights or snapshot_bins)
        if not complete:
            needs_fix = True

    if not target.exists() or needs_fix:
        if needs_fix:
            print(f"Removing malformed {target} and re-downloading...")
            import shutil, time
            shutil.rmtree(target)
            time.sleep(0.1)
        else:
            print(f"Downloading {repo}...")

        snapshot_download(
            repo_id=repo,
            cache_dir=str(hub_root),
            resume_download=True,
            max_workers=4,
            token=True,
        )
        print(f"Finished {repo}\n")
    else:
        print(f"{repo} already present, skipping")

print(f"All models downloaded to {hub_root}")
