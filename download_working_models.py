#!/usr/bin/env python3
"""download_working_models
==========================
Batch downloader for all 7-B models used in the DRM memory experiments.

Behaviour
---------
* Ensures the Hugging Face cache root ``/scratch/gpfs/ck2867/models2/hub``
  exists and is exported via ``HF_HOME`` for the duration of the run.
* Treats a repo as **complete** only if at least one weight shard is found
  under ``snapshots/<commit>/``. Flat or partial caches are deleted and
  re-downloaded.
* Safe to re-run at any time – finished repos are skipped.

Prerequisites
-------------
    pip install -U "huggingface_hub>=0.23" "transformers>=4.40"

Example
-------
    export HF_HOME=/scratch/gpfs/$USER/models2/hub
    export HF_TOKEN=<your-token>
    python download_working_models.py
"""

import getpass, os, pathlib, sys
from typing import List

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("Please `pip install -U huggingface_hub transformers` first.")

# ------------------------------------------------------------------
# 🔴  EDIT THIS LIST if you need more / fewer repos
# ------------------------------------------------------------------
MODELS: List[str] = [
    "facebook/opt-125m",
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen1.5-7B-Chat",
    "openchat/openchat-3.5-0106",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "codellama/CodeLlama-7b-Instruct-hf",
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
]

# ------------------------------------------------------------------
# 1. Decide cache root  →  **always** /scratch/gpfs/ck2867/models2
#    (You can still override by exporting HF_HOME before running.)
# ------------------------------------------------------------------
cache_root = pathlib.Path("/scratch/gpfs/ck2867/models2")
hub_root = cache_root / "hub"
hub_root.mkdir(parents=True, exist_ok=True)

# make Hub libs see the same cache for this run
os.environ["HF_HOME"] = str(cache_root)

print(f"📦  Cache root: {cache_root}\n")

# ------------------------------------------------------------------
# 2. Download / resume each repo
# ------------------------------------------------------------------
for repo in MODELS:
    safe_name = repo.replace("/", "--")              # models--<repo-id>
    target = hub_root / f"models--{safe_name}"

    needs_fix = False
    if target.exists():
        # weight shards that live under snapshots/<commit>/
        snapshot_weights = list((target / "snapshots").glob("*/*.safetensors"))
        snapshot_bins     = list((target / "snapshots").glob("*/*.bin"))
        complete = bool(snapshot_weights or snapshot_bins)

        if not complete:
            needs_fix = True  # old flat layout or partial download

    if not target.exists() or needs_fix:
        if needs_fix:
            print(f"🗑️  Removing malformed {target} and re-downloading …")
            import shutil, time
            shutil.rmtree(target)
            time.sleep(0.1)
        else:
            print(f"⬇  Downloading {repo} → {target}")
    else:
        print(f"✅ {repo} already present → skipping")
        continue

    snapshot_download(
        repo_id=repo,
        cache_dir=str(hub_root),     # let HF Hub create standard snapshots layout
        resume_download=True,        # resumes partial pulls safely
        max_workers=4,               # gentle on GPFS
        token=True,                  # picks up your HF token
    )
    print(f"✔  finished {repo}\n")

print("🎉  All requested repos are now in", hub_root)
