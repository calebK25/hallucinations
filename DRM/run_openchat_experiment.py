#!/usr/bin/env python3
"""run_openchat_experiment
===========================
Convenience wrapper that executes the 2 × 2 DRM memory experiment for a
*single* 7-B model – OpenChat 3.5 – instead of the full model roster.

Why a separate script?
----------------------
During cluster debugging it is often useful to test one model in
isolation (smaller wall-time, faster iteration).  This script achieves
that while re-using all the heavy-lifting functions in
``hallucination_test_models_full_experiment``.

Key behaviour
-------------
1. Imports the main experiment helper and **patches** the global ``MODELS``
   dict at runtime to force the OpenChat entry to use the safetensors
   repo `openchat/openchat-3.5-0106`.
2. Runs ``run_full_2x2_drm_study_ultra_optimized`` for that single model.
3. Results are written to a timestamped directory in the working folder.

Environment variables required (same as full experiment)
-------------------------------------------------------
HF_HOME                Path to the shared Hugging Face cache (offline nodes)
HF_TOKEN               Token for gated models (not needed for OpenChat)
TRANSFORMERS_OFFLINE   1 on compute nodes with no internet
HF_DATASETS_OFFLINE    1 on compute nodes with no internet

Example SLURM line
------------------
    python run_openchat_experiment.py
"""

from datetime import datetime
from pathlib import Path

from hallucination_test_models_full_experiment import (
    run_full_2x2_drm_study_ultra_optimized,
    MODELS as _MODELS,
)

# ------------------------------------------------------------------
# Ensure the OpenChat entry points to the safetensors repo (3.5-0106).
# This overrides any stale copy of the module that might still reference
# the deprecated 3.2 weights.
# ------------------------------------------------------------------
_MODELS["openchat-7b"]["name"] = "openchat/openchat-3.5-0106"

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Configure the single model to test
    # ------------------------------------------------------------------
    MODELS_TO_TEST = ["openchat-7b"]

    # Optional: customise number of sessions if you want quicker runs
    NUM_SESSIONS = 10  # keep same as full study

    # ------------------------------------------------------------------
    # Create a timestamped output directory
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"openchat_experiment_{timestamp}")
    out_dir.mkdir(exist_ok=True)

    # Run the experiment (function handles CSV/analysis saving)
    run_full_2x2_drm_study_ultra_optimized(
        models_to_test=MODELS_TO_TEST,
        num_sessions=NUM_SESSIONS,
        word_list_file="related_words.csv",
        lure_words=["sleep", "anger"],  # adjust if needed
    )

    print("🎉 OpenChat-only experiment finished. Results are in", out_dir) 