# DRM-LLM Memory Experiments (Ultra-Optimised)

This repository contains all code required to reproduce the
Deese–Roediger–McDermott (DRM) false-memory experiments described in our
project report.  The core idea is to assess how often different 7-billion
parameter language models 'recall' a thematic **lure word** that never
appeared in their study list.

---
## Quick-start
```bash
# 1. Clone the repo (if you haven't already)
# 2. Activate a Python 3.10+ environment with torch & transformers ≥ 4.40
conda env create -f environment.yml   # example; adjust to your cluster
conda activate hallucination

# 3. Download the models (login node with internet)
export HF_HOME=/scratch/gpfs/$USER/models2/hub
export HF_TOKEN=<your-token>           # only if you need gated models
python download_working_models.py

# 4. Submit a SLURM job
sbatch SLURM_TEMPLATE.sh
```

---
## Repository tour
| File | Purpose |
|------|---------|
| `download_working_models.py` | Idempotent downloader/repair tool for every model used in the study.  Skips repos that already contain weight shards in a `snapshots/<commit>/` directory. |
| `hallucination_test_models_full_experiment.py` | Main library that implements the ultra-optimised DRM experiment and analysis pipeline. Contains the **`MODELS`** dictionary – edit this to add/remove LLMs. |
| `run_openchat_experiment.py` | Convenience wrapper to execute the 2 × 2 experiment for **only** the OpenChat 3.5 model. Helpful for quick debugging. |
| `SLURM_TEMPLATE.sh` | Boiler-plate submission script; customise for your cluster. |
| `related_words.csv` | Input word lists ranked by semantic similarity. |

---
## Environment variables
Variable | Description
---------|------------
`HF_HOME` | Root of the Hugging Face cache (point this at shared scratch on clusters).
`HF_TOKEN` | Personal access token – required for gated models (e.g. Llama-2).
`TRANSFORMERS_OFFLINE` | Set to `1` on compute nodes without outbound internet.
`HF_DATASETS_OFFLINE` | Same as above for the *datasets* library.

---
## Adding a new model
1. Open **`hallucination_test_models_full_experiment.py`**.
2. Append a new entry to the `MODELS` dict:
   ```python
   "my-model-key": {
       "name": "author/repo",         # HF repo ID
       "context_window": 4096,
       "requires_auth": False,        # or True if gated
       "performance_tier": "gpt4.0", # for reporting only
       "trust_remote_code": True       # if repo uses custom code
   },
   ```
3. Run `download_working_models.py` (login node) – the new repo will be pulled.
4. Re-submit your SLURM job. No other code changes required.

---
## Troubleshooting
| Symptom | Likely cause & fix |
|---------|-------------------|
| `Error loading … couldn't connect to 'https://huggingface.co'` | The compute node is offline **and** the model weights aren't present under `$HF_HOME`. Run the downloader on a login node. |
| `requires users to upgrade torch to at least v2.6` | You are trying to load a `.bin` checkpoint on torch ≤ 2.5. Switch to a safetensors variant (as we did for OpenChat) or upgrade torch. |
| Model tries to download despite cache | Make sure your repo lives in `$HF_HOME/hub/models--<repo>/snapshots/<commit>/`. Delete flat/partial directories and rerun the downloader. |

---
## License
All Python code in this repository is released under the MIT License.  Individual models are subject to their own licences and terms of use – please consult the corresponding pages on Hugging Face. 