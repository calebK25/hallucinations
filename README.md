# DRM-LLM Memory Experiments

This repository contains code to reproduce DRM false-memory experiments with language models. The core idea is to assess how often models 'recall' a thematic lure word that never appeared in their study list.

## Quick-start

```bash
conda env create -f environment.yml
conda activate hallucination

export HF_HOME=/scratch/gpfs/$USER/models2/hub
export HF_TOKEN=<your-token>
python download_working_models.py

sbatch SLURM_TEMPLATE.sh
```

## Repository tour

| File | Purpose |
|------|---------|
| `download_working_models.py` | Downloader for models used in the study. |
| `hallucination_test_models_full_experiment.py` | Main experiment library with MODELS dictionary. |
| `hallucination_test_models.py` | Alternative experiment implementation. |
| `prompt_quality_sweep.py` | Prompt quality testing across semantic distances. |
| `SLURM_TEMPLATE.sh` | SLURM submission script template. |
| `SLURM_PROMPT_SWEEP.sh` | SLURM script for prompt sweep experiments. |
| `related_words.csv` | Input word lists ranked by semantic similarity. |

## How to Use the Code

### Basic DRM Experiment

Run a standard DRM experiment with all models:

```bash
python hallucination_test_models_full_experiment.py
```

This will:
1. Test each model on lure words from `related_words.csv`
2. Run 10 sessions per model-lure combination
3. Generate CSV files with results and interaction logs

### Prompt Quality Sweep

Test how different recall prompt qualities affect performance:

```bash
python prompt_quality_sweep.py --models all --lure-words sleep,anger,needle,stream --num-sessions 20
```

This sweeps through 5 prompt variants ranging from very specific to very vague, measuring accuracy and hallucination rates.

### Custom Experiment

Run with specific models and parameters:

```bash
python prompt_quality_sweep.py --models gpt-oss-20b,glm-4.5 --lure-words anger --num-sessions 5 --list-size 15
```

## Understanding related_words.csv

The `related_words.csv` file contains word lists organized by lure words. Each row represents a related word for a specific lure.

### File Structure

```csv
Lure Word,Related Word,Similarity,Num Characters,Num Syllables,Frequency
sleep,slept,0.622322917,5,1,4.18
sleep,nap,0.618222535,3,1,3.82
sleep,doze,0.582604706,4,1,2.64
```

### Columns

- **Lure Word**: The thematic word that should NOT appear in study lists (e.g., "sleep")
- **Related Word**: Words semantically related to the lure (e.g., "slept", "nap", "doze")
- **Similarity**: Semantic similarity score (higher = more related)
- **Num Characters**: Character count of the related word
- **Num Syllables**: Syllable count
- **Frequency**: Word frequency in language

### How It Works

1. **Study Phase**: Model sees 10 related words (e.g., "slept", "nap", "doze", etc.)
2. **Filler Task**: Model solves math problems to clear working memory
3. **Recall Phase**: Model tries to recall the words from step 1
4. **Analysis**: Check if model "remembers" the lure word ("sleep") that was never presented

### Adding New Lure Words

To add a new lure word category:

1. Find semantically related words using word embedding tools
2. Add rows to CSV with your new lure word
3. Run experiments with the new lure word

Example addition:

```csv
dream,wake,0.45,4,1,4.2
dream,nightmare,0.52,9,2,3.8
dream,lucid,0.38,5,2,2.1
```

### Customizing Word Lists

You can modify the experiment to use different list sizes:

```bash
python prompt_quality_sweep.py --list-size 15  # Use top 15 most similar words
```

## Environment variables

- `HF_HOME`: Root of the Hugging Face cache.
- `HF_TOKEN`: Access token for gated models.
- `TRANSFORMERS_OFFLINE`: Set to 1 on compute nodes.
- `HF_DATASETS_OFFLINE`: Same as above.

## Adding a new model

1. Edit `hallucination_test_models_full_experiment.py`.
2. Add to MODELS dict.
3. Run downloader.
4. Submit job.

## Understanding Output Files

### Results CSV
Contains trial data and summaries:
- `individual_trial`: Per-session metrics
- `summary`: Averaged metrics across sessions

### Logs CSV
Contains all prompts and responses for detailed analysis.

### Key Metrics

- **Recall Accuracy**: Percentage of presented words correctly recalled
- **Precision**: Accuracy of recalled words (correct / total recalled)
- **False Memory Rate**: Percentage of recalled words that weren't presented
- **Lure Hallucination**: Whether the lure word was "recalled"

## Troubleshooting

- Offline error: Run downloader on login node.
- Torch version issue: Upgrade or use safetensors.
- Cache issue: Ensure proper directory structure.

## License

MIT License. Model licenses vary. 