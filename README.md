# DRM-LLM Memory Experiments

This repository contains code to reproduce DRM false-memory experiments with language models. The core idea is to assess how often models 'recall' a thematic lure word that never appeared in their study list.

## Repository Contents

| File | Purpose |
|------|---------|
| `README.md` | This documentation file |
| `DRM/download_working_models.py` | Downloader for models used in the study |
| `DRM/hallucinations_full_experiment.py` | Main DRM experiment with multiple model configurations |
| `DRM/prompt_experiment.py` | Prompt quality testing experiment |
| `DRM/SLURM_TEMPLATE.sh` | SLURM script for main experiment |
| `DRM/SLURM_PROMPT_EXPERIMENT.sh` | SLURM script for prompt experiment |
| `DRM/related_words.csv` | Input word lists ranked by semantic similarity |

## Quick Start

```bash
# Set up environment
export HF_HOME=/scratch/gpfs/$USER/models2/hub
export HF_TOKEN=<your-token>
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Download models (on login node)
python DRM/download_working_models.py

# Run experiments
sbatch DRM/SLURM_TEMPLATE.sh          # Main DRM experiment
sbatch DRM/SLURM_PROMPT_EXPERIMENT.sh # Prompt quality experiment
```

## Experiments

### 1. Main DRM Experiment (`hallucinations_full_experiment.py`)

Tests DRM false memory effects across multiple language models. This is the primary experiment that measures how often models hallucinate thematic lure words.

**Key Features:**
- Tests 9 different language models (GPT-3.5 to GPT-4 level)
- Uses 2x2 experimental design (list length × context timing)
- Measures recall accuracy, precision, and false memory rates
- Tracks lure word hallucinations specifically

**Run the main experiment:**
```bash
python DRM/hallucinations_full_experiment.py
# or via SLURM:
sbatch DRM/SLURM_TEMPLATE.sh
```

**What it does:**
1. Loads each model and presents word lists
2. Runs math filler tasks to clear working memory
3. Tests recall of presented words
4. Measures false memories (lure word hallucinations)

### 2. Prompt Quality Experiment (`prompt_experiment.py`)

Tests how different recall prompt formulations affect model performance. This experiment varies the specificity and structure of recall instructions to see how they impact accuracy and hallucinations.

**Key Features:**
- Tests 5 different prompt variants (from very specific to very open-ended)
- Measures how prompt structure affects memory performance
- Compares precision vs. creativity in recall tasks

**Run the prompt experiment:**
```bash
python DRM/prompt_experiment.py --models all --lure-words sleep,anger,needle,stream --num-sessions 20
# or via SLURM:
sbatch DRM/SLURM_PROMPT_EXPERIMENT.sh
```

**Prompt Variants Tested:**
1. **Strict**: "List only the words you remember, separated by commas, with no additional text"
2. **Concise**: "List exactly the words you remember from before, separated by commas"
3. **Neutral**: "List the words you remember from before, separated by commas"
4. **Loose**: "What words do you remember from earlier? You may provide a short description"
5. **Open-ended**: "Describe what you remember about the earlier content, including any themes or related ideas"

**Expected Results:**
- More restrictive prompts (Strict, Concise) should show higher recall accuracy and lower false memory rates
- More open-ended prompts (Loose, Open-ended) should show more creative responses but potentially more hallucinations

### Custom Experiment Runs

Run with specific models and parameters:

```bash
# Test specific models
python DRM/prompt_experiment.py --models gpt-oss-20b,glm-4.5 --lure-words anger --num-sessions 5

# Use larger word lists
python DRM/prompt_experiment.py --models all --list-size 15
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
python DRM/prompt_experiment.py --list-size 15  # Use top 15 most similar words
```

## Environment Setup

**Required Environment Variables:**
- `HF_HOME`: Root directory for Hugging Face model cache (e.g., `/scratch/gpfs/$USER/models2/hub`)
- `HF_TOKEN`: Your Hugging Face access token for gated models
- `TRANSFORMERS_OFFLINE=1`: Forces offline mode on compute nodes
- `HF_DATASETS_OFFLINE=1`: Prevents dataset downloads

**Model Management:**
1. Edit `DRM/hallucinations_full_experiment.py` or `DRM/prompt_experiment.py`
2. Add new model to the `MODELS` dictionary
3. Run `python DRM/download_working_models.py` on login node
4. Submit SLURM job

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