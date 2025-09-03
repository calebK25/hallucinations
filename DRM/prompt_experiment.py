#!/usr/bin/env python3
"""Run prompt quality experiment for DRM recall testing"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Model configurations
MODELS: Dict[str, Dict] = {
    # GPT-3.5-ish (7B)
    "llama2-7b": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "context_window": 4096,
        "requires_auth": True,
        "performance_tier": "gpt3.5",
        "trust_remote_code": False,
    },
    "mistral-7b-v1": {
        "name": "mistralai/Mistral-7B-Instruct-v0.1",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "gpt3.5",
        "trust_remote_code": False,
    },
    "zephyr-7b": {
        "name": "HuggingFaceH4/zephyr-7b-beta",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "gpt3.5",
        "trust_remote_code": False,
    },
    # GPT-4.0-ish (7B)
    "qwen-7b": {
        "name": "Qwen/Qwen1.5-7B-Chat",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "gpt4.0",
        "trust_remote_code": True,
    },
    "mistral-7b-v3": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "context_window": 32768,
        "requires_auth": False,
        "performance_tier": "gpt4.0",
        "trust_remote_code": False,
    },
    "openchat-7b": {
        "name": "openchat/openchat-3.5-0106",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "gpt4.0",
        "trust_remote_code": True,
    },
    # Reasoning/code (7B)
    "models--meta-llama--Llama-2-7B-Chat-hf": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "context_window": 4096,
        "requires_auth": True,
        "performance_tier": "gpt3.5",
        "trust_remote_code": False,
    },
    "nous-hermes-7b": {
        "name": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "reasoning",
        "trust_remote_code": False,
    },
    # New additions
    "gpt-oss-20b": {
        "name": "openai/gpt-oss-20b",
        "context_window": 8192,
        "requires_auth": False,
        "performance_tier": "open-20b",
        "trust_remote_code": True,
    },
    "glm-4.5": {
        "name": "zai-org/GLM-4.5",
        "context_window": 8192,
        "requires_auth": False,
        "performance_tier": "glm-4.5",
        "trust_remote_code": True,
    },
}


# Model loading and generation functions
def load_model(model_key: str):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODELS.keys())}")

    cfg = MODELS[model_key]
    model_id = cfg["name"]

    print(f"Loading {model_key} ({model_id})...")
    cache_dir = os.environ.get("HF_HOME", os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/model_cache")))

    token = os.environ.get("HF_TOKEN") if cfg.get("requires_auth", False) else None

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        token=token,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        token=token,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    model.eval()

    print("Model loaded")
    return tokenizer, model, cfg


def generate_response(tokenizer, model, model_cfg, messages: List[Dict], max_new_tokens: int = 128) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max(256, min(model_cfg["context_window"] - max_new_tokens - 50, 30000)),
        padding=False,
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            no_repeat_ngram_size=2,
            early_stopping=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


# Data processing and metrics functions
def read_word_list(file_path: str, lure_word: str, size: int = 10) -> List[str]:
    try:
        df = pd.read_csv(file_path)
        words = df[df["Lure Word"] == lure_word]["Related Word"].dropna().tolist()
        if not words:
            return [f"fallback_word_{i}" for i in range(1, size + 1)]
        words = words[: min(size, len(words))]
        random.shuffle(words)
        return words
    except Exception as e:
        print(f"Could not read word list from '{file_path}': {e}")
        return [f"fallback_word_{i}" for i in range(1, size + 1)]


def clean_word(word: str) -> str:
    import re
    if pd.isna(word) or word is None:
        return ""
    cleaned = re.sub(r"[^\w]", "", str(word).lower().strip())
    mapping = {
        "irritation": "irritated",
        "frustration": "frustrated",
        "madness": "mad",
        "anger": "angry",
        "furious": "fury",
        "enraged": "rage",
        "hatred": "hate",
        "lividity": "livid",
    }
    return mapping.get(cleaned, cleaned)


def extract_recalled_words(text: str) -> set:
    if pd.isna(text) or not text:
        return set()
    import re
    parts = re.split(r"[,\n\s]+", str(text))
    return {w for w in (clean_word(p) for p in parts) if w}


@dataclass
class Performance:
    session: int
    total_original: int
    total_recalled: int
    total_correct: int
    total_false: int
    total_missed: int
    accuracy: float
    precision: float
    false_memory_rate: float
    lure_word_hallucinated: bool


def analyze_variant_performance(original_words: set, lure_word: str, recall_texts: List[str]) -> List[Performance]:
    perfs: List[Performance] = []
    lure = clean_word(lure_word)
    total_original = len(original_words)
    for idx, text in enumerate(recall_texts, start=1):
        recalled = extract_recalled_words(text)
        correctly = original_words.intersection(recalled)
        false = recalled - original_words
        missed = original_words - recalled
        total_recalled = len(recalled)
        total_correct = len(correctly)
        total_false = len(false)
        total_missed = len(missed)
        accuracy = (total_correct / total_original * 100.0) if total_original else 0.0
        precision = (total_correct / total_recalled * 100.0) if total_recalled else 0.0
        false_rate = (total_false / total_recalled * 100.0) if total_recalled else 0.0
        perfs.append(
            Performance(
                session=idx,
                total_original=total_original,
                total_recalled=total_recalled,
                total_correct=total_correct,
                total_false=total_false,
                total_missed=total_missed,
                accuracy=round(accuracy, 2),
                precision=round(precision, 2),
                false_memory_rate=round(false_rate, 2),
                lure_word_hallucinated=(lure in recalled),
            )
        )
    return perfs


def classify_result(accuracy: float, false_rate: float) -> str:
    if accuracy >= 80.0 and false_rate <= 10.0:
        return "good"
    if accuracy >= 50.0 and false_rate <= 25.0:
        return "mid"
    return "bad"


# Prompt variants for testing semantic proximity
PROMPT_VARIANTS = [
    {
        "id": "good_strict",
        "label": "Strict - exact words only",
        "text": (
            "Please recall the list of words I asked you to remember earlier. "
            "List only the words you remember, separated by commas, with no additional text or explanation."
        ),
    },
    {
        "id": "good_plain",
        "label": "Concise - clear format",
        "text": "List exactly the words you remember from before, separated by commas. Do not add extra words.",
    },
    {
        "id": "neutral",
        "label": "Neutral - basic instruction",
        "text": "List the words you remember from before, separated by commas.",
    },
    {
        "id": "loose",
        "label": "Loose - allows description",
        "text": "What words do you remember from earlier? You may provide a short description.",
    },
    {
        "id": "associative",
        "label": "Open-ended - themes and ideas",
        "text": "Describe what you remember about the earlier content, including any themes or related ideas.",
    },
]


# Main experiment function
def run_prompt_sweep(
    model_keys: List[str],
    lure_words: List[str],
    word_list_file: str,
    num_sessions: int,
    list_size: int = 10,
) -> Tuple[str, str]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_logs: List[Dict] = []
    all_rows: List[Dict] = []

    for model_key in model_keys:
        tokenizer, model, cfg = load_model(model_key)
        try:
            for lure in lure_words:
                words = read_word_list(word_list_file, lure, size=list_size)
                original_words = {clean_word(w) for w in words if w}

                system_prompt = (
                    "You are a student doing a memory test which has 2 parts: "
                    "1) You will be asked to remember a list of words. "
                    "2) Later, you will be asked to recall only those words. Follow instructions carefully."
                )
                present_prompt = f"Please remember the following words: {', '.join(words)}."

                for variant in PROMPT_VARIANTS:
                    recall_prompt = variant["text"]

                    recall_texts: List[str] = []
                    for session in range(1, num_sessions + 1):
                        messages = [{"role": "system", "content": system_prompt}]
                        # Present
                        messages.append({"role": "user", "content": present_prompt})
                        present_response = generate_response(tokenizer, model, cfg, messages, max_new_tokens=48)
                        # Maintain user/assistant alternation for chat templates
                        messages.append({"role": "assistant", "content": present_response})
                        # Recall with variant
                        messages.append({"role": "user", "content": recall_prompt})
                        recall = generate_response(tokenizer, model, cfg, messages, max_new_tokens=80)
                        recall_texts.append(recall)

                        # Log
                        all_logs.append(
                            {
                                "model": model_key,
                                "model_name": cfg["name"],
                                "performance_tier": cfg["performance_tier"],
                                "lure_word": lure,
                                "session": session,
                                "variant_id": variant["id"],
                                "variant_label": variant["label"],

                                "present_prompt": present_prompt,
                                "recall_prompt": recall_prompt,
                                "recall_response": recall,
                            }
                        )

                    # Analyze after sessions for this variant
                    perfs = analyze_variant_performance(original_words, lure, recall_texts)
                    if perfs:
                        avg_acc = sum(p.accuracy for p in perfs) / len(perfs)
                        avg_prec = sum(p.precision for p in perfs) / len(perfs)
                        avg_false = sum(p.false_memory_rate for p in perfs) / len(perfs)
                        lure_count = sum(1 for p in perfs if p.lure_word_hallucinated)
                        label = classify_result(avg_acc, avg_false)

                        all_rows.append(
                            {
                                "model": model_key,
                                "model_name": cfg["name"],
                                "performance_tier": cfg["performance_tier"],
                                "lure_word": lure,
                                "variant_id": variant["id"],
                                "variant_label": variant["label"],

                                "num_sessions": num_sessions,
                                "avg_recall_accuracy_percent": round(avg_acc, 2),
                                "avg_precision_percent": round(avg_prec, 2),
                                "avg_false_memory_rate_percent": round(avg_false, 2),
                                "lure_word_hallucinated_count": lure_count,
                                "lure_word_hallucination_rate_percent": round((lure_count / num_sessions) * 100.0, 2),
                                "result_class": label,
                            }
                        )
        finally:
            # free memory between models
            del tokenizer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results_file = f"prompt_sweep_results_{timestamp}.csv"
    logs_file = f"prompt_sweep_logs_{timestamp}.csv"

    if all_rows:
        pd.DataFrame(all_rows).to_csv(results_file, index=False)
    if all_logs:
        pd.DataFrame(all_logs).to_csv(logs_file, index=False)

    print(f"\nResults saved:")
    print(f"   - {results_file} (aggregated)")
    print(f"   - {logs_file} (detailed)")
    return results_file, logs_file


def parse_args():
    ap = argparse.ArgumentParser(description="Prompt quality sweep for DRM recall")
    ap.add_argument(
        "--models",
        type=str,
        default="all",
        help=f"Comma-separated model keys from: {list(MODELS.keys())} or 'all'",
    )
    ap.add_argument(
        "--lure-words",
        type=str,
        default="anger",
        help="Comma-separated lure words present in related_words.csv",
    )
    ap.add_argument(
        "--word-list-file",
        type=str,
        default="related_words.csv",
        help="CSV file with columns: 'Lure Word', 'Related Word'",
    )
    ap.add_argument("--num-sessions", type=int, default=5, help="Sessions per (model × lure × prompt variant)")
    ap.add_argument("--list-size", type=int, default=10, help="Study list size per session")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.models.strip().lower() == "all":
        model_keys = list(MODELS.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    lure_words = [w.strip() for w in args.lure_words.split(",") if w.strip()]
    run_prompt_sweep(
        model_keys=model_keys,
        lure_words=lure_words,
        word_list_file=args.word_list_file,
        num_sessions=args.num_sessions,
        list_size=args.list_size,
    )


