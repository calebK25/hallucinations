e-exfrom transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import random
import pandas as pd
import os
import gc

# Model configurations
MODELS = {
    # GPT-3.5 level models
    "llama2-7b": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "context_window": 4096,
        "requires_auth": True,
        "performance_tier": "gpt3.5"
    },
    "mistral-7b-v1": {
        "name": "mistralai/Mistral-7B-Instruct-v0.1",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "gpt3.5",
        "trust_remote_code": False
    },
    "zephyr-7b": {
        "name": "HuggingFaceH4/zephyr-7b-beta",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "gpt3.5",
        "trust_remote_code": False
    },
    
    # GPT-4.0 level models
    "qwen-7b": {
        "name": "Qwen/Qwen1.5-7B-Chat",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "gpt4.0",
        "trust_remote_code": True
    },
    "mistral-7b-v3": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "context_window": 32768,
        "requires_auth": False,
        "performance_tier": "gpt4.0",
        "trust_remote_code": False
    },
    "openchat-7b": {
        "name": "openchat/openchat-3.5-0106",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "gpt4.0",
        "trust_remote_code": True
    },
    
    # Reasoning models
    "deepseek-coder-7b": {
        "name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "reasoning",
        "trust_remote_code": False
    },
    "codellama-7b": {
        "name": "codellama/CodeLlama-7b-Instruct-hf",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "reasoning",
        "trust_remote_code": False
    },
    "nous-hermes-7b": {
        "name": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "reasoning",
        "trust_remote_code": False
    },
}

# Global cache for token counting
TOKEN_CACHE = {}

def load_model(model_key):
    """Load a model by its key from the MODELS dictionary"""
    if model_key not in MODELS:
        raise ValueError(f"Model {model_key} not found. Available: {list(MODELS.keys())}")
    
    model_config = MODELS[model_key]
    model_name = model_config["name"]
    
    print(f"Loading {model_key} ({model_name})...")
    cache_dir = os.environ.get('HF_HOME', os.environ.get('TRANSFORMERS_CACHE', os.path.expanduser("~/model_cache")))

    try:
        token = os.environ.get('HF_TOKEN') if model_config["requires_auth"] else None

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,  # Force offline mode
            trust_remote_code=model_config.get("trust_remote_code", False),
            token=token
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,  # Force offline mode
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=model_config.get("trust_remote_code", False),
            token=token,
            low_cpu_mem_usage=True  # Reduce memory usage
        )
        model.eval()
        
        print(f"Loaded {model_key}")
        return tokenizer, model, model_config

    except Exception as e:
        print(f"Error loading {model_key}: {e}")
        if "local_files_only" in str(e) or "couldn't find" in str(e):
            print("Model not found in cache. Please run download_models.py on login node first.")
            print(f"Cache directory: {cache_dir}")
        elif model_config["requires_auth"]:
            print("This model requires authentication. Make sure you have access and are logged in.")
        return None, None, None

def generate_response_fast(tokenizer, model, model_config, messages, max_new_tokens=512):
    """Optimized response generation with minimal overhead"""
    
    # Fast chat template application
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Ultra-fast fallback
        formatted_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) + "\nAssistant:"
    
    # Fast tokenization with minimal processing
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=model_config["context_window"] - max_new_tokens - 50,
        padding=False  # No padding needed for single sequence
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Optimized generation parameters for speed
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,  # Slightly reduced for speed
            no_repeat_ngram_size=2,   # Reduced from 3 for speed
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache
            num_beams=1      # Greedy decoding for speed
        )
    
    # Decode response exactly as model produces it
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response

def read_word_list(file_path, lure_word, list_length="short"):
    """
    Read a list of related words from the CSV file.

    It assumes the CSV is pre-sorted by similarity and selects the top N words.
    - "short" list_length will return 10 words.
    - "long" list_length will return 50 words.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Filter for the specific lure word and get the related words
        words = df[df['Lure Word'] == lure_word]['Related Word'].dropna().tolist()
        
        if not words:
            print(f"No words found for lure word '{lure_word}' in {file_path}")
            return []

        # Determine the number of words to select
        list_size = 10 if list_length == "short" else 50
        
        # Ensure we don't request more words than are available
        list_size = min(list_size, len(words))
        
        # Select the top N words and shuffle them
        selected_words = words[:list_size]
        random.shuffle(selected_words)
        return selected_words
        
    except FileNotFoundError:
        print(f"❌ ERROR: Word list file not found at '{file_path}'.")
        # Fallback to prevent crashing during experiment
        list_size = 10 if list_length == "short" else 50
        return [f"fallback_word_{i}" for i in range(1, list_size + 1)]
        
    except Exception as e:
        print(f"❌ ERROR: Could not read word list from '{file_path}': {e}")
        # Generic fallback
        list_size = 10 if list_length == "short" else 50
        return [f"fallback_word_{i}" for i in range(1, list_size + 1)]

def count_tokens_fast(tokenizer, text):
    """Ultra-fast token counting with global cache"""
    global TOKEN_CACHE
    
    # Use hash for faster lookup
    text_hash = hash(text)
    if text_hash in TOKEN_CACHE:
        return TOKEN_CACHE[text_hash]
    
    # Only encode if not cached
    count = len(tokenizer.encode(text, add_special_tokens=False))
    TOKEN_CACHE[text_hash] = count
    
    # Limit cache size to prevent memory issues
    if len(TOKEN_CACHE) > 10000:
        TOKEN_CACHE.clear()
    
    return count

def generate_math_problems_batch(num_problems):
    """Generate math problems without tokenization"""
    problems = []
    for _ in range(num_problems):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        problems.append(f"{a} + {b} = ?")
    return problems

def run_drm_experiment_2x2_ultra_optimized(model_key, num_sessions=10, word_list_file="related_words.csv", 
                                          lure_word="sleep", list_length="short", context_window_condition="early"):
    """Ultra-optimized DRM experiment with minimal overhead"""
    
    print(f"Loading {model_key} for {list_length}-{context_window_condition} condition...")
    tokenizer, model, model_config = load_model(model_key)
    if model is None:
        print(f"Failed to load model {model_key}")
        return []
    
    logs = []
    
    if context_window_condition == "early":
        num_math_problems = 5
        print(f"Early condition: {num_math_problems} math problems")
    else:
        num_math_problems = 50
        print(f"Late condition: {num_math_problems} math problems")
    
    # Pre-generate all components
    word_list = read_word_list(word_list_file, lure_word, list_length)
    word_list_str = ', '.join(word_list)
    
    # Pre-calculate token counts for fixed prompts
    system_prompt = ("You are a student doing a memory test which has 3 parts: "
                    "1) First, you will be asked to remember a list of words "
                    "2) Then, you will solve some math problems "
                    "3) Finally, you will be asked to recall the words from step 1. "
                    "Please complete each section to the best of your ability and follow instructions carefully.")
    
    present_prompt = f"Please remember the following words: {word_list_str}."
    recall_prompt = "Please recall the list of words I asked you to remember earlier. List only the words you remember, separated by commas, with no additional text or explanation."
    
    # Pre-generate math problems
    all_math_problems = generate_math_problems_batch(num_math_problems)
    
    # Batch process sessions
    batch_size = 5  # Process 5 sessions at a time for progress updates
    
    for batch_start in range(0, num_sessions, batch_size):
        batch_end = min(batch_start + batch_size, num_sessions)
        print(f"Processing sessions {batch_start+1}-{batch_end} ({((batch_end/num_sessions)*100):.0f}% complete)")
        
        for session in range(batch_start + 1, batch_end + 1):
            messages = [{"role": "system", "content": system_prompt}]
            
            # Step 1: Present words (minimal processing)
            messages.append({"role": "user", "content": present_prompt})
            present_response = generate_response_fast(tokenizer, model, model_config, messages, max_new_tokens=50)
            messages.append({"role": "assistant", "content": present_response})
            
            # Log word presentation
            logs.append({
                "session": session,
                "model": model_key,
                "model_name": model_config["name"],
                "lure_word": lure_word,
                "performance_tier": model_config["performance_tier"],
                "section": "word_list",
                "list_length": list_length,
                "context_window_condition": context_window_condition,
                "prompt": present_prompt,
                "response": present_response,
                "prompt_tokens": count_tokens_fast(tokenizer, present_prompt),
                "response_tokens": count_tokens_fast(tokenizer, present_response),
                "total_tokens": count_tokens_fast(tokenizer, present_prompt) + count_tokens_fast(tokenizer, present_response)
            })
            
            if context_window_condition == "early":
                math_prompt = "Please solve these addition problems and respond with only the numerical answers (one per line):\n" + "\n".join(all_math_problems)
                messages.append({"role": "user", "content": math_prompt})
                math_response = generate_response_fast(tokenizer, model, model_config, messages, max_new_tokens=30)
                messages.append({"role": "assistant", "content": math_response})
                
                logs.append({
                    "session": session,
                    "model": model_key,
                    "model_name": model_config["name"],
                    "lure_word": lure_word,
                    "performance_tier": model_config["performance_tier"],
                    "section": "filler_task",
                    "list_length": list_length,
                    "context_window_condition": context_window_condition,
                    "prompt": math_prompt,
                    "response": math_response,
                    "prompt_tokens": count_tokens_fast(tokenizer, math_prompt),
                    "response_tokens": count_tokens_fast(tokenizer, math_response),
                    "total_tokens": count_tokens_fast(tokenizer, math_prompt) + count_tokens_fast(tokenizer, math_response)
                })
            else:
                for i in range(0, len(all_math_problems), 10):
                    batch_problems = all_math_problems[i:i+10]
                    if batch_problems:
                        math_prompt = "Please solve these addition problems and respond with only the numerical answers (one per line):\n" + "\n".join(batch_problems)
                        messages.append({"role": "user", "content": math_prompt})
                        math_response = generate_response_fast(tokenizer, model, model_config, messages, max_new_tokens=50)
                        messages.append({"role": "assistant", "content": math_response)

                        if i == 0 or i >= len(all_math_problems) - 10:
                            logs.append({
                                "session": session,
                                "model": model_key,
                                "model_name": model_config["name"],
                                "lure_word": lure_word,
                                "performance_tier": model_config["performance_tier"],
                                "section": "filler_task",
                                "list_length": list_length,
                                "context_window_condition": context_window_condition,
                                "prompt": math_prompt,
                                "response": math_response,
                                "prompt_tokens": count_tokens_fast(tokenizer, math_prompt),
                                "response_tokens": count_tokens_fast(tokenizer, math_response),
                                "total_tokens": count_tokens_fast(tokenizer, math_prompt) + count_tokens_fast(tokenizer, math_response)
                            })
            
            # Step 3: Recall (minimal tokens)
            messages.append({"role": "user", "content": recall_prompt})
            recall_response = generate_response_fast(tokenizer, model, model_config, messages, max_new_tokens=80)
            
            logs.append({
                "session": session,
                "model": model_key,
                "model_name": model_config["name"],
                "lure_word": lure_word,
                "performance_tier": model_config["performance_tier"],
                "section": "recall",
                "list_length": list_length,
                "context_window_condition": context_window_condition,
                "prompt": recall_prompt,
                "response": recall_response,
                "prompt_tokens": count_tokens_fast(tokenizer, recall_prompt),
                "response_tokens": count_tokens_fast(tokenizer, recall_response),
                "total_tokens": count_tokens_fast(tokenizer, recall_prompt) + count_tokens_fast(tokenizer, recall_response)
            })
            
            # Minimal session summary
            logs.append({
                "session": session,
                "model": model_key,
                "model_name": model_config["name"],
                "performance_tier": model_config["performance_tier"],
                "lure_word": lure_word,
                "list_length": list_length,
                "context_window_condition": context_window_condition,
                "section": "session_summary",
                "total_math_problems": num_math_problems
            })
    
    # Cleanup
    print(f"Cleaning up memory...")
    del tokenizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Completed {num_sessions} sessions for {model_key}-{list_length}-{context_window_condition}\n")
    
    return logs



def extract_original_words(df_all):
    """Extract the original word list from the word_list prompts"""
    word_list_entries = df_all[df_all['section'] == 'word_list']
    all_original_words = set()

    for _, entry in word_list_entries.iterrows():
        prompt = entry['prompt']
        if 'Please remember the following words:' in prompt:
            words_part = prompt.split('Please remember the following words:')[1]
            words = words_part.split(', ')
            all_original_words.update([w.strip() for w in words if w.strip()])

    return all_original_words

def extract_recalled_words(recall_response):
    """Extract words from a recall response"""
    if pd.isna(recall_response) or not recall_response:
        return set()

    import re
    words = re.split(r'[,\n\s]+', str(recall_response))
    return {w.strip() for w in words if w.strip()}

def analyze_recall_performance(df_all, base_filename):
    """Analyze recall performance and create detailed performance metrics"""
    
    # Get original words presented to the model
    original_words = extract_original_words(df_all)
    lure_word = df_all['lure_word'].iloc[0] if not df_all.empty else "unknown"
    model_name = df_all['model'].iloc[0] if not df_all.empty else "unknown"
    list_length = df_all['list_length'].iloc[0] if not df_all.empty else "unknown"
    context_condition = df_all['context_window_condition'].iloc[0] if not df_all.empty else "unknown"
    
    # Analyze each recall session
    recall_entries = df_all[df_all['section'] == 'recall']
    performance_data = []
    
    for _, entry in recall_entries.iterrows():
        session = entry['session']
        recalled_words = extract_recalled_words(entry['response'])
        
        # Calculate metrics
        correctly_recalled = original_words.intersection(recalled_words)
        missed_words = original_words - recalled_words
        false_memories = recalled_words - original_words
        
        # Check for lure word hallucination
        lure_word_hallucinated = lure_word in recalled_words
        
        # Calculate accuracy metrics
        total_original = len(original_words)
        total_recalled = len(recalled_words)
        total_correct = len(correctly_recalled)
        total_false = len(false_memories)
        total_missed = len(missed_words)
        
        recall_accuracy = (total_correct / total_original * 100) if total_original > 0 else 0
        precision = (total_correct / total_recalled * 100) if total_recalled > 0 else 0
        false_memory_rate = (total_false / total_recalled * 100) if total_recalled > 0 else 0
        
        performance_entry = {
            'session': session,
            'model': model_name,
            'lure_word': lure_word,
            'list_length': list_length,
            'context_window_condition': context_condition,
            'total_original_words': total_original,
            'total_recalled_words': total_recalled,
            'correctly_recalled_count': total_correct,
            'missed_words_count': total_missed,
            'false_memories_count': total_false,
            'lure_word_hallucinated': lure_word_hallucinated,
            'recall_accuracy_percent': round(recall_accuracy, 2),
            'precision_percent': round(precision, 2),
            'false_memory_rate_percent': round(false_memory_rate, 2),
            'correctly_recalled_words': ', '.join(sorted(correctly_recalled)),
            'missed_words': ', '.join(sorted(missed_words)),
            'false_memories': ', '.join(sorted(false_memories))
        }
        performance_data.append(performance_entry)
    
    return performance_data

def run_full_2x2_drm_study_ultra_optimized(models_to_test, num_sessions=10, word_list_file="related_words.csv", 
                                           lure_words=["sleep"]):
    """Ultra-optimized 2x2 DRM experiment for cluster execution"""
    
    all_trial_results = []
    all_interaction_logs = []
    
    # Define experimental conditions (2x2 design)
    conditions = [
        {"list_length": "short", "context_window_condition": "early"},
        {"list_length": "short", "context_window_condition": "late"},
        {"list_length": "long", "context_window_condition": "early"},
        {"list_length": "long", "context_window_condition": "late"}
    ]
    
    total_experiments = len(models_to_test) * len(lure_words) * len(conditions)
    experiment_count = 0
    
    start_time = time.time()
    
    for model_key in models_to_test:
        model_start_time = time.time()
        
        for lure_word in lure_words:
            for condition in conditions:
                experiment_count += 1
                list_length = condition["list_length"]
                context_condition = condition["context_window_condition"]
                
                print(f"\n{'='*80}")
                print(f"Experiment {experiment_count}/{total_experiments}")
                print(f"Model: {model_key}")
                print(f"Lure word: '{lure_word}'")
                print(f"List length: {list_length} ({'10 words' if list_length == 'short' else '50 words'})")
                print(f"Context window: {context_condition}")
                print(f"Sessions: {num_sessions}")

                elapsed = time.time() - start_time
                if experiment_count > 1:
                    avg_time_per_exp = elapsed / (experiment_count - 1)
                    remaining_time = avg_time_per_exp * (total_experiments - experiment_count)
                    print(f"Estimated time remaining: {remaining_time/60:.1f} minutes")
                print(f"{'='*80}")
                
                condition_start = time.time()
                
                results = run_drm_experiment_2x2_ultra_optimized(
                    model_key=model_key,
                    num_sessions=num_sessions,
                    word_list_file=word_list_file,
                    lure_word=lure_word,
                    list_length=list_length,
                    context_window_condition=context_condition
                )
                
                print(f"Condition completed in {(time.time() - condition_start)/60:.1f} minutes")
                
                if results:
                    # Process results efficiently
                    df_all = pd.DataFrame(results)
                    
                    # Only log recall interactions to reduce file size
                    for _, row in df_all.iterrows():
                        if row['section'] == 'recall':
                            interaction_log = {
                                'model': model_key,
                                'model_name': row['model_name'],
                                'performance_tier': row['performance_tier'],
                                'lure_word': lure_word,
                                'list_length': list_length,
                                'context_window_condition': context_condition,
                                'session_number': row['session'],
                                'phase': row['section'],
                                'prompt': str(row['prompt'])[:100] + "...",  # Truncate prompts
                                'response': str(row['response']),
                                'response_preview': str(row['response'])[:100] + "..."
                            }
                            all_interaction_logs.append(interaction_log)
                    
                    # Analyze performance
                    performance_data = analyze_recall_performance(df_all, f"temp_{model_key}_{lure_word}_{list_length}_{context_condition}")
                    
                    # Add trial results
                    for entry in performance_data:
                        trial_result = {
                            'result_type': 'individual_trial',
                            'model': model_key,
                            'model_name': df_all['model_name'].iloc[0],
                            'performance_tier': df_all['performance_tier'].iloc[0],
                            'lure_word': lure_word,
                            'list_length': list_length,
                            'context_window_condition': context_condition,
                            'session_number': entry['session'],
                            'recall_accuracy_percent': entry['recall_accuracy_percent'],
                            'precision_percent': entry['precision_percent'],
                            'false_memory_rate_percent': entry['false_memory_rate_percent'],
                            'lure_word_hallucinated': entry['lure_word_hallucinated'],
                            'total_original_words': entry['total_original_words'],
                            'total_recalled_words': entry['total_recalled_words'],
                            'correctly_recalled_count': entry['correctly_recalled_count'],
                            'false_memories_count': entry['false_memories_count']
                        }
                        all_trial_results.append(trial_result)
                    
                    # Add summary
                    if performance_data:
                        avg_accuracy = sum(e['recall_accuracy_percent'] for e in performance_data) / len(performance_data)
                        avg_precision = sum(e['precision_percent'] for e in performance_data) / len(performance_data)
                        avg_false_memory = sum(e['false_memory_rate_percent'] for e in performance_data) / len(performance_data)
                        lure_recall_count = sum(1 for e in performance_data if e['lure_word_hallucinated'])
                        
                        summary_result = {
                            'result_type': 'summary',
                            'model': model_key,
                            'model_name': df_all['model_name'].iloc[0],
                            'performance_tier': df_all['performance_tier'].iloc[0],
                            'lure_word': lure_word,
                            'list_length': list_length,
                            'context_window_condition': context_condition,
                            'session_number': 'ALL',
                            'num_sessions': num_sessions,
                            'recall_accuracy_percent': round(avg_accuracy, 2),
                            'precision_percent': round(avg_precision, 2),
                            'false_memory_rate_percent': round(avg_false_memory, 2),
                            'lure_word_hallucination_rate_percent': round((lure_recall_count / num_sessions) * 100, 2),
                            'total_sessions_completed': len(performance_data)
                        }
                        all_trial_results.append(summary_result)
                        
                    print(f"Processed {len(performance_data)} sessions")

        model_elapsed = time.time() - model_start_time
        print(f"\nModel {model_key} completed in {model_elapsed/60:.1f} minutes")
    
    total_elapsed = time.time() - start_time
    print(f"\nTotal experiment time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")

    if all_trial_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        results_file = f"drm_2x2_ultra_results_{timestamp}.csv"
        results_df = pd.DataFrame(all_trial_results)
        results_df = results_df.sort_values(['model', 'lure_word', 'list_length', 'context_window_condition', 'result_type', 'session_number'])
        results_df.to_csv(results_file, index=False)

        logs_file = f"drm_2x2_ultra_logs_{timestamp}.csv"
        logs_df = pd.DataFrame(all_interaction_logs)
        logs_df.to_csv(logs_file, index=False)

        print(f"\nResults saved:")
        print(f"{results_file} - Results")
        print(f"{logs_file} - Recall logs only")

        print(f"\n{'='*100}")
        print("2x2 Experimental Summary")
        print(f"{'='*100}")
        print(f"{'Model':<15} {'Lure':<8} {'List':<6} {'Context':<8} {'Accuracy':<9} {'False Mem':<10} {'Lure Rate':<10}")
        print("-" * 100)
        
        summary_rows = [r for r in all_trial_results if r['result_type'] == 'summary']
        for entry in summary_rows:
            print(f"{entry['model']:<15} {entry['lure_word']:<8} "
                  f"{entry['list_length']:<6} {entry['context_window_condition']:<8} "
                  f"{entry['recall_accuracy_percent']:>7.1f}% "
                  f"{entry['false_memory_rate_percent']:>8.1f}% "
                  f"{entry['lure_word_hallucination_rate_percent']:>8.1f}%")
        
        print("Optimizations applied:")
        print("   - Single model load per condition")
        print("   - Batch math problem processing (5-10 at once)")
        print("   - Reduced math problems (5 early, 50 late)")
        print("   - Global token caching with hash lookup")
        print("   - Minimal logging (recall only)")
        print("   - Pre-generated word lists")
        print("   - Optimized generation parameters")
        print("   - Batch session processing")

        return results_file, logs_file
    else:
        print("No results to save")
        return None

if __name__ == "__main__":
    print("DRM Memory Experiment")

    test_models = list(MODELS.keys())

    result_files = run_full_2x2_drm_study_ultra_optimized(
        models_to_test=test_models,
        num_sessions=10,
        lure_words=["anger", "sleep", "doctor"]
    )

    if result_files:
        results_file, logs_file = result_files
        print(f"\nExperiment completed!")
        print(f"\nResults saved to:")
        print(f"   - {results_file}")
        print(f"   - {logs_file}")
    else:
        print("\nExperiment failed - no results generated") 