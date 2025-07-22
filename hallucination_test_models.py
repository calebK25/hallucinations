# =============================================================================
# Multi-Model Hallucination Testing Setup
# =============================================================================

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import random
import pandas as pd
import os

# =============================================================================
# Model Configurations
# =============================================================================
MODELS = {

    # === GPT-3.5 LEVEL MODELS (7B parameters, 4096 tokens) ===
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
        "performance_tier": "gpt3.5"
    },
    "zephyr-7b": {
        "name": "HuggingFaceH4/zephyr-7b-beta",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "gpt3.5"
    },
    
    # === GPT-4.0 LEVEL MODELS (7B parameters) ===
    "qwen-7b": {
        "name": "Qwen/Qwen1.5-7B-Chat",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "gpt4.0"
    },
    "mistral-7b-v3": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "context_window": 32768,
        "requires_auth": False,
        "performance_tier": "gpt4.0"
    },
    "openchat-7b": {
        "name": "openchat/openchat_v3.2_super",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "gpt4.0"
    },
    
    # === REASONING MODELS (7B parameters) ===
    "deepseek-coder-7b": {
        "name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "reasoning"
    },
    "codellama-7b": {
        "name": "codellama/CodeLlama-7b-Instruct-hf",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "reasoning"
    },
    "nous-hermes-7b": {
        "name": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "context_window": 4096,
        "requires_auth": False,
        "performance_tier": "reasoning"
    },

}

# =============================================================================
# Model Loading Function
# =============================================================================
def load_model(model_key):
    """Load a model by its key from the MODELS dictionary"""
    if model_key not in MODELS:
        raise ValueError(f"Model {model_key} not found. Available: {list(MODELS.keys())}")
    
    model_config = MODELS[model_key]
    model_name = model_config["name"]
    
    print(f"Loading {model_key} ({model_name})...")
    print(f"Context window: {model_config['context_window']} tokens")
    print(f"Performance tier: {model_config['performance_tier']}")
    
    # Set up cache directory for offline mode
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', os.path.expanduser("~/model_cache"))
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,  # Force offline mode
            trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,  # Force offline mode
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=False
        )
        model.eval()
        
        print(f"✅ {model_key} loaded successfully from cache!")
        return tokenizer, model, model_config
        
    except Exception as e:
        print(f"❌ Error loading {model_key}: {e}")
        if "local_files_only" in str(e) or "couldn't find" in str(e):
            print("💡 Model not found in cache. Please run download_models.py on login node first.")
            print(f"💡 Cache directory: {cache_dir}")
        elif model_config["requires_auth"]:
            print("💡 This model requires authentication. Make sure you have access and are logged in.")
        return None, None, None

# =============================================================================
# Unified Generation Function
# =============================================================================
def generate_response(tokenizer, model, model_config, messages, max_new_tokens=512):
    """Generate response using any of the loaded models with improved generation parameters"""
    
    # Try to use chat template if available
    try:
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            raise AttributeError("No chat template")
    except:
        # Fallback formatting
        formatted_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                formatted_prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                formatted_prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted_prompt += f"Assistant: {msg['content']}\n"
        formatted_prompt += "Assistant:"
    
    # Tokenize with appropriate context window
    max_length = min(model_config["context_window"] - max_new_tokens - 100, 30000)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=max_length)
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Enhanced generation parameters to prevent loops and improve quality
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,  # Prevent repetition loops
            no_repeat_ngram_size=3,  # Avoid repeating 3-grams
            early_stopping=True,     # Stop when EOS is generated
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    # Additional safety check - truncate if response is repetitive
    words = response.split()
    if len(words) > 50:
        # Check for excessive repetition in last 20 words
        last_words = words[-20:]
        if len(set(last_words)) < 5:  # Too much repetition
            # Find a good cutoff point
            for i in range(len(words) - 20, max(0, len(words) - 100), -1):
                if words[i].endswith('.') or words[i].endswith(','):
                    response = ' '.join(words[:i+1])
                    break
    
    return response

# =============================================================================
# DRM Experiment Functions
# =============================================================================
def read_word_list(file_path, lure_word):
    """Read word list from CSV file for a specific lure word"""
    df = pd.read_csv(file_path)
    # Assuming the word list is in a column named 'Related Word' and lure words in 'Lure Word'
    words = df[df['Lure Word'] == lure_word]['Related Word'].dropna().tolist()
    random.shuffle(words)
    return words

def count_tokens(tokenizer, text):
    """Count tokens for a given text using the tokenizer"""
    tokens = tokenizer.encode(text)
    return len(tokens)

def generate_addition_problems(tokenizer, token_limit):
    """Generate double-digit addition problems within token limit"""
    problems = []
    total_tokens = 0
    sample_problem = "45 + 78 = ?"
    tokens_per_problem = count_tokens(tokenizer, sample_problem)

    while total_tokens + tokens_per_problem <= token_limit:
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        problem = f"{a} + {b} = ?"
        problems.append(problem)
        total_tokens += tokens_per_problem

    return "\n".join(problems), total_tokens

def get_model_response(tokenizer, model, model_config, messages, prompt, session_num=1, max_tokens=512):
    """Get response from model and log the interaction with improved error handling"""
    messages.append({"role": "user", "content": prompt})
    token_count = count_tokens(tokenizer, prompt)
    print(f"Session {session_num} - Prompt: {prompt[:100]}...")  # Truncate long prompts in logs
    print(f"Session {session_num} - Token count: {token_count}")

    try:
        # Use shorter responses for recall tasks to prevent loops
        if "memory study" in prompt.lower() or "list all the words" in prompt.lower():
            max_tokens = min(256, max_tokens)  # Shorter for recall tasks
        elif "recall" in prompt.lower() and "words" in prompt.lower():
            max_tokens = min(150, max_tokens)  # Even shorter for word recall to prevent confabulation
            
        response_text = generate_response(tokenizer, model, model_config, messages, max_new_tokens=max_tokens)
        response_token_count = count_tokens(tokenizer, response_text)
        print(f"Session {session_num} - Response token count: {response_token_count}")
        print(f"Session {session_num} - Response: {response_text[:150]}...")  # Show first 150 chars

        # Check for obviously irrelevant responses in recall tasks
        if "recall" in prompt.lower() and "words" in prompt.lower():
            concerning_patterns = [
                "mental health", "disorder", "disease", "medical", "diagnosis",
                "weather", "storm", "natural disaster", "flood", "earthquake",
                "sorry", "cannot", "can't help", "don't understand"
            ]
            response_lower = response_text.lower()
            if any(pattern in response_lower for pattern in concerning_patterns):
                print(f"⚠️  WARNING: Model response seems irrelevant to word recall task in session {session_num}")
                print(f"⚠️  Response contains concerning patterns - possible confabulation or confusion")

        # Append assistant response to messages
        messages.append({"role": "assistant", "content": response_text})

        # Log the prompt and response with cleaner structure
        log_entry = {
            "session": session_num,
            "model": "temp",  # Will be set by caller
            "model_name": "temp",  # Will be set by caller  
            "lure_word": "temp",  # Will be set by caller
            "performance_tier": "temp",  # Will be set by caller
            "section": "temp",  # Will be set by caller
            "prompt": prompt,
            "response": response_text,
            "prompt_tokens": token_count,
            "response_tokens": response_token_count,
            "total_tokens": token_count + response_token_count
        }

        return response_text, token_count + response_token_count, log_entry

    except Exception as e:
        print(f"❌ Error in model generation: {e}")
        return "", token_count, {"session": session_num, "prompt": prompt, "error": str(e)}

# =============================================================================
# Main DRM Experiment
# =============================================================================
def run_drm_experiment(model_key, num_sessions=5, word_list_file="related_words.csv", 
                       lure_word="anger", total_token_limit=4096, submit_individual_problems=True):
    """Run the complete DRM memory experiment on a specific model with complete session isolation"""
    
    logs = []
    
    for session in range(1, num_sessions + 1):
        # Load the model fresh for each session to ensure complete isolation
        print(f"\n🔄 Loading model fresh for Session {session} to ensure complete isolation...")
        tokenizer, model, model_config = load_model(model_key)
        if model is None:
            print(f"❌ Failed to load model for session {session}")
            continue
        try:
            word_list = read_word_list(word_list_file, lure_word)
        except Exception as e:
            print(f"Error reading word list: {e}")
            print("Using default word list...")
            word_list = ["angry", "mad", "fury", "rage", "hate", "irritated", "upset", "frustrated", "livid", "fuming"]
        
        current_token_count = 0
        # Fresh message context for each session - complete isolation with test-taking role
        system_prompt = ("You are a student doing a memory test which has 3 parts: "
                        "1) First, you will be asked to remember a list of words "
                        "2) Then, you will solve some math problems "
                        "3) Finally, you will be asked to recall the words from step 1. "
                        "Please complete each section to the best of your ability and follow instructions carefully. ")
        messages = [{"role": "system", "content": system_prompt}]

        # Step 1: Present the word list
        present_prompt = f"Please remember the following words: {', '.join(word_list)}."
        print(f"Session {session} - Presenting words to {model_key}...")
        present_response, token_count, log_entry = get_model_response(
            tokenizer, model, model_config, messages, present_prompt, session_num=session
        )
        print(f"✅ Word list presented: {present_response[:100]}...")
        
        # Update log entry with metadata
        log_entry.update({
            "section": "word_list",
            "model": model_key,
            "model_name": model_config["name"],
            "lure_word": lure_word,
            "performance_tier": model_config["performance_tier"]
        })
        logs.append(log_entry)
        word_list_tokens = token_count
        current_token_count += token_count

        # Step 2: Filler task with addition problems
        token_limit_for_math = 80  # Allow ~8-10 problems at 8 tokens each
        addition_problems_str, math_problems_tokens = generate_addition_problems(tokenizer, token_limit_for_math)
        math_problems_response_tokens = 0

        if submit_individual_problems:
            # Submit each addition problem one at a time
            addition_problems_list = addition_problems_str.split('\n')
            for problem in addition_problems_list:
                if problem.strip():  # Skip empty lines
                    filler_prompt = f"Please solve the following addition problem and respond only with the numerical answer: {problem}"
                    print(f"Session {session} - Performing filler task for problem: {problem}")
                    filler_response, token_count, log_entry = get_model_response(
                        tokenizer, model, model_config, messages, filler_prompt, session_num=session
                    )
                    print(f"🧮 Math solved: {filler_response[:50]}...")
                    
                    # Update log entry with metadata
                    log_entry.update({
                        "section": "filler_task",
                        "model": model_key,
                        "model_name": model_config["name"],
                        "lure_word": lure_word,
                        "performance_tier": model_config["performance_tier"]
                    })
                    logs.append(log_entry)
                    math_problems_response_tokens += token_count
                    current_token_count += token_count
        else:
            # Submit all addition problems at once
            filler_prompt = f"Please solve all of the following addition problems and respond only with the numerical answers (one per line):\n{addition_problems_str}"
            print(f"Session {session} - Performing filler task...")
            filler_response, token_count, log_entry = get_model_response(
                tokenizer, model, model_config, messages, filler_prompt, session_num=session
            )
            print(f"🧮 Math batch solved: {filler_response[:50]}...")
            
            # Update log entry with metadata
            log_entry.update({
                "section": "filler_task",
                "model": model_key,
                "model_name": model_config["name"],
                "lure_word": lure_word,
                "performance_tier": model_config["performance_tier"]
            })
            logs.append(log_entry)
            math_problems_response_tokens = token_count
            current_token_count += token_count

        # Step 3: Recall the words
        recall_prompt = "Please recall the list of words I asked you to remember earlier. List only the words you remember, separated by commas, with no additional text or explanation."
        print(f"Session {session} - Asking for recall...")
        recall_response, token_count, log_entry = get_model_response(
            tokenizer, model, model_config, messages, recall_prompt, session_num=session
        )
        print(f"🧠 Recalled words: {recall_response[:100]}...")
        
        # Update log entry with metadata
        log_entry.update({
            "section": "recall",
            "model": model_key,
            "model_name": model_config["name"],
            "lure_word": lure_word,
            "performance_tier": model_config["performance_tier"]
        })
        logs.append(log_entry)
        recall_tokens = token_count
        current_token_count += token_count

        # Create session summary matching notebook format
        session_summary = {
            "session": session,
            "total_token_limit": total_token_limit,
            "total_tokens_for_word_inputs": word_list_tokens,
            "total_tokens_for_instructions": count_tokens(tokenizer, present_prompt) + count_tokens(tokenizer, filler_prompt) + count_tokens(tokenizer, recall_prompt),
            "total_tokens_for_math_problems": math_problems_tokens,
            "total_tokens_for_math_problem_responses": math_problems_response_tokens,
            "total_tokens_for_llm_response_for_word_recall": recall_tokens,
            "total_token_count_for_experiment": current_token_count,
            "model": model_key,
            "model_name": model_config["name"],
            "performance_tier": model_config["performance_tier"],
            "lure_word": lure_word,
            "section": "session_summary"
        }
        logs.append(session_summary)

        print(f"Session {session} - Total token count for the entire experiment: {current_token_count}")
        if current_token_count <= total_token_limit:
            print(f"Session {session} - The experiment fits within the token limit.")
        else:
            print(f"Session {session} - The experiment exceeds the token limit.")
        
        # Clean up memory after each session for complete isolation
        print(f"🧹 Cleaning up memory after Session {session}...")
        del tokenizer, model, model_config, messages
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"✅ Session {session} complete and memory cleaned\n")
    
    return logs

# =============================================================================
# Performance Analysis Functions
# =============================================================================
def clean_word(word):
    """Clean a word for comparison - remove punctuation, whitespace, convert to lowercase"""
    import re
    if pd.isna(word) or word is None:
        return ""
    
    # Convert to lowercase and remove punctuation/whitespace
    cleaned = re.sub(r'[^\w]', '', str(word).lower().strip())
    
    # Handle common word variations to improve matching
    word_variations = {
        'irritation': 'irritated',
        'frustration': 'frustrated',
        'madness': 'mad',
        'anger': 'angry',  # Note: 'anger' is typically the lure word
        'furious': 'fury',
        'enraged': 'rage',
        'hatred': 'hate',
        'lividity': 'livid'
    }
    
    # Convert variations to base form
    if cleaned in word_variations:
        cleaned = word_variations[cleaned]
    
    return cleaned

def extract_original_words(df_all):
    """Extract the original word list from the word_list prompts"""
    word_list_entries = df_all[df_all['section'] == 'word_list']
    all_original_words = set()
    
    for _, entry in word_list_entries.iterrows():
        prompt = entry['prompt']
        # Extract words from "Please remember the following words: word1, word2, ..."
        if 'Please remember the following words:' in prompt:
            words_part = prompt.split('Please remember the following words:')[1].strip(' .')
            words = [clean_word(w) for w in words_part.split(', ')]
            all_original_words.update([w for w in words if w])  # Remove empty strings
    
    return all_original_words

def extract_recalled_words(recall_response):
    """Extract words from a recall response"""
    if pd.isna(recall_response) or not recall_response:
        return set()
    
    # Split by common delimiters and clean
    import re
    words = re.split(r'[,\n\s]+', str(recall_response))
    cleaned_words = {clean_word(w) for w in words if clean_word(w)}
    return cleaned_words

def analyze_recall_performance(df_all, base_filename):
    """Analyze recall performance and create detailed performance metrics"""
    
    # Get original words presented to the model
    original_words = extract_original_words(df_all)
    lure_word = df_all['lure_word'].iloc[0] if not df_all.empty else "unknown"
    model_name = df_all['model'].iloc[0] if not df_all.empty else "unknown"
    
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
        lure_word_clean = clean_word(lure_word)
        lure_word_hallucinated = lure_word_clean in recalled_words
        
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
    
    # Save performance analysis
    if performance_data:
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(f"{base_filename}_performance.csv", index=False)
        
        # Print summary to console
        print(f"\n🎯 PERFORMANCE SUMMARY for {model_name}:")
        for entry in performance_data:
            print(f"   Session {entry['session']}: {entry['recall_accuracy_percent']:.1f}% accuracy, "
                  f"{entry['false_memory_rate_percent']:.1f}% false memories, "
                  f"Lure word: {'YES' if entry['lure_word_hallucinated'] else 'NO'}")
    
    return performance_data

# =============================================================================
# Simple CSV Output Function
# =============================================================================
def save_summary_results(results, model_key, lure_word, num_sessions):
    """Save a single comprehensive summary CSV file"""
    
    df_all = pd.DataFrame(results)
    
    # Extract recall data for analysis
    recalls = df_all[df_all['section'] == 'recall'].copy()
    if recalls.empty:
        print(f"⚠️  No recall data found for {model_key} - {lure_word}")
        return
    
    # Analyze recall performance
    performance_data = analyze_recall_performance(df_all, f"temp_{model_key}_{lure_word}")
    
    # Create summary entry for this model-lure combination
    if performance_data:
        # Calculate averages across all sessions
        avg_accuracy = sum(entry['recall_accuracy_percent'] for entry in performance_data) / len(performance_data)
        avg_precision = sum(entry['precision_percent'] for entry in performance_data) / len(performance_data)
        avg_false_memory = sum(entry['false_memory_rate_percent'] for entry in performance_data) / len(performance_data)
        lure_recall_count = sum(1 for entry in performance_data if entry['lure_word_hallucinated'])
        
        return {
            'model': model_key,
            'model_name': df_all['model_name'].iloc[0],
            'performance_tier': df_all['performance_tier'].iloc[0],
            'lure_word': lure_word,
            'num_sessions': num_sessions,
            'avg_recall_accuracy_percent': round(avg_accuracy, 2),
            'avg_precision_percent': round(avg_precision, 2), 
            'avg_false_memory_rate_percent': round(avg_false_memory, 2),
            'lure_word_hallucinated_count': lure_recall_count,
            'lure_word_hallucination_rate_percent': round((lure_recall_count / num_sessions) * 100, 2),
            'total_sessions_completed': len(performance_data)
        }
    
    return None

# =============================================================================
# Comparative DRM Analysis
# =============================================================================
def run_comparative_drm_study(models_to_test, num_sessions=3, word_list_file="related_words.csv", 
                              lure_words=["anger", "sleep", "doctor"], submit_individual_problems=True):
    """Run DRM experiment across multiple models and lure words for comparison"""
    
    all_trial_results = []  # Store detailed trial data
    all_interaction_logs = []  # Store detailed interaction logs
    summary_results = []    # Store summary data
    
    for model_key in models_to_test:
        for lure_word in lure_words:
            print(f"\n{'='*60}")
            print(f"🧠 Testing {model_key.upper()} with lure word: '{lure_word}'")
            print(f"{'='*60}")
            
            # Adjust token limit based on model context window
            model_config = MODELS[model_key]
            if model_config["context_window"] >= 30000:
                token_limit = 8192
            elif model_config["context_window"] >= 8000:
                token_limit = 4096
            else:
                token_limit = 2048
            
            results = run_drm_experiment(
                model_key=model_key,
                num_sessions=num_sessions,
                word_list_file=word_list_file,
                lure_word=lure_word,
                total_token_limit=token_limit,
                submit_individual_problems=submit_individual_problems
            )
            
            if results:
                # Extract individual trial results
                df_all = pd.DataFrame(results)
                recalls = df_all[df_all['section'] == 'recall'].copy()
                
                # Extract detailed interaction logs
                for _, row in df_all.iterrows():
                    if row['section'] in ['word_list', 'filler_task', 'recall']:
                        # Clean and truncate response for readability
                        response_preview = str(row['response'])[:200] + "..." if len(str(row['response'])) > 200 else str(row['response'])
                        
                        interaction_log = {
                            'model': model_key,
                            'model_name': row['model_name'],
                            'performance_tier': row['performance_tier'],
                            'lure_word': lure_word,
                            'session_number': row['session'],
                            'phase': row['section'],
                            'prompt': str(row['prompt']),
                            'response': str(row['response']),
                            'response_preview': response_preview,
                            'prompt_tokens': row.get('prompt_tokens', 0),
                            'response_tokens': row.get('response_tokens', 0),
                            'total_tokens': row.get('total_tokens', 0)
                        }
                        all_interaction_logs.append(interaction_log)
                
                # Get detailed performance data for each session
                performance_data = analyze_recall_performance(df_all, f"temp_{model_key}_{lure_word}")
                
                # Add individual trial data
                for entry in performance_data:
                    trial_result = {
                        'result_type': 'individual_trial',
                        'model': model_key,
                        'model_name': df_all['model_name'].iloc[0],
                        'performance_tier': df_all['performance_tier'].iloc[0],
                        'lure_word': lure_word,
                        'session_number': entry['session'],
                        'recall_accuracy_percent': entry['recall_accuracy_percent'],
                        'precision_percent': entry['precision_percent'],
                        'false_memory_rate_percent': entry['false_memory_rate_percent'],
                        'lure_word_hallucinated': entry['lure_word_hallucinated'],
                        'total_original_words': entry['total_original_words'],
                        'total_recalled_words': entry['total_recalled_words'],
                        'correctly_recalled_count': entry['correctly_recalled_count'],
                        'false_memories_count': entry['false_memories_count'],
                        'correctly_recalled_words': entry['correctly_recalled_words'],
                        'false_memories': entry['false_memories'],
                        'missed_words': entry['missed_words']
                    }
                    all_trial_results.append(trial_result)
                
                # Get summary data for this model-lure combination
                if performance_data:
                    # Calculate averages across all sessions
                    avg_accuracy = sum(entry['recall_accuracy_percent'] for entry in performance_data) / len(performance_data)
                    avg_precision = sum(entry['precision_percent'] for entry in performance_data) / len(performance_data)
                    avg_false_memory = sum(entry['false_memory_rate_percent'] for entry in performance_data) / len(performance_data)
                    lure_recall_count = sum(1 for entry in performance_data if entry['lure_word_hallucinated'])
                    
                    summary_result = {
                        'result_type': 'summary',
                        'model': model_key,
                        'model_name': df_all['model_name'].iloc[0],
                        'performance_tier': df_all['performance_tier'].iloc[0],
                        'lure_word': lure_word,
                        'session_number': 'ALL',
                        'num_sessions': num_sessions,
                        'recall_accuracy_percent': round(avg_accuracy, 2),
                        'precision_percent': round(avg_precision, 2),
                        'false_memory_rate_percent': round(avg_false_memory, 2),
                        'lure_word_hallucinated': f"{lure_recall_count}/{num_sessions}",
                        'lure_word_hallucination_rate_percent': round((lure_recall_count / num_sessions) * 100, 2),
                        'total_sessions_completed': len(performance_data),
                        'total_original_words': sum(entry['total_original_words'] for entry in performance_data),
                        'total_recalled_words': sum(entry['total_recalled_words'] for entry in performance_data),
                        'correctly_recalled_count': sum(entry['correctly_recalled_count'] for entry in performance_data),
                        'false_memories_count': sum(entry['false_memories_count'] for entry in performance_data)
                    }
                    all_trial_results.append(summary_result)
                    
                print(f"✅ Results processed for {model_key} - {lure_word} ({len(performance_data)} sessions)")
    
    # Save comprehensive results in two CSV files
    if all_trial_results and all_interaction_logs:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. Results file (trials + summaries)
        results_file = f"drm_hallucination_complete_results_{timestamp}.csv"
        results_df = pd.DataFrame(all_trial_results)
        results_df = results_df.sort_values(['model', 'lure_word', 'result_type', 'session_number'])
        results_df.to_csv(results_file, index=False)
        
        # 2. Interaction logs file (prompts + responses)
        logs_file = f"drm_hallucination_interaction_logs_{timestamp}.csv"
        logs_df = pd.DataFrame(all_interaction_logs)
        
        # Create proper phase ordering (execution order, not alphabetical)
        phase_order = {'word_list': 1, 'filler_task': 2, 'recall': 3}
        logs_df['phase_order'] = logs_df['phase'].map(phase_order)
        logs_df = logs_df.sort_values(['model', 'lure_word', 'session_number', 'phase_order'])
        logs_df = logs_df.drop('phase_order', axis=1)  # Remove the helper column
        logs_df.to_csv(logs_file, index=False)
        
        # Count totals
        total_trials = len([r for r in all_trial_results if r['result_type'] == 'individual_trial'])
        total_summaries = len([r for r in all_trial_results if r['result_type'] == 'summary'])
        total_interactions = len(all_interaction_logs)
        
        print(f"\n🎯 RESULTS SAVED TO TWO FILES:")
        print(f"📊 1. {results_file}")
        print(f"   - Individual trials: {total_trials}")
        print(f"   - Model summaries: {total_summaries}")
        print(f"💬 2. {logs_file}")
        print(f"   - Interaction logs: {total_interactions}")
        
        # Print summary analysis to console
        print(f"\n{'='*80}")
        print("📈 DRM HALLUCINATION EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Lure':<10} {'Sessions':<10} {'Accuracy':<10} {'False Mem':<12} {'Lure Recall':<12}")
        print("-" * 80)
        
        summary_rows = [r for r in all_trial_results if r['result_type'] == 'summary']
        for entry in summary_rows:
            print(f"{entry['model']:<20} {entry['lure_word']:<10} "
                  f"{entry['total_sessions_completed']:>7} "
                  f"{entry['recall_accuracy_percent']:>7.1f}% "
                  f"{entry['false_memory_rate_percent']:>9.1f}% "
                  f"{entry['lure_word_hallucination_rate_percent']:>9.1f}%")
        
        print(f"\n📁 FILES SAVED:")
        print(f"   📊 {results_file} - Trial results & summaries")
        print(f"   💬 {logs_file} - Detailed interaction logs")
        print("\n💡 USAGE:")
        print("   📊 Use results file for statistical analysis")
        print("   💬 Use logs file to examine actual model responses")
        
        return results_file, logs_file
    else:
        print("❌ No results to save")
        return None

# =============================================================================
# Hallucination Test Example (Original)
# =============================================================================
def run_hallucination_test(model_key, test_prompts):
    """Run hallucination tests on a specific model"""
    
    tokenizer, model, model_config = load_model(model_key)
    if model is None:
        return None
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} for {model_key} ---")
        print(f"Prompt: {prompt}")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Please be accurate and factual."},
            {"role": "user", "content": prompt}
        ]
        
        start_time = time.time()
        response = generate_response(tokenizer, model, model_config, messages)
        end_time = time.time()
        
        result = {
            "model": model_key,
            "model_name": model_config["name"],
            "performance_tier": model_config["performance_tier"],
            "prompt": prompt,
            "response": response,
            "response_time": end_time - start_time
        }
        
        results.append(result)
        print(f"Response: {response}")
        print(f"Time: {result['response_time']:.2f}s")
    
    return results

# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    print("🧠 Multi-Model DRM Memory Experiment")
    print("=====================================")
    
    # Models to test (9 downloaded models with 4096 token context windows)
    test_models = [
        # GPT-3.5 Level Models (7B parameters, 4096 tokens)
        "llama2-7b", "mistral-7b-v1", "zephyr-7b",
        # GPT-4.0 Level Models (7B parameters) 
        "qwen-7b", "mistral-7b-v3", "openchat-7b",
        # Reasoning Models (7B parameters)
        "deepseek-coder-7b", "codellama-7b", "nous-hermes-7b"
    ]
    
    # Run comparative DRM study
    result_files = run_comparative_drm_study(
        models_to_test=test_models,
        num_sessions=10,  # 10 sessions per model-lure combination
        lure_words=["anger"],  # Start with one lure word
        submit_individual_problems=True
    )
    
    if result_files:
        results_file, logs_file = result_files
        print(f"\n✅ DRM experiment completed successfully!")
        print(f"\n📊 TWO CSV FILES GENERATED:")
        print(f"   1. {results_file}")
        print(f"      🔍 Individual trial results (each session)")
        print(f"      📈 Summary statistics (averaged across sessions)")
        print(f"      📊 Model performance metrics")
        print(f"   2. {logs_file}")
        print(f"      💬 Complete interaction logs")
        print(f"      🗣️ All prompts and responses")
        print(f"      📝 Token counts and phases")
        print("\n💡 ANALYSIS TIPS:")
        print("   📊 Use results file for statistical analysis")
        print("   🔍 Filter by 'result_type': 'individual_trial' or 'summary'")
        print("   💬 Use logs file to examine actual model conversations")
        print("   📋 Filter logs by 'phase': 'word_list', 'filler_task', or 'recall'")
    else:
        print("\n❌ Experiment failed - no results generated") 