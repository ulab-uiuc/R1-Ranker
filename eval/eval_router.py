import json
import os
import numpy as np
# from rank_bm25 import BM25Okapi
import re
import string
from typing import List, Dict, Any
import torch
import gc
from tqdm import tqdm
from vllm import LLM, SamplingParams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import html
import pylcs
from datasets import load_dataset

# Configuration
LLM_DESCRIPTIONS = {
    "LLaMA-3 (8b)": "Each token price is 0.2. Handles simple Hybrid QA, basic reasoning, short reading comprehension, and concise summaries.",
    "Mixtral-8x7B": "Each token price is 0.6. Excels at instruction-based tasks, step-by-step reasoning, solid reading comprehension, and structured summaries.",
    "NousResearch (34b)": "Each token price is 0.9. Ideal for complex research queries, multi-step reasoning, deep reading comprehension, and thorough summaries.",
    "LLaMA-2 (7b)": "Each token price is 0.2. Lightweight chat model for straightforward Hybrid QA, moderate reasoning, short reading comprehension, and concise summaries.",
    "Mistral-7b": "Each token price is 0.2. Fast for moderate Hybrid QA, quick reasoning, short reading comprehension, and brief summaries.",
    "LLaMA-3 (70b)": "Each token price is 0.9. High-capacity model for advanced Hybrid QA, deep reasoning, detailed reading comprehension, and extensive summaries.",
    "LLaMA-3-Turbo (8b)": "Each token price is 0.2. Balanced performance for moderate Hybrid QA, reasonable reasoning, clear reading comprehension, and concise summaries.",
    "LLaMA-3-Turbo (70b)": "Each token price is 0.9. Powerful model for advanced Hybrid QA, strong reasoning, in-depth reading comprehension, and high-quality summaries.",
    "Llama-3.1-Turbo (70b)": "Each token price is 0.9. Instruction-focused, providing thorough reasoning, structured reading comprehension, and well-organized summaries.",
    "Qwen-1.5 (72b)": "Each token price is 0.9. Versatile for challenging Hybrid QA, nuanced reasoning, extended reading comprehension, and comprehensive summaries."
}

CONFIG = {
    'Router-Balance': {
        'checkpoint_path': '',
        'prompt_template': """I've been given the following query:
{query}

The LLM names and their descriptions are:
{llm_descriptions}

Now there are {num_candidates} candidate LLMs that could handle this query:
{candidate_text_order}

Please select the one LLM that is least BALANCED for handling this query, considering both the query requirements and cost efficiency. Please think step by step.
You MUST choose exactly one LLM from the given candidate list.
You can NOT generate or reference LLMs that are not in the given candidate list.
Return only the full name of the LLM."""
    },
    'Router-Cost': {
        'checkpoint_path': '',
        'prompt_template': """I've been given the following query:
{query}

The LLM names and their descriptions are:
{llm_descriptions}

Now there are {num_candidates} candidate LLMs that could handle this query:
{candidate_text_order}

Please select the one LLM that is least COST-EFFICIENT for handling this query, prioritizing cost efficiency while still meeting the minimum requirements. Please think step by step.
You MUST choose exactly one LLM from the given candidate list.
You can NOT generate or reference LLMs that are not in the given candidate list.
Return only the full name of the LLM."""
    },
    'Router-Performance': {
        'checkpoint_path': '',
        'prompt_template': """I've been given the following query:
{query}

The LLM names and their descriptions are:
{llm_descriptions}

Now there are {num_candidates} candidate LLMs that could handle this query:
{candidate_text_order}

Please select the one LLM that is least PERFORMANCE-EFFICIENT for handling this query, prioritizing performance and quality while still being cost-conscious. Please think step by step.
You MUST choose exactly one LLM from the given candidate list.
You can NOT generate or reference LLMs that are not in the given candidate list.
Return only the full name of the LLM."""
    }
}

def make_prefix(query: str) -> str:
    """Create prefix for Qwen Instruct Models."""
    return f"""<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n {query} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""

def construct_prompt(dataset_name: str, query: str, candidate_text_order: List[str]) -> str:
    """Construct prompt for the model."""
    if dataset_name not in CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(CONFIG.keys())}")
    
    # Get descriptions for only the LLMs in the candidate list
    llm_descriptions = []
    for llm in candidate_text_order:
        if llm in LLM_DESCRIPTIONS:
            llm_descriptions.append(f"# {llm}: {LLM_DESCRIPTIONS[llm]}")
    
    template = CONFIG[dataset_name]['prompt_template']
    return template.format(
        query=query,
        num_candidates=len(candidate_text_order),
        candidate_text_order=candidate_text_order,
        llm_descriptions="\n".join(llm_descriptions)
    )

def get_prompt(dataset_name: str, query: str, candidate_text_order: List[str]) -> str:
    """Get complete prompt with prefix."""
    ini_query = construct_prompt(dataset_name, query, candidate_text_order)
    return make_prefix(ini_query)

def extract_solution(solution_str: str) -> str:
    """Extract solution from model output."""
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if len(matches) >= 2:
        return matches[-1].group(1).strip()
    
    strict_pattern = r"<answer>\n(.*?)\n</answer>"
    strict_match = re.search(strict_pattern, solution_str, re.DOTALL)
    if strict_match:
        return strict_match.group(1)
    
    flexible_pattern = r"<answer>\s*(.*?)\s*</answer>"
    flexible_match = re.search(flexible_pattern, solution_str, re.DOTALL)
    if flexible_match:
        return flexible_match.group(1).strip()
    
    return None

def preprocess_text(text: str) -> List[str]:
    """Preprocess text by converting to lowercase, removing punctuation, and tokenizing."""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    return [token for token in text.split() if token.strip()]

def find_most_similar(response: str, candidate_texts: List[str]) -> str:
    """Find most similar text from candidates using TF-IDF and cosine similarity."""
    vectorizer = TfidfVectorizer()
    all_texts = [response] + candidate_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return candidate_texts[similarities.argmax()]

def extract_model_info(text: str) -> tuple:
    """
    Extract model information from text.
    Returns (backbone, variant, size, is_turbo)
    """
    text = text.lower().strip()
    
    # Check for turbo
    is_turbo = "turbo" in text
    
    # Extract size
    size = None
    if "8b" in text:
        size = "8b"
    elif "7b" in text:
        size = "7b"
    elif "70b" in text or "72b" in text:
        size = "70b"
    elif "34b" in text:
        size = "34b"
    
    # Extract backbone and variant
    backbone = None
    variant = None
    
    if "llama" in text:
        backbone = "llama"
        if "3.1" in text:
            variant = "3.1"
        elif "3" in text:
            variant = "3"
        elif "2" in text:
            variant = "2"
    elif "mixtral" in text:
        backbone = "mixtral"
        variant = "8x7b"
    elif "mistral" in text:
        backbone = "mistral"
        variant = "7b"
    elif "qwen" in text:
        backbone = "qwen"
        variant = "1.5"
    elif "nousresearch" in text:
        backbone = "nousresearch"
        variant = "34b"
    
    return (backbone, variant, size, is_turbo)

def calculate_model_similarity(model1_info: tuple, model2_info: tuple) -> float:
    """
    Calculate similarity score between two model infos.
    Each model_info is a tuple of (backbone, variant, size, is_turbo).
    Returns a score between 0 and 1.
    """
    backbone1, variant1, size1, is_turbo1 = model1_info
    backbone2, variant2, size2, is_turbo2 = model2_info
    
    score = 0.0
    components = 0
    
    # Compare backbones
    if backbone1 and backbone2:
        components += 1
        if backbone1 == backbone2:
            score += 1.0
    
    # Compare variants
    if variant1 and variant2:
        components += 1
        if variant1 == variant2:
            score += 1.0
    
    # Compare sizes
    if size1 and size2:
        components += 1
        if size1 == size2:
            score += 1.0
    
    # Compare turbo status
    if is_turbo1 is not None and is_turbo2 is not None:
        components += 1
        if is_turbo1 == is_turbo2:
            score += 1.0
    
    return score / max(components, 1)

def match_and_order_lists(generated_list, candidate_list):
    """
    Matches items from a generated list to a candidate list and returns an ordered list
    of matched candidate items, ensuring no repeated matches.

    Args:
        generated_list: A list of strings generated by a model or process.
        candidate_list: A list of strings representing candidate items to match against.

    Returns:
        A list of strings from the candidate_list that were matched to items in the
        generated_list, maintaining the order of the first match in the generated list
        and ensuring no candidate item is matched more than once.
    """
    matched_candidate_items = []
    used_candidate_indices = set()

    for generated_item_detail in generated_list:
        if not generated_item_detail:
            continue

        # Clean up the generated item text
        pr = generated_item_detail.find('. ')
        if generated_item_detail[:pr].isdigit():
            generated_item_name = generated_item_detail[pr + 2:].strip()
        else:
            generated_item_name = generated_item_detail.strip()

        # Extract model info from generated item
        generated_model_info = extract_model_info(generated_item_name)
        
        best_match = None
        best_match_index = -1
        best_match_score = -1

        for i, candidate_text_single in enumerate(candidate_list):
            if i in used_candidate_indices:
                continue
                
            clean_candidate_text_single = html.unescape(candidate_text_single.strip())
            candidate_model_info = extract_model_info(clean_candidate_text_single)
            
            # Calculate model info similarity score
            model_similarity = calculate_model_similarity(generated_model_info, candidate_model_info)
            
            # Calculate text similarity score using existing criteria
            text_similarity = 0.0
            if (clean_candidate_text_single in generated_item_name):
                text_similarity = 1.0
            elif (generated_item_name in clean_candidate_text_single):
                text_similarity = 0.9
            else:
                lcs_length = pylcs.lcs_sequence_length(generated_item_name, clean_candidate_text_single)
                if lcs_length > 0.9 * len(clean_candidate_text_single):
                    text_similarity = lcs_length / len(clean_candidate_text_single)
            
            # Combine scores with weights
            combined_score = (0.7 * model_similarity) + (0.3 * text_similarity)
            
            if combined_score > best_match_score:
                best_match = candidate_text_single
                best_match_index = i
                best_match_score = combined_score

        if best_match is not None and best_match_score > 0.5:  # Threshold for accepting a match
            matched_candidate_items.append(best_match)
            used_candidate_indices.add(best_match_index)

    return matched_candidate_items

def process_batch(batch_data: List[Dict[str, Any]], llm: LLM, sampling_params: SamplingParams, dataset_name: str) -> List[Dict[str, Any]]:
    """Process a batch of samples in parallel until all reach final state."""
    # Initialize tracking for each sample in the batch
    active_samples = []
    for sample in batch_data:
        active_samples.append({
            'problem': sample['problem'],
            'gt': sample['gt'],
            'candidates': sample['candidates'].copy(),
            'original_length': len(sample['candidates']),
            'removed_items': [],
            'ground_truth_rank': None,
            'is_complete': False
        })
    
    results = []
    iteration = 0
    rank_findings = {}  # Track number of ground truths found at each rank
    
    while True:
        iteration += 1
        print(f"\nIteration {iteration}:")
        print(f"{len(active_samples[0]['candidates'])}/{active_samples[0]['original_length']} items remaining")
        
        # Prepare batch of prompts for active samples
        batch_prompts = []
        for sample in active_samples:
            prompt = get_prompt(dataset_name, sample['problem'], sample['candidates'])
            batch_prompts.append(prompt)  

        # Generate responses for all prompts in one batch
        outputs = llm.generate(batch_prompts, sampling_params)

        print("\nSample Response:")
        print("="*80)
        print(outputs[0].outputs[0].text.strip())
        print("="*80)
        
        # Process results and update samples
        for output, sample in zip(outputs, active_samples):
            response = output.outputs[0].text.strip()
            equation = extract_solution(response)
            if equation is None:
                equation = response[-40:]
            
            response_list = equation.split('\n')[0].strip()
            # First check for exact match
            if response_list in sample['candidates']:
                final_response = response_list
            elif response_list.lower() in [x.lower() for x in sample['candidates']]:
                final_response = sample['candidates'][[x.lower() for x in sample['candidates']].index(response_list.lower())]
            else:
                matched_items = match_and_order_lists([response_list], sample['candidates'])
                final_response = matched_items[0] if matched_items else find_most_similar(response_list, sample['candidates'])
            
            if response_list != final_response:
                print(f"Vanil response: {response_list}")
                print(f"Final response: {final_response}")
                # import pdb; pdb.set_trace()
            
            # Update sample state
            sample['removed_items'].append(final_response)
            sample['candidates'].remove(final_response)
            
            # Check if ground truth was found
            if final_response == sample['gt']:
                rank = sample['original_length'] - len(sample['removed_items']) + 1
                sample['ground_truth_rank'] = rank
                rank_findings[rank] = rank_findings.get(rank, 0) + 1
        
        # Check if all samples have reached final state
        all_complete = True
        for sample in active_samples:
            if len(sample['candidates']) > 1:
                all_complete = False
                break
        
        if all_complete:
            # Process final state for all samples
            for sample in active_samples:
                if len(sample['candidates']) == 1:
                    if sample['ground_truth_rank'] is None:
                        sample['ground_truth_rank'] = 1
                        rank_findings[1] = rank_findings.get(1, 0) + 1
                
                results.append({
                    'ground_truth_rank': sample['ground_truth_rank'],
                    'removed_items': sample['removed_items'],
                    'final_candidates': sample['candidates'],
                    'iterations': iteration,
                    'final_remaining': len(sample['candidates']),
                    'problem': sample['problem'],
                    'gt': sample['gt']
                })
            break
    
    # Print summary of ground truth findings by rank
    print("\nGround Truth Findings Summary by Rank:")
    for rank in sorted(rank_findings.keys()):
        print(f"Rank {rank}: Found {rank_findings[rank]} ground truths")
    print(f"Total ground truths found: {sum(rank_findings.values())}")
    print(f"\nProcessing complete. Processed {len(results)} samples.")
    return results

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate evaluation metrics from results."""
    metrics = {
        'hit@5': [],
        'hit@10': [],
        'ndcg@5': [],
        'ndcg@10': [],
        'mrr': []
    }
    
    for result in results:
        ground_truth_rank = result['ground_truth_rank']
        
        # Calculate Hit@k
        for k in [5, 10]:
            metrics[f'hit@{k}'].append(1 if ground_truth_rank <= k else 0)
        
        # Calculate NDCG@k
        for k in [5, 10]:
            if ground_truth_rank <= k:
                dcg = 1.0 / np.log2(ground_truth_rank + 1)
                idcg = 1.0
                metrics[f'ndcg@{k}'].append(dcg / idcg)
            else:
                metrics[f'ndcg@{k}'].append(0.0)
        
        # Calculate MRR
        metrics['mrr'].append(1.0 / ground_truth_rank)
    
    # Calculate averages
    return {k: np.mean(v) for k, v in metrics.items()}

def main(dataset_name: str, gpu_id: str, model_path: str):
    if dataset_name not in CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(CONFIG.keys())}")
    
    config = CONFIG[dataset_name].copy()  # Create a copy to avoid modifying the original
    config['checkpoint_path'] = model_path  # Override the checkpoint path
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # Initialize GPU
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available!")
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    llm = LLM(
        model=config['checkpoint_path'],
        gpu_memory_utilization=0.95
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.7,
        max_tokens=1024,
        stop=["</s>", "<|endoftext|>"],
    )

    # Load all data from HuggingFace datasets
    print("Loading input data from HuggingFace datasets...")
    ds = load_dataset("ulab-ai/Ranking-bench", "direct")['test']
    # Map dataset_name to the correct subset in ds
    if dataset_name == 'Router-Balance':
        data = [x for x in ds if x['task_name'] == 'Router-Balance']
    elif dataset_name == 'Router-Cost':
        data = [x for x in ds if x['task_name'] == 'Router-Cost']
    elif dataset_name == 'Router-Performance':
        data = [x for x in ds if x['task_name'] == 'Router-Performance']
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    # Convert to expected format for process_batch
    batch_data = []
    for x in data:
        batch_data.append({
            'problem': x['problem'],
            'gt': x['gt'],
            'candidates': x['candidates']
        })
    # Process all data
    results = process_batch(batch_data, llm, sampling_params, dataset_name)
    
    # Calculate and save metrics
    metrics = calculate_metrics(results)
    print("\nEvaluation Results:")
    for metric, score in metrics.items():
        print(f"{metric}: {score}")
    
    output_file = f"./eval/eval_result/{dataset_name}_{config['checkpoint_path'].split('/')[-1]}.json"
    print(f"\nSaving results to {output_file}...")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'metrics': metrics,
            'results': results
        }, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate LLM router model')
    parser.add_argument('--dataset', type=str, default='Router-Balance',
                      choices=['Router-Balance', 'Router-Cost', 'Router-Performance'],
                      help='Dataset to evaluate on (default: Router-Balance)')
    parser.add_argument('--gpu_id', type=str, default='0',
                      help='GPU ID to use (default: 0)')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint to use for evaluation')
    args = parser.parse_args()
    main(args.dataset, args.gpu_id, args.model_path) 