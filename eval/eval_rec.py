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
CONFIG = {
    'Rec-Movie': {
        'checkpoint_path': '',
        'prompt_template': "I've watched the following movies in the past, in order:\n{user_his_text}\n\n" \
                          "Now there are {num_candidates} candidate movies that I might watch next:\n{candidate_text_order}\n\n" \
                          "Please select the one movie that is least likely to be my next watch, according to my watching history. Please think step by step.\n" \
                          "You MUST choose exactly one movie from the given candidate list.\n" \
                          "You can NOT generate or reference movies that are not in the given candidate list.\n" \
                          "Return only the full name of the movie."
    },
    'Rec-Game': {
        'checkpoint_path': '',
        'prompt_template': "I've purchased the following items in the past, in order:\n{user_his_text}\n\n" \
                          "Now there are {num_candidates} candidate items that I might purchase next:\n{candidate_text_order}\n\n" \
                          "Please select the one item that is least likely to be my next purchase, according to my purchase history. Please think step by step.\n" \
                          "You MUST choose exactly one item from the given candidate list.\n" \
                          "You can NOT generate or reference items that are not in the given candidate list.\n" \
                          "Return only the full name of the item."
    },
    'Rec-Music': {
        'checkpoint_path': '',
        'prompt_template': "I've purchased the following items in the past, in order:\n{user_his_text}\n\n" \
                          "Now there are {num_candidates} candidate items that I might purchase next:\n{candidate_text_order}\n\n" \
                          "Please select the one item that is least likely to be my next purchase, according to my purchase history. Please think step by step.\n" \
                          "You MUST choose exactly one item from the given candidate list.\n" \
                          "You can NOT generate or reference items that are not in the given candidate list.\n" \
                          "Return only the full name of the item."
    }
}

def preprocess_text(text: str) -> List[str]:
    """Preprocess text by converting to lowercase, removing punctuation, and tokenizing."""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    return [token for token in text.split() if token.strip()]

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

def find_most_similar(response: str, candidate_texts: List[str]) -> str:
    """Find most similar text from candidates using TF-IDF and cosine similarity."""
    vectorizer = TfidfVectorizer()
    all_texts = [response] + candidate_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return candidate_texts[similarities.argmax()]

def make_prefix(query: str) -> str:
    """Create prefix for Qwen Instruct Models."""
    return f"""<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n {query} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    # return f"""{query} Show your thinking process and give definitive answer in the end."""

def construct_prompt(dataset_name: str, user_his_text: str, candidate_text_order: List[str]) -> str:
    """Construct prompt for the model."""
    if dataset_name not in CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(CONFIG.keys())}")
    
    template = CONFIG[dataset_name]['prompt_template']
    return template.format(
        user_his_text=user_his_text,
        num_candidates=len(candidate_text_order),
        candidate_text_order=candidate_text_order
    )

def get_prompt(dataset_name: str, user_his_text: str, candidate_text_order: List[str]) -> str:
    """Get complete prompt with prefix."""
    ini_query = construct_prompt(dataset_name, user_his_text, candidate_text_order)
    return make_prefix(ini_query)

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

        matched_name = None
        matched_candidate_index = -1

        for i, candidate_text_single in enumerate(candidate_list):
            clean_candidate_text_single = html.unescape(candidate_text_single.strip())

            # Define your matching criteria here. You can adjust these conditions.
            if (clean_candidate_text_single in generated_item_name) or \
               (generated_item_name in clean_candidate_text_single) or \
               (pylcs.lcs_sequence_length(generated_item_name, clean_candidate_text_single) > 0.9 * len(clean_candidate_text_single)):
                if i not in used_candidate_indices:
                    matched_name = candidate_text_single
                    matched_candidate_index = i
                    break

        if matched_name is not None:
            matched_candidate_items.append(matched_name)
            used_candidate_indices.add(matched_candidate_index)

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
            user_his_text = sample['problem'].split("Now there are")[0].split("in order:")[1].strip()
            prompt = get_prompt(dataset_name, user_his_text, sample['candidates'])
            batch_prompts.append(prompt)
        
        outputs = llm.generate(batch_prompts, sampling_params)

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
            else:
                matched_items = match_and_order_lists([response_list], sample['candidates'])
                final_response = matched_items[0] if matched_items else find_most_similar(response_list, sample['candidates'])
            if response_list != final_response:
                print(f"Vanil response: {equation}")
                print(f"Final response: {final_response}\n")
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
                    'final_remaining': len(sample['candidates'])
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
        'hit@20': [],
        'ndcg@5': [],
        'ndcg@10': [],
        'ndcg@20': [],
        'mrr': []
    }
    
    for result in results:
        ground_truth_rank = result['ground_truth_rank']
        
        # Calculate Hit@k
        for k in [5, 10, 20]:
            metrics[f'hit@{k}'].append(1 if ground_truth_rank <= k else 0)
        
        # Calculate NDCG@k
        for k in [5, 10, 20]:
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

    # Load dataset from HuggingFace and filter by task_name
    print("Loading input data from HuggingFace dataset...")
    ds = load_dataset("ulab-ai/Ranking-bench", "direct")['test']
    filtered_data = [x for x in ds if x.get('task_name') == dataset_name]
    if not filtered_data:
        raise ValueError(f"No data found for task_name={dataset_name} in the loaded dataset.")
    # Each item in filtered_data should have 'problem', 'gt', and 'candidate_items'
    # If not, you may need to adapt the field names accordingly
    # Process all data in one go
    results = process_batch(filtered_data, llm, sampling_params, dataset_name)
    
    # Calculate and save metrics
    metrics = calculate_metrics(results)
    print("\nEvaluation Results:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")
    
    output_file = f"./eval/eval_result/{dataset_name}_batch_{config['checkpoint_path'].split('/')[-1]}.json"
    print(f"\nSaving results to {output_file}...")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate recommendation model on different datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['Rec-Movie', 'Rec-Game', 'Rec-Music'],
                      help='Dataset to evaluate on (Rec-Movie, Rec-Game, Rec-Music)')
    parser.add_argument('--gpu_id', type=str, default='3',
                      help='GPU ID to use (default: 0)')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint to use for evaluation')
    args = parser.parse_args()
    main(args.dataset, args.gpu_id, args.model_path) 