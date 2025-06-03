"""
MS MARCO Dataset Processing Script

This script processes the MS MARCO dataset to create training and evaluation samples
for passage ranking and selection tasks. It handles data cleaning, formatting, and
generates various types of prompts for different evaluation scenarios.
"""

import pandas as pd
import json
import random
from typing import List, Dict, Any
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessedSample:
    """Data class to store processed sample information."""
    query: str
    selected_passage: str
    candidate_passages: Dict[str, str]


def process_list(original_dict: Dict[str, str], selected_passage_id: str) -> List[Dict[str, str]]:
    """
    Process a dictionary of passages to create variations with different numbers of candidates.
    
    Args:
        original_dict: Dictionary mapping passage IDs to passage text
        selected_passage_id: ID of the passage that should be included in all variations
        
    Returns:
        List of dictionaries, each containing a subset of passages with the selected passage
        inserted at a random position
    """
    results = []
    selected_passage = original_dict[selected_passage_id]
    
    # Create working dictionary without selected passage
    working_dict = original_dict.copy()
    del working_dict[selected_passage_id]
    
    ids = list(working_dict.keys())
    passages = list(working_dict.values())
    
    # Generate variations for different numbers of candidates
    for k in range(1, len(working_dict) + 1):
        # Shuffle passages while maintaining ID-passage correspondence
        combined = list(zip(ids, passages))
        random.shuffle(combined)
        shuffled_ids, shuffled_passages = zip(*combined) if combined else ([], [])
        
        # Take top k elements
        top_k_ids = shuffled_ids[:k]
        top_k_passages = shuffled_passages[:k]
        
        # Create result dictionary with selected passage inserted at random position
        insert_position = random.randint(0, k)
        result_dict = {}
        
        # Add passages before insert position
        for i, id_val in enumerate(top_k_ids[:insert_position]):
            result_dict[id_val] = top_k_passages[i]
            
        # Add selected passage
        result_dict[selected_passage_id] = selected_passage
        
        # Add passages after insert position
        for i, id_val in enumerate(top_k_ids[insert_position:]):
            result_dict[id_val] = top_k_passages[insert_position + i]
        
        results.append(result_dict)
    
    return results


def process_msmarco_data(df: pd.DataFrame, min_total_chars: int = 0, min_num_passages: int = 0) -> List[ProcessedSample]:
    """
    Process MS MARCO data into a standardized format.
    
    Args:
        df: DataFrame containing MS MARCO data with 'passages' column
        min_total_chars: Minimum total characters across all passages
        min_num_passages: Minimum number of passages required
        
    Returns:
        List of ProcessedSample objects containing query and passage information
    """
    samples = []
    
    for _, row in df.iterrows():
        passages = row['passages']
        selected_pos_i = passages['is_selected'].tolist().index(1)
        
        # Create candidate passages dictionary
        candidate_passages = {
            f"passage {i}": passage 
            for i, passage in enumerate(passages['passage_text'])
        }
        
        sample = ProcessedSample(
            query=row['query'],
            selected_passage=f"passage {selected_pos_i}",
            candidate_passages=candidate_passages
        )
        
        samples.append(sample)
    
    return samples


def format_passages(passages: Dict[str, str]) -> str:
    """Format passages for display in prompts."""
    return "\n".join([f"{pid}: {passage}" for pid, passage in passages.items()])


def main():
    """Main execution function."""
    # Configuration
    data_splits = ['test', 'train']
    candidate_lens = [5, 7, 9]
    random.seed(42)
    
    for data_split in data_splits:
        for candidate_len in candidate_lens:
            print(f"\nProcessing {data_split} split with candidate length {candidate_len}")
            
            # Load and filter data
            df = pd.read_parquet(f'data_ms_marco/data/{data_split}-00000-of-00001.parquet', engine='pyarrow')
            df = df[
                (df['passages'].apply(lambda x: x['is_selected'].tolist().count(1) == 1)) &
                (df['passages'].apply(lambda x: len(x['is_selected']) == candidate_len)) &
                (df['passages'].apply(lambda x: sum(len(p) for p in x['passage_text']) <= 2600))
            ]
            
            # Process samples
            processed_samples = process_msmarco_data(df)
            processed_samples = random.sample(
                processed_samples,
                k=min(len(processed_samples), (10000 // (candidate_len - 1)))
            )
            print(f"Number of samples: {len(processed_samples)}")
            
            # Define prompt templates
            ranking_template = '''## Here is a query: {query} and candidate passages:
{formatted_passages}

Please think step by step according to the content of each passage and how well it supports or relates to the query. Rank all passages from most relevant to least relevant. Return the passage IDs in order, one per line (e.g.,
passage 1
passage 3
passage 2). You MUST rank all passages from the candidate list. You can not generate content that is not in the given candidate list.
    '''
            
            exclusion_template = '''## Here is a query: {query} and candidate passages:
{formatted_passages}

Please think step by step according to the content of each passage and how well it supports or relates to the query. Select the least likely passage from the candidate list. Only return the passage ID corresponding to the excluded passage (e.g., "passage 3"). You MUST choose one passage from the candidate list. You can not generate content that is not in the given candidate list.
    '''
            
            # Generate samples for different evaluation scenarios
            data_router_all = []
            data_router_cases = []
            
            for sample in processed_samples:
                # Generate ranking sample
                ranking_sample = {
                    'problem': ranking_template.format(
                        query=sample.query,
                        formatted_passages=format_passages(sample.candidate_passages)
                    ),
                    'gt_item': sample.selected_passage,
                    'candidate_items': list(sample.candidate_passages.values())
                }
                data_router_all.append(ranking_sample)
                
                # Generate exclusion samples
                multi_results = process_list(sample.candidate_passages, sample.selected_passage)
                for case_dict in multi_results:
                    exclusion_sample = {
                        'problem': exclusion_template.format(
                            query=sample.query,
                            formatted_passages=format_passages(case_dict)
                        ),
                        'gt_item': sample.selected_passage,
                        'candidate_items': list(case_dict.values())
                    }
                    data_router_cases.append(exclusion_sample)
            
            # Save results
            output_dir = Path('data')
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / f'passage_all_{data_split}_len{candidate_len}.json', 'w', encoding='utf-8') as f:
                json.dump(data_router_all, f, ensure_ascii=False, indent=4)
            
            with open(output_dir / f'passage_cases_{data_split}_len{candidate_len}.json', 'w', encoding='utf-8') as f:
                json.dump(data_router_cases, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()


