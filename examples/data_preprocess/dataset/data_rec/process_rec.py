"""
Movie Recommendation Dataset Processing Script

This script processes movie recommendation data to create training and evaluation samples
for sequential recommendation tasks. It handles data loading, formatting, and generates
various types of prompts for different evaluation scenarios.
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple, Any
import json
from sklearn.model_selection import train_test_split
import copy

def load_data(data_source: str) -> Tuple[Dict[int, str], pd.DataFrame]:
    """Load item and interaction data from movie dataset files.
    
    Args:
        data_source: One of 'ml-1m', 'Amazon_CDs_and_Vinyl', or 'Amazon_Video_Games'
    
    Returns:
        Tuple containing:
        - Dictionary mapping item IDs to movie titles
        - DataFrame containing user interactions with movies
    """
    # Load item data
    items = {}
    with open(f'data_rec/{data_source}.item', 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                item_id = str(parts[0])
                title = parts[1]
                items[item_id] = title
    
    # Load interaction data
    interactions = []
    
    with open(f'data_rec/{data_source}.inter', 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                user_id = str(parts[0])
                item_id = str(parts[1])
                rating = float(parts[2])
                timestamp = int(parts[3])
                interactions.append((user_id, item_id, rating, timestamp))
    
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    df = df.sort_values(['user_id', 'timestamp'])
    
    return items, df

def create_sequential_samples(items: Dict[int, str], 
                           df: pd.DataFrame, 
                           num_samples: int = 657,
                           hist_len: int = 20,
                           num_candidates: int = 20) -> List[Dict]:
    """Create sequential recommendation samples."""
    # Get users with at least hist_len + 1 interactions
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= hist_len + 1].index.tolist()
    
    if len(valid_users) < num_samples:
        raise ValueError(f"Not enough users with sufficient interactions. Found {len(valid_users)} users, need {num_samples}")
    
    # Randomly select users
    selected_users = random.sample(valid_users, num_samples)
    samples = []
    all_items = set(items.keys())
    
    for user_id in selected_users:
        # Get user's interactions
        user_interactions = df[df['user_id'] == user_id].sort_values('timestamp')
        item_sequence = user_interactions['item_id'].tolist()
        
        # Get history and ground truth
        history = item_sequence[:hist_len]
        ground_truth = item_sequence[hist_len]
        
        # Sample negative candidates
        excluded_items = set(history + [ground_truth])
        available_items = list(all_items - excluded_items)
        negative_candidates = random.sample(available_items, num_candidates - 1)
        
        # Combine ground truth and negative candidates
        candidates = [ground_truth] + negative_candidates
        random.shuffle(candidates)  # Shuffle to avoid position bias
        
        # Convert IDs to names
        history_names = [items[item_id] for item_id in history]
        candidates_names = [items[item_id] for item_id in candidates]
        ground_truth_name = items[ground_truth]
        
        # Create sample
        sample = {
            'user_id': user_interactions['user_id'].iloc[0],
            'history': history_names,
            'candidates': candidates_names,
            'ground_truth': ground_truth_name,
        }
        samples.append(sample)
    
    return samples

def process_list(original_dict: Dict[str, str], selected_movie: str) -> List[Dict[str, str]]:
    """Process a list of movies to create different combinations."""
    results = []
    
    # Get the selected movie
    selected_id = next((k for k, v in original_dict.items() if v == selected_movie), None)
    if selected_id is None:
        return results
    
    # Create a working dictionary without the selected movie
    working_dict = original_dict.copy()
    del working_dict[selected_id]
    
    # Get list of ids and movies separately
    ids = list(working_dict.keys())
    movies = list(working_dict.values())
    
    # Loop through K from 1 to the maximum length of the dictionary
    for k in range(1, len(working_dict) + 1):
        # Create a temporary copy of the ids and movies
        temp_ids = ids.copy()
        temp_movies = movies.copy()
        
        # Shuffle both lists in the same order
        combined = list(zip(temp_ids, temp_movies))
        random.shuffle(combined)
        shuffled_ids, shuffled_movies = zip(*combined) if combined else ([], [])
        
        # Take the top K elements
        top_k_ids = shuffled_ids[:k]
        top_k_movies = shuffled_movies[:k]
        
        # Create a dictionary from the top K items
        top_k_dict = {id_val: movie for id_val, movie in zip(top_k_ids, top_k_movies)}
        
        # Add the selected movie
        insert_position = random.randint(0, k)
        result_dict = {}
        
        # Add items before insert position
        for i, id_val in enumerate(top_k_ids[:insert_position]):
            result_dict[id_val] = top_k_movies[i]
            
        # Add the selected movie
        result_dict[selected_id] = selected_movie
        
        # Add items after insert position
        for i, id_val in enumerate(top_k_ids[insert_position:]):
            result_dict[id_val] = top_k_movies[insert_position + i]
        
        # Add the result to results list
        results.append(result_dict)
    
    return results

def process_samples(samples: List[Dict], template: str) -> Tuple[List[Dict], List[Dict]]:
    """Process samples to create both all and cases versions."""
    data_all = []
    data_cases = []
    
    for sample in samples:
        # Format history as a numbered list
        history_formatted = '[' + ', '.join([f"'{i}. {movie}'" for i, movie in enumerate(sample['history'])]) + ']'
        
        # Format candidates as a numbered list
        candidates_formatted = '[' + ', '.join([f"'{i}. {movie}'" for i, movie in enumerate(sample['candidates'])]) + ']'
        
        # Create the all version
        all_sample = {
            'user_id': str(sample['user_id']),
            'problem': template.format(
                history=history_formatted,
                candidates=candidates_formatted
            ),
            'gt_item': sample['ground_truth'],
            'candidate_items': sample['candidates']
        }
        data_all.append(all_sample)
        
        # Create the cases version
        candidate_dict = {str(i): movie for i, movie in enumerate(sample['candidates'])}
        case_results = process_list(candidate_dict, sample['ground_truth'])
        
        for case_dict in case_results:
            # Format candidates for this case
            case_candidates_formatted = '[' + ', '.join([f"'{pid}. {movie}'" for pid, movie in case_dict.items()]) + ']'
            
            case_sample = {
                'user_id': str(sample['user_id']),
                'problem': template.format(
                    history=history_formatted,
                    candidates=case_candidates_formatted
                ),
                'gt_item': sample['ground_truth'],
                'candidate_items': list(case_dict.values())
            }
            data_cases.append(case_sample)
    
    return data_all, data_cases

def main():
    """Main execution function."""
    # Define data sources
    data_sources = ['ml-1m', 'Amazon_CDs_and_Vinyl', 'Amazon_Video_Games']
    
    for data_source in data_sources:
        try:
            # Load data
            items, df = load_data(data_source)
            
            # Create samples
            samples = create_sequential_samples(items, df)
            
            # Split into train and test
            train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)
            
            # Template for the problem
            template = '''I've watched the following movies in the past, in order:
{history}

Now there are 20 candidate movies that I might watch next:
{candidates}

Please select the one movie that is least likely to be my next watch, according to my watching history. Please think step by step.
You MUST choose exactly one movie from the given candidate list.
You can NOT generate or reference movies that are not in the given candidate list.
Return only the full name of the movie.'''
            
            # Process train samples
            train_all, train_cases = process_samples(train_samples, template)
            
            # Process test samples
            test_all, test_cases = process_samples(test_samples, template)
            
            # Save the processed data
            with open(f'data_rec/data/{data_source}_all_train.json', 'w', encoding='utf-8') as f:
                json.dump(train_all, f, ensure_ascii=False, indent=4)
            
            with open(f'data_rec/data/{data_source}_cases_train.json', 'w', encoding='utf-8') as f:
                json.dump(train_cases, f, ensure_ascii=False, indent=4)
            
            with open(f'data_rec/data/{data_source}_all_test.json', 'w', encoding='utf-8') as f:
                json.dump(test_all, f, ensure_ascii=False, indent=4)
            
            with open(f'data_rec/data/{data_source}_cases_test.json', 'w', encoding='utf-8') as f:
                json.dump(test_cases, f, ensure_ascii=False, indent=4)
            
            print(f"Created {len(train_all)} train all samples and {len(train_cases)} train case samples for {data_source}")
            print(f"Created {len(test_all)} test all samples and {len(test_cases)} test case samples for {data_source}")
            print(f"Each sample contains {len(samples[0]['history'])} historical movies and {len(samples[0]['candidates'])} candidate movies")
            
        except Exception as e:
            print(f"Error processing {data_source}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 