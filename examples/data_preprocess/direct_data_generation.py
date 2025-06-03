import re
import os
from datasets import Dataset, load_dataset, concatenate_datasets
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp):
    """Generate prefix for the prompt."""
    query = dp['problem']
    prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n {query} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/rec')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=6000)
    parser.add_argument('--test_size', type=int, default=39)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')

    args = parser.parse_args()

    data_source = 'direct_ranking'

    # Load dataset from HuggingFace
    dataset = load_dataset("ulab-ai/Ranking-bench", "direct")
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example)
            
            solution = {
                'candidate_text': example['candidates'],
                'gt': example['gt']
            }
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "ranking",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'task_name': example['task_name']
                }
            }
            return data

        return process_fn

    # Process datasets
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Save datasets
    os.makedirs('data/direct_ranking', exist_ok=True)
    train_dataset.to_parquet(os.path.join('data/direct_ranking', 'train.parquet'))
    test_dataset.to_parquet(os.path.join('data/direct_ranking', 'test.parquet'))

    # Handle HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)