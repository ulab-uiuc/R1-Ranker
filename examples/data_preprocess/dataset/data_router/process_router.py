"""
LLM Router Dataset Processing Script

This script processes LLM router data to create training and evaluation samples
for LLM selection tasks. It handles data loading, formatting, and generates
various types of prompts for different evaluation scenarios.
"""

import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def process_list(original_list: list, element_to_remove: str) -> list:
    """Process a list to create variations with different numbers of candidates.
    
    Args:
        original_list: List of elements to process
        element_to_remove: Element that should be included in all variations
        
    Returns:
        List of lists, each containing a subset of elements with the selected element
        inserted at a random position
    """
    results = []
    working_list = original_list.copy()

    if element_to_remove not in working_list:
        return []

    working_list.remove(element_to_remove)

    for k in range(1, len(working_list) + 1):
        temp_list = working_list.copy()
        random.shuffle(temp_list)
        top_k = temp_list[:k]
        insert_position = random.randint(0, k)
        top_k.insert(insert_position, element_to_remove)
        results.append(top_k)

    return results

def main():
    """Main execution function."""
    # Configuration
    input_csv = "data_router/data/router_data.csv"
    scenarios = ["Performance First", "Balance", "Cost First"]
    random.seed(42)
    
    for scenario in scenarios:
        # Load and process data
        data_df = pd.read_csv(input_csv)
        data_df = data_df[data_df['task_id'] != 'multi_news']
        effect_list = np.array(data_df['effect'].tolist())
        cost_list = np.array(data_df['cost'].tolist())
        query_list = data_df['query'].tolist()
        query = query_list[::10]

        # Calculate effect scores based on scenario
        if scenario == "Performance First":
            effect_list = 1.0 * effect_list - 0.0 * cost_list
        elif scenario == "Balance":
            effect_list = 0.5 * effect_list - 0.5 * cost_list
        else:
            effect_list = 0.2 * effect_list - 0.8 * cost_list

        llm_names = [
            "LLaMA-3 (8b)",
            "Mixtral-8x7B",
            "NousResearch (34b)",
            "LLaMA-2 (7b)",
            "Mistral-7b",
            "LLaMA-3 (70b)",
            "LLaMA-3-Turbo (8b)",
            "LLaMA-3-Turbo (70b)",
            "Llama-3.1-Turbo (70b)",
            "Qwen-1.5 (72b)"
        ]
        
        effect_re = effect_list.reshape(-1, len(llm_names))
        label = np.argmax(effect_re, axis=1)
        gt_llm_list = [llm_names[item] for item in label]

        template = """# The LLM names and their discriptions are: 
    # LLaMA-3 (8b): Each token price is 0.2. Handles simple Hybrid QA, basic reasoning, short reading comprehension, and concise summaries. 
    # Mixtral-8x7B: Each token price is 0.6. Excels at instruction-based tasks, step-by-step reasoning, solid reading comprehension, and structured summaries. 
    # NousResearch (34b): Each token price is 0.9. Ideal for complex research queries, multi-step reasoning, deep reading comprehension, and thorough summaries. 
    # LLaMA-2 (7b): Each token price is 0.2. Lightweight chat model for straightforward Hybrid QA, moderate reasoning, short reading comprehension, and concise summaries.
    # Mistral-7b: Each token price is 0.2. Fast for moderate Hybrid QA, quick reasoning, short reading comprehension, and brief summaries. 
    # LLaMA-3 (70b): Each token price is 0.9. High-capacity model for advanced Hybrid QA, deep reasoning, detailed reading comprehension, and extensive summaries. 
    # LLaMA-3-Turbo (8b): Each token price is 0.2. Balanced performance for moderate Hybrid QA, reasonable reasoning, clear reading comprehension, and concise summaries.
    # LLaMA-3-Turbo (70b): Each token price is 0.9. Powerful model for advanced Hybrid QA, strong reasoning, in-depth reading comprehension, and high-quality summaries.
    # Llama-3.1-Turbo (70b): Each token price is 0.9. Instruction-focused, providing thorough reasoning, structured reading comprehension, and well-organized summaries.
    # Qwen-1.5 (72b): Each token price is 0.9. Versatile for challenging Hybrid QA, nuanced reasoning, extended reading comprehension, and comprehensive summaries. 
    ## Here is a query: {query} and LLM condidates: {llm_candidates}. Please think step by step according to the description of each query and LLM, and evaluate from the perspectives of performance in answering the query and token price, and select the least likely LLM from the LLM condidates. Only return the LLM name corresponding to the LLM. You MUST choose one LLM name from LLM condidates. You can not generate content that are not in the given LLM condidates.
    """

        # Generate samples
        data_router_all = []
        data_router_cases = []
        
        for inter in range(len(query)):
            query_ = query[inter]
            gt_llm = gt_llm_list[inter]
            prompt = template.format(query=query_, llm_candidates=llm_names)
            
            router_sample = {
                'problem': prompt,
                'gt_item': gt_llm,
                'candidate_items': llm_names
            }
            data_router_all.append(router_sample)

            multi_results = process_list(llm_names, gt_llm)
            for inter in multi_results:
                router_case = {
                    'problem': template.format(query=query_, llm_candidates=inter),
                    'gt_item': gt_llm,
                    'candidate_items': inter
                }
                data_router_cases.append(router_case)

        # Perform train/test split
        train_all, test_all = train_test_split(data_router_all, test_size=0.2, random_state=42)
        train_cases, test_cases = train_test_split(data_router_cases, test_size=0.2, random_state=42)

        # Save results
        output_dir = Path('data')
        output_dir.mkdir(exist_ok=True)
        
        # Save all data
        # with open(output_dir / f'router_all_{scenario}.json', 'w', encoding='utf-8') as f:
        #     json.dump(data_router_all, f, ensure_ascii=False, indent=4)

        # with open(output_dir / f'router_cases_{scenario}.json', 'w', encoding='utf-8') as f:
        #     json.dump(data_router_cases, f, ensure_ascii=False, indent=4)

        # Save train/test splits
        with open(output_dir / f'router_all_train_{scenario}.json', 'w', encoding='utf-8') as f:
            json.dump(train_all, f, ensure_ascii=False, indent=4)
        
        with open(output_dir / f'router_all_test_{scenario}.json', 'w', encoding='utf-8') as f:
            json.dump(test_all, f, ensure_ascii=False, indent=4)
        
        with open(output_dir / f'router_cases_train_{scenario}.json', 'w', encoding='utf-8') as f:
            json.dump(train_cases, f, ensure_ascii=False, indent=4)
        
        with open(output_dir / f'router_cases_test_{scenario}.json', 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()