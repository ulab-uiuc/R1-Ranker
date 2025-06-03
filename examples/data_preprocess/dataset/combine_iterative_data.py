import json
import os
import argparse
from typing import Dict, List

def load_json_file(file_path: str) -> dict:
    """Load a JSON file and return its contents."""
    with open(file_path, 'r') as f:
        return json.load(f)

def process_rec_data(data: List[dict]) -> List[dict]:
    """Process recommendation data by renaming fields."""
    processed_data = []
    for item in data:
        processed_item = item.copy()
        if "gt_item" in processed_item:
            processed_item["gt"] = processed_item.pop("gt_item")
        if "candidate_items" in processed_item:
            processed_item["candidates"] = processed_item.pop("candidate_items")
        # Ensure problem field exists
        if "problem" not in processed_item:
            processed_item["problem"] = ""
        processed_data.append(processed_item)
    return processed_data

def process_router_data(data: List[dict]) -> List[dict]:
    """Process router data by renaming fields."""
    processed_data = []
    for item in data:
        processed_item = item.copy()
        if "gt_llm" in processed_item:
            processed_item["gt"] = processed_item.pop("gt_llm")
        if "candidate_text" in processed_item:
            processed_item["candidates"] = processed_item.pop("candidate_text")
        # Ensure problem field exists
        if "problem" not in processed_item:
            processed_item["problem"] = ""
        processed_data.append(processed_item)
    return processed_data

def process_passage_data(data: List[dict]) -> List[dict]:
    """Process passage data by renaming fields."""
    processed_data = []
    for item in data:
        processed_item = item.copy()
        if "gt_passage" in processed_item:
            processed_item["gt"] = processed_item.pop("gt_passage")
        if "candidate_passages" in processed_item:
            processed_item["candidates"] = processed_item.pop("candidate_passages")
        # Ensure problem field exists
        if "problem" not in processed_item:
            processed_item["problem"] = ""
        processed_data.append(processed_item)
    return processed_data

def process_data(data: List[dict], data_type: str) -> List[dict]:
    """Process data based on its type."""
    if data_type.startswith("Rec-"):
        return process_rec_data(data)
    elif data_type.startswith("Router-"):
        return process_router_data(data)
    elif data_type.startswith("Passage-"):
        return process_passage_data(data)
    return data

def combine_json_files() -> dict:
    """
    Combine JSON files from multiple directories into a structured format.
    
    Returns:
        Combined JSON structure with train/test splits and specific keys
    """
    base_path = "/data/taofeng2/tiny_rec/rank_dataset"
    
    # Define file paths and their corresponding keys
    file_mapping = {
        "train": {
            "Rec-Movie": f"{base_path}/data_rec/data/movie_cases_train.json",
            "Rec-Music": f"{base_path}/data_rec/data/music_cases_train.json",
            "Rec-Game": f"{base_path}/data_rec/data/game_cases_train.json",
            "Router-Performance": f"{base_path}/data_router/data_split/router_cases_all_Performance First_train.json",
            "Router-Balance": f"{base_path}/data_router/data_split/router_cases_all_Balance_train.json",
            "Router-Cost": f"{base_path}/data_router/data_split/router_cases_all_Cost First_train.json",
            "Passage-5": f"{base_path}/data_ms_marco/data/passage_cases_train_5_candidate.json",
            "Passage-7": f"{base_path}/data_ms_marco/data/passage_cases_train_7_candidate.json",
            "Passage-9": f"{base_path}/data_ms_marco/data/passage_cases_train_9_candidate.json"
        },
        "test": {
            "Rec-Movie": f"{base_path}/data_rec/data/movie_cases_test.json",
            "Rec-Music": f"{base_path}/data_rec/data/music_cases_test.json",
            "Rec-Game": f"{base_path}/data_rec/data/game_cases_test.json",
            "Router-Performance": f"{base_path}/data_router/data_split/router_cases_all_Performance First_test.json",
            "Router-Balance": f"{base_path}/data_router/data_split/router_cases_all_Balance_test.json",
            "Router-Cost": f"{base_path}/data_router/data_split/router_cases_all_Cost First_test.json",
            "Passage-5": f"{base_path}/data_ms_marco/data/passage_cases_test_5_candidate.json",
            "Passage-7": f"{base_path}/data_ms_marco/data/passage_cases_test_7_candidate.json",
            "Passage-9": f"{base_path}/data_ms_marco/data/passage_cases_test_9_candidate.json"
        }
    }
    
    combined_data = {
        "train": {},
        "test": {}
    }
    
    # Load and combine the files
    for split in ["train", "test"]:
        for key, file_path in file_mapping[split].items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            try:
                data = load_json_file(file_path)
                # Process the data based on its type
                processed_data = process_data(data, key)
                combined_data[split][key] = processed_data
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                raise
    
    return combined_data

def save_json(data: dict, file_path: str):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Combine JSON files from multiple directories')
    parser.add_argument('--output_dir', required=True, help='Output directory for JSON files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Combine the JSON files
    combined_data = combine_json_files()
    
    # Save the overall combined data
    overall_path = os.path.join(args.output_dir, "iterative_raw_data.json")
    save_json(combined_data, overall_path)
    print(f"Overall combined data saved to {overall_path}")
    
    # Save train data separately
    train_path = os.path.join(args.output_dir, "iterative_train_data.json")
    save_json(combined_data["train"], train_path)
    print(f"Training data saved to {train_path}")
    
    # Save test data separately
    test_path = os.path.join(args.output_dir, "iterative_test_data.json")
    save_json(combined_data["test"], test_path)
    print(f"Testing data saved to {test_path}")

if __name__ == "__main__":
    main()
