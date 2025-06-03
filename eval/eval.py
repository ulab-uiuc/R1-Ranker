import argparse
import importlib.util
import os
import sys

def load_module_from_file(file_path):
    """Load a Python module from a file path."""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def get_task_from_dataset(dataset):
    """Infer the task from the dataset name."""
    if dataset.startswith('Passage-'):
        return 'passage'
    elif dataset.startswith('Rec-'):
        return 'rec'
    elif dataset.startswith('Router-'):
        return 'router'
    else:
        raise ValueError(f"Invalid dataset name: {dataset}. Dataset must start with 'Passage-', 'Rec-', or 'Router-'")

def main():
    parser = argparse.ArgumentParser(description='Evaluation script for different tasks')
    
    # Dataset argument
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset to evaluate on (e.g., Passage-5, Rec-Movie, Router-Balance)')
    
    # GPU ID argument
    parser.add_argument('--gpu_id', type=str, default='0',
                      help='GPU ID to use (default: 0)')
    
    # Model path argument
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint to use for evaluation')
    
    args = parser.parse_args()
    
    # Infer task from dataset name
    task = get_task_from_dataset(args.dataset)
    
    # Map task to script file
    script_map = {
        'passage': 'eval_passage.py',
        'rec': 'eval_rec.py',
        'router': 'eval_router.py'
    }
    
    # Validate dataset based on task
    if task == 'passage':
        valid_datasets = ['Passage-5', 'Passage-7', 'Passage-9']
        if args.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset for passage task. Must be one of {valid_datasets}")
    elif task == 'rec':
        valid_datasets = ['Rec-Movie', 'Rec-Game', 'Rec-Music']
        if args.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset for rec task. Must be one of {valid_datasets}")
    elif task == 'router':
        valid_datasets = ['Router-Balance', 'Router-Cost', 'Router-Performance']
        if args.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset for router task. Must be one of {valid_datasets}")
    
    # Get the script path - ensure we use the directory where eval.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_map[task])
    
    # Verify the script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Evaluation script not found: {script_path}")
    
    # Load and run the appropriate module
    module = load_module_from_file(script_path)
    module.main(args.dataset, args.gpu_id, args.model_path)

if __name__ == "__main__":
    main() 