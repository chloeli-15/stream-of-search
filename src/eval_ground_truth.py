import os
import json
import random
import argparse
from datetime import datetime
import wandb
from tqdm import tqdm

import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM
from datasets import load_dataset, DatasetDict, Dataset

from countdown_utils import *
from countdown_bfs import bfs
from countdown_dfs import dfs
import sys
import pandas as pd
from src.eval_custom import log_results_to_wandb
from src.result_parsers.countdown_trajectories import evaluate_countdown_trajectory

# sys path append the parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finetune.run_adapter_model import load_model, generate, generate_batch
from result_parsers.countdown_trajectories import evaluate_countdown_trajectories

from typing import List, Tuple

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
                                                         
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="MelinaLaimon/stream-of-search")
parser.add_argument("--messages_field", type=str, default="messages")  # Add this argument
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--nums", type=int, default=400)
parser.add_argument("--upload_results", type=bool, default=False)  # Add this argument
parser.add_argument("--wandb_project", type=str, default="stream-of-search")  # Add this argument
parser.add_argument("--wandb_entity", type=str, default=None)  # Add this argument


def log_results_to_wandb(eval_results, model_name, dataset_name, split, results_file):
    """Centralized function to handle all wandb logging tasks"""
    # Log the overall metrics
    if len(eval_results) >= 2 and "mean" in eval_results[1]:
        wandb.log({
            "model": model_name,
            "dataset_name": dataset_name,
            "split": split,
            "success_rate_mean": eval_results[1]["mean"],
            "success_rate_best_of_n": eval_results[1]["best_of_n"],
            "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        })
    
    # Create a table for example trajectories
    trajectory_data = []
    num_trajectories = min(10, len(eval_results) - 2)
    
    if num_trajectories > 0:
        for i in range(num_trajectories):
            trajectory_idx = i + 2  # Skip hyperparams and metrics
            if trajectory_idx < len(eval_results):
                trajectory = eval_results[trajectory_idx]
                # Extract the key information
                trajectory_data.append({
                    "prompt": trajectory.get("prompt", "N/A"),  
                    "completion": trajectory.get("completion", "N/A"),
                    "solved": trajectory.get("parsed_results", {}).get("solved", False),
                    "target": trajectory.get("parsed_results", {}).get("target", "N/A"),
                    "initial_numbers": str(trajectory.get("parsed_results", {}).get("initial_numbers", [])),
                    "operations": str(trajectory.get("parsed_results", {}).get("operations", [])),
                    "final_value": trajectory.get("parsed_results", {}).get("final_value", "N/A")
                })
        
        # Create and log the table
        df = pd.DataFrame.from_dict(trajectory_data)
        trajectory_table = wandb.Table(dataframe=df)
        wandb.log({"example_trajectories": trajectory_table})
        
    # Save the full results as an artifact
    artifact = wandb.Artifact(
        name=f"{model_name.replace('/', '_')}_{dataset_name.replace('/', '_')}_{split}_samples", 
        type="results"
    )
    artifact.add_file(results_file)
    wandb.log_artifact(artifact)
    
    # # Log the first few completions as HTML for easier inspection
    # for i in range(min(3, len(eval_results) - 2)):
    #     trajectory_idx = i + 2  # Skip hyperparams and metrics
    #     if trajectory_idx < len(eval_results):
    #         trajectory = eval_results[trajectory_idx]
    #         # Create HTML with syntax highlighting
    #         html_content = f"""
    #         <h3>Problem {i+1}</h3>
    #         <div style="margin-bottom: 10px;"><strong>Prompt:</strong> {trajectory.get('prompt', 'N/A')}</div>
    #         <div style="margin-bottom: 10px;"><strong>Solved:</strong> {trajectory.get('parsed_results', {}).get('solved', False)}</div>
    #         <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto; max-height: 400px;">
    #         {trajectory.get('completion', 'N/A')}
    #         </pre>
    #         """
    #         wandb.log({f"example_{i+1}": wandb.Html(html_content)})

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists("./results/ground_truths"):
        os.makedirs("./results/ground_truths")
    # Initialize wandb if upload_results is True
    
    messages_field_keys = {
        "optimal": "messages_optimal",
        "search": "messages_sos",
        "search-react": "messages_sos_react",
        "deepseek_r1_distill_llama_70b": "messages_deepseek_r1_distill_llama_70b",
        "deepseek": "messages_deepseek",
    }
    
    import copy
    data_all = data = load_dataset(
        "MelinaLaimon/stream-of-search", 
        revision="4dc3d9dd567dc6629597f7bd0cf332e964d575dd", 
        split='train').select(range(0, args.nums))
        
    for model_name, message_field in messages_field_keys.items():
        data = copy.copy(data_all)
        
        data = data.map(lambda x: { # type: ignore
            'completion': 
                # {'role': 'system', 'content': SYSTEM_PROMPT},
                x[message_field][1]['content'],
            # 'answer': extract_hash_answer(x['answer'])
        })
        data = data.map(lambda x: { 
            'tokens': len(enc.encode(x['completion'])
        )})
        
        # data_rejection_sampled = data.filter(lambda x: x['tokens'] < 8192)
            
        if args.upload_results:
            run_name = model_name-datetime.now().strftime("%Y%m%d-%H%M%S")
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args)
            )
            
        torch.manual_seed(args.seed)

        data = data.add_column("parsed_results", list(map(evaluate_countdown_trajectory, data)))
        results = sum([d['solved'] for d in data['parsed_results']]) / len(data['parsed_results'])
        
        
        # Save results locally
        save_path = os.path.join("./results/ground_truths/", f'{model_name}')
      
            
        timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_file = f"{save_path}_{timenow}.json"       
        
        eval_results = [
            {
                "hyperparams": {
                    "experiment_name": f"{timenow}-gt-eval-{model_name}",
                    "model": model_name,
                    "dataset": args.dataset_name,
                    "messages_field": message_field,
                    "num": args.nums,
                }
            },
            {
                "success_rate": results,
                "percentage_solved_at_8192": len(data.filter(lambda x: x['tokens'] < 8192 and x['parsed_results']['solved']==True))/len(data),
                "percentage_solved_at_10000": len(data.filter(lambda x: x['tokens'] < 10000 and x['parsed_results']['solved']==True))/len(data),            
            }
        ]
        print(f"Model: {model_name}, Success Rate: {results:.2%}")
        with open(results_file, "w") as f:
            json.dump(eval_results, f, indent=4)
        
        # Then in your main code, replace the wandb logging section with:
        if args.upload_results:
            log_results_to_wandb(
                eval_results=eval_results,
                model_name=model_name,
                dataset_name=args.dataset_name,
                split='train',
                results_file=results_file
            )
                    
        # Finish the wandb run
        if args.upload_results:
            wandb.finish()