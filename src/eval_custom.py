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

# sys path append the parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finetune.run_adapter_model import load_model, generate, generate_batch
from result_parsers.countdown_trajectories import evaluate_countdown_trajectories

from typing import List, Tuple

                                                         
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--ckpt", type=str, help="path to checkpoint")
parser.add_argument("--adapter", type=str, help="path to adapter")
parser.add_argument("-n", "--num",type=int, default=10)
parser.add_argument("--dataset_name", type=str, default="MelinaLaimon/stream-of-search")
parser.add_argument("-d", "--data",type=str, default="val_b3_t100_n100000_random.json")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--ctx", type=int, default=4096)
parser.add_argument("--gens", type=int, default=1)
# parser.add_argument("--split_range", type=str, default="0:6")
parser.add_argument("--chat_template", type=bool, default="True")
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--messages_field", type=str, default="messages")  # Add this argument
parser.add_argument("--upload_results", type=bool, default=False)  # Add this argument
parser.add_argument("--wandb_project", type=str, default="stream-of-search")  # Add this argument
parser.add_argument("--wandb_entity", type=str, default=None)  # Add this argument



def eval_ll(model, tokenizer, data, batch_size=128, context_len=4096, temperature=0.0, n=1):
    """
    Evaluate the model on the data using a sliding window so that the context length is not exceeded
    """
    output_texts_concat = []

    for i, data_batch in tqdm(enumerate(data.iter(batch_size=batch_size)), total=len(data)//batch_size):   
        # tokenize and generate data_batch['test_prompt']. Input is a list of dicts with role
        # if args.chat_template:
        chat_inputs = tokenizer.apply_chat_template(data_batch["test_prompt"], return_tensors="pt", padding=True, truncation=True, max_length=context_len, return_length=True, tokenize=False)
        # else:
        #     chat_inputs = data_batch["test_prompt"]["content"] # if no chat template 
        outputs = generate_batch(model, tokenizer, chat_inputs, max_new_tokens=context_len, temperature=temperature)
        output_texts_concat.extend(outputs)
            
    return output_texts_concat

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


def custom_eval(args=None):
    """Entry point that can be called programmatically or via command line"""
    if args is None:
        args = parser.parse_args()
    
    timenow = datetime.now().strftime('%Y%m%d-%H%M%S')
    # Initialize wandb if upload_results is True
    if args.experiment_name is None:
        args.experiment_name = f"{timenow}-custom_eval-{args.adapter.split('/')[-1]}"
        
    if args.upload_results:
        run_name = args.experiment_name if args.experiment_name else datetime.now().strftime("%Y%m%d-%H%M%S")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )
        
    torch.manual_seed(args.seed)
    
    
    if "," in args.adapter: # if many
        adapters = args.adapter.split(",")
        adapters = [adapter.strip() for adapter in adapters] 
    else:
        adapters = [args.adapter] 

    for adapter in adapters:
        model, tokenizer = load_model(adapter, args.ckpt)

        model.eval()
        model.cuda()

        tokenizer.pad_token = tokenizer.eos_token

        # data_file = os.path.join(args.data_dir, args.data)
        data_all = load_dataset(args.dataset_name)

        # keys = list(data_all.keys())[int(args.split_range.split(":")[0]):int(args.split_range.split(":")[1])]
        # only the keys that have the word 'search' in them
        # keys = [key for key in data_all.keys() if 'search' in key]
        keys = data_all.keys()
        
        # if it is the regular split, reverse the order so we get test results first
        if keys == ['train', 'test']:
            keys = ["test"]
        elif keys == ["countdown_3num", "countdown_5num"]:
            pass        
            
        for split in keys: # keys:
            results_all_trials = []
            for trial in range(args.gens):
                data = data_all[split].select(range(args.num))
                
                predictions = []
                pred_ratings = []
                pred_reasons = []
                tokenizer.padding_side = "left"

                if split in ["train", "test"]:
                    data = data.map(lambda x: { # type: ignore
                        'test_prompt': [
                            # {'role': 'system', 'content': SYSTEM_PROMPT},
                            x[args.messages_field]["role"=="user"]
                        ],
                        # 'answer': extract_hash_answer(x['answer'])
                    })
                elif split in ["countdown_3num", "countdown_5num"]:
                    data = data.map(lambda x: { # type: ignore
                        'test_prompt': [{"content": x['user_prompt'], "role": "user"}],
                    })
                results = []
                completions = eval_ll(model, tokenizer, data, batch_size=args.batch_size, context_len=args.ctx, temperature=args.temperature, n=args.gens)
                # parse into list of dictionaries
                for i in range(len(data['test_prompt'])):
                    results.append({
                        'nums': data['nums'][i],
                        'target': data['target'][i],
                        'solution': data['solution'][i],
                        'prompt': data['test_prompt'][i][0]['content'],
                        'completion': completions[i]
                    })
                results_all_trials.append(results)
                
            eval_results = evaluate_countdown_trajectories(results_all_trials)
            eval_results.insert(0, {"hyperparams": vars(args)})
            
            # Save results locally
            model_name = args.adapter.split("/")[-1]
            save_path = os.path.join("results/", f'{model_name}')
                
            timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
            results_file = f"{save_path}/{split}_{args.num}_{timenow}.json"           
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(results_file, "w") as f:
                json.dump(eval_results, f, indent=4)
            
            # Then in your main code, replace the wandb logging section with:
            if args.upload_results:
                log_results_to_wandb(
                    eval_results=eval_results,
                    model_name=model_name,
                    dataset_name=args.dataset_name,
                    split=split,
                    results_file=results_file
                )
                
    # Finish the wandb run
    if args.upload_results:
        wandb.finish()
        
    # run visualization script
    from results.visualize import visualize_results
    visualize_results()
    
if __name__ == "__main__":
    custom_eval()