import os
import json
import random
import argparse
from datetime import datetime

from tqdm import tqdm

import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM
from datasets import load_dataset, DatasetDict, Dataset

from countdown_utils import *
from countdown_bfs import bfs
from countdown_dfs import dfs
from finetune.run_adapter_model import load_model, generate, generate_batch

from typing import List, Tuple

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--ckpt", type=str, help="path to checkpoint")
parser.add_argument("--adapter", type=str, help="path to adapter")
parser.add_argument("-n", "--num",type=int, default=10)
parser.add_argument("-o", "--offset",type=int, default=0)
parser.add_argument("--data_dir", type=str, default="data/")
parser.add_argument("-d", "--data",type=str, default="val_b3_t100_n100000_random.json")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--ctx", type=int, default=4096)
parser.add_argument("--gens", type=int, default=1)

def eval_ll(model, tokenizer, data, batch_size=128, context_len=4096, temperature=0.0, n=1):
    """
    Evaluate the model on the data using a sliding window so that the context length is not exceeded
    """
    output_texts_concat = []

    for i, data_batch in tqdm(enumerate(data.iter(batch_size=batch_size)), total=len(data)//batch_size):   
        # tokenize and generate data_batch['test_prompt']. Input is a list of dicts with role
        chat_inputs = tokenizer.apply_chat_template(data_batch["test_prompt"], return_tensors="pt", padding=True, truncation=True, max_length=context_len, return_length=True, tokenize=False)
        # generate
        outputs = generate_batch(model, tokenizer, chat_inputs, max_new_tokens=context_len, temperature=temperature)
        output_texts_concat.extend(outputs)

    return output_texts_concat


def evaluate_trajectory(target: int, numbers: List[int], operations: List[str]) -> bool:
    """
    Evaluates whether the sequence of operations correctly reaches the target value.
    
    :param target: The desired target value.
    :param numbers: The initial list of numbers available for operations.
    :param operations: A list of arithmetic operations applied sequentially.
    :return: True if the operations correctly reach the target, False otherwise.
    """
    available_numbers = set(numbers)
    
    for op in operations:
        try:
            left, operator, right, result = parse_operation(op)
            
            if left not in available_numbers or right not in available_numbers:
                return False  # Invalid step, using unavailable numbers
            
            computed_result = apply_operation(left, right, operator)
            if computed_result != result:
                return False  # Computed result doesn't match stated result
            
            available_numbers.remove(left)
            available_numbers.remove(right)
            available_numbers.add(result)
        except:
            return False  # Any parsing or calculation error leads to failure
    
    return target in available_numbers

def parse_operation(operation: str) -> Tuple[int, str, int, int]:
    """Parses an operation of the form 'a+b=c' or 'a-b=c' etc."""
    for op in ['+', '-', '*', '/']:
        if op in operation:
            left, right_result = operation.split(op)
            right, result = right_result.split('=')
            return int(left.strip()), op, int(right.strip()), int(float(result.strip()))
    raise ValueError("Invalid operation format")

def apply_operation(left: int, right: int, operator: str) -> int:
    """Applies an arithmetic operation to two numbers."""
    if operator == '+':
        return left + right
    elif operator == '-':
        return left - right
    elif operator == '*':
        return left * right
    elif operator == '/':
        if right == 0:
            raise ZeroDivisionError("Division by zero")
        return left // right  # Integer division assumed
    else:
        raise ValueError("Unknown operator")
    
def extract_problem(text):
    """Extract the target number and initial numbers from the problem statement."""
    match = re.search(r"Make (\d+) with the numbers \[(\d+(?:,\s*\d+)*)\]", text)
    if match:
        target = int(match.group(1))
        numbers = list(map(int, match.group(2).split(',')))
        return target, numbers
    return None, None

def extract_operations(text):
    """Extract the sequence of operations performed by the LLM."""
    # pattern_operations = r"Exploring Operation: (\d+[+\-*/]\d+=\d+(?:\.\d+)?)"
    pattern_operations = r"Exploring Operation: (\d+\s*[\+\-\*/]\s*\d+\s*=\s*\d+(?:\.\d+)?)"
    return re.findall(pattern_operations, text)

args = parser.parse_args()
torch.manual_seed(args.seed)

model, tokenizer = load_model(args.adapter, args.ckpt)

model.eval()
model.half().cuda()

tokenizer.pad_token = tokenizer.eos_token

data_file = os.path.join(args.data_dir, args.data)

data_all = load_dataset("chloeli/stream-of-search-countdown-10k")

for split in data_all.keys():
    results_all_trials = {}
    for trial in range(args.gens):
        data = data_all[split].select(range(args.num))
        
        predictions = []
        pred_ratings = []
        pred_reasons = []
        tokenizer.padding_side = "left"

        data = data.map(lambda x: { # type: ignore
            'test_prompt': [
                # {'role': 'system', 'content': SYSTEM_PROMPT},
                x['messages']["role"=="user"]
            ],
            # 'answer': extract_hash_answer(x['answer'])
        })

        completions = eval_ll(model, tokenizer, data, batch_size=args.batch_size, context_len=args.ctx, temperature=args.temperature, n=args.gens)
        results = []
        # parse into list of dictionaries
        for i in range(len(data['test_prompt'])):
            results.append({
                'prompt': data['test_prompt'][i][0]['content'],
                'completion': completions[i]
            })
        
        for i in tqdm(range(len(results))):    
            problem_text = results[i].get("prompt", "")
            solution_text = results[i].get("completion", "")
            
            target, numbers = extract_problem(problem_text)
            if target is None: raise ValueError(f"Failed to extract problem from: {problem_text}")

            operations = extract_operations(solution_text)

            print(f"\nProblem: Target={target}, Numbers={numbers}")
            print(f"Operations: {operations}")
            results[i]["success"] = evaluate_trajectory(target, numbers, operations)
            print(f"Success: {results[i]['success']}")

        results_all_trials[trial] = results

    # take the best results from 3 trials
    # turn into np.array
    res_dict = {}
    # take average across results_all_trials[i]
    # results_all_trials[trial][i]['success'] is the success of the i-th problem in trial 
    # turn into np.array of shape (num_problems, args.gens)
    full_results = []
    for q in range(args.num):
        full_results.append([results_all_trials[trial][q]['success'] for trial in results_all_trials.keys()])
    full_results = np.array(full_results)            

    results_best_of_n = list(np.max(full_results, axis=1))
    results_mean = list(np.mean(full_results, axis=1))
    full_results = full_results.tolist()
    
    # objects of type bool are not serializable
    # convert to list of ints
    res_dict["results_best_of_n"] = [int(x) for x in results_all_trials]
    res_dict["results_mean"] = [int(x) for x in results_mean]
    res_dict["full_results"] = [[int(x) for x in y] for y in full_results]
    res_dict["results_all_trials"] = results_all_trials
    
    save_path = os.path.join("results", f'{args.data.replace("/", "_")}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save results
    timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(save_path, f"{args.adapter.split('/')[-1]}_{split}_{args.num}_{args.offset}_{timenow}")
    with open(results_file, "w") as f:
        json.dump(res_dict, f, indent=4)

    # # rate outputs
    # true_rating = []
    # for i in range(len(predictions)):
    #     # rating, reason = metric_fn(predictions[i].split(tokenizer.bos_token)[1], mode="sft")
    #     # 'system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nuser\nMake 96 with the numbers [58, 84, 48, 62] using standard arithmetic operations.\nassistant\nCurrent State: 96:[58,'
        
    #     tr, _ = metric_fn(f"{data[i]['search_path']}", mode="sft")
    #     pred_ratings.append(rating)
    #     true_rating.append(tr)
    #     pred_reasons.append(reason)

    # # get max rating for each sample with its index
    # pred_ratings = np.array(pred_ratings)

    # # print results
    # print("Results Summary:")
    # print(f"Average rating: {np.mean(pred_ratings)}")
    # print(f"Average true rating: {np.mean(true_rating)}")
    # print(f"Accuracy: {np.mean([r > 0 for r in pred_ratings])}")
    # print(f"True Accuracy: {np.mean([r > 0 for r in true_rating])}")

    # save_path = os.path.join("results", f'{args.data.replace("/", "_")}')
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # # save results
    # timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
    # results_file = os.path.join(save_path, f"{args.adapter.split('/')[-1]}_{args.num}_{args.offset}_{timenow}")
    # with open(results_file, "w") as f:
    #     json.dump({"trajectories": predictions, "ratings": pred_ratings.tolist(), "reasons": pred_reasons}, f, indent=4)
