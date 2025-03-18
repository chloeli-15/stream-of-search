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
from result_parsers.countdown_trajectories import evaluate_countdown_trajectories

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
parser.add_argument("--split_range", type=str, default="0:6")
parser.add_argument("--chat_template", type=bool, default="True")


def eval_ll(model, tokenizer, data, batch_size=128, context_len=4096, temperature=0.0, n=1):
    """
    Evaluate the model on the data using a sliding window so that the context length is not exceeded
    """
    output_texts_concat = []

    for i, data_batch in tqdm(enumerate(data.iter(batch_size=batch_size)), total=len(data)//batch_size):   
        # tokenize and generate data_batch['test_prompt']. Input is a list of dicts with role
        if args.chat_template:
            chat_inputs = tokenizer.apply_chat_template(data_batch["test_prompt"], return_tensors="pt", padding=True, truncation=True, max_length=context_len, return_length=True, tokenize=False)
        else:
            chat_inputs = data_batch["test_prompt"]["content"] # if no chat template 
        outputs = generate_batch(model, tokenizer, chat_inputs, max_new_tokens=context_len, temperature=temperature)
        output_texts_concat.extend(outputs)
            
    return output_texts_concat


args = parser.parse_args()
torch.manual_seed(args.seed)

model, tokenizer = load_model(args.adapter, args.ckpt)

model.eval()
model.bfloat16().cuda()

tokenizer.pad_token = tokenizer.eos_token

data_file = os.path.join(args.data_dir, args.data)

data_all = load_dataset("chloeli/stream-of-search-countdown-10k")

keys = list(data_all.keys())[int(args.split_range.split(":")[0]):int(args.split_range.split(":")[1])]

for split in keys:
    results_all_trials = []
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
        results = []
        completions = eval_ll(model, tokenizer, data, batch_size=args.batch_size, context_len=args.ctx, temperature=args.temperature, n=args.gens)
        # parse into list of dictionaries
        for i in range(len(data['test_prompt'])):
            results.append({
                'prompt': data['test_prompt'][i][0]['content'],
                'completion': completions[i]
            })
        results_all_trials.append(results)
    
    eval_results = evaluate_countdown_trajectories(results_all_trials)
            
    save_path = os.path.join("results", f'{args.data.replace("/", "_")}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save results
    timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(save_path, f"{args.adapter.split('/')[-1]}_{split}_{args.num}_{args.offset}_{timenow}")
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=4)

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
