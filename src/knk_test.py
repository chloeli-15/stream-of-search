import sys
# sys.path.append('/cs/student/msc/ml/2024/ycheah/projects/sos/stream-of-search')
from finetune.run_adapter_model import load_model, generate, generate_batch, load_model_from_hub
from tqdm import tqdm
import datasets
import re
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def extract_parts(string):
    # Try the original pattern for folders with "countdown-"
    pattern1 = re.compile(r'(\d+\.\d+B).*countdown-(.+?)$')
    match = pattern1.search(string)
    
    if match:
        return [match.group(1), match.group(2)]
    
    string = string.lower()
    # Pattern for folders like "Qwen2.5-1.5B-Instruct"
    pattern2 = re.compile(r'qwen\d+\.\d+-(\d+\.\d+B)-instruct', re.IGNORECASE)
    match = pattern2.search(string)
    
    if match:
        return [match.group(1), "base_model"]
    
    # If no pattern matches, return default values
    return ["unknown", "unknown"]


def verify_solution_text(names, solution, solution_text):
    """
    Verifies if the solution_text correctly describes the knight/knave status of each person.
    
    Args:
        names: List of names
        solution: List of booleans (True for knight, False for knave)
        solution_text: String describing the solution
    
    Returns:
        Boolean indicating if the solution_text is correct, and any discrepancies found
    """
    # Make sure we have the same number of names and solutions
    if len(names) != len(solution):
        return False, "Mismatch in lengths of names and solution arrays"
    
    # Clean up the solution text and split by commas and 'and'
    text = solution_text.split("RESULT:")[-1].strip().replace('.', '')
    # Handle 'and' at the end
    text = text.replace(' and ', ', ')
    
    parts = text.split(', ')
    
    if len(parts) != len(names):
        return False, f"Solution text has {len(parts)} parts but there are {len(names)} people"
    
    # Check each person
    discrepancies = []
    
    for i, part in enumerate(parts):
        # Find which name this part refers to
        name_idx = -1
        for j, name in enumerate(names):
            if name in part:
                name_idx = j
                break
        
        if name_idx == -1:
            discrepancies.append(f"Couldn't find any name in '{part}'")
            continue
            
        # Check if the knight/knave status is correct
        is_knight = "knight" in part.lower()
        is_knave = "knave" in part.lower()
        
        if is_knight and not solution[name_idx]:
            discrepancies.append(f"{names[name_idx]} is described as knight but should be knave")
        elif is_knave and solution[name_idx]:
            discrepancies.append(f"{names[name_idx]} is described as knave but should be knight")
        elif not is_knight and not is_knave:
            discrepancies.append(f"Couldn't determine if {names[name_idx]} is knight or knave in '{part}'")
    
    return len(discrepancies) == 0, discrepancies


def eval_dataset(data, field='solution_text', verified_col='verified', discrepancies_col='discrepancies'):
    """
    Updates the dataset with verification results.
    
    Args:
        data: The dataset to update
    """
    verified = []
    discrepancies = []
    
    for i in range(len(data)):
        names = data['names'][i]
        solution = data['solution'][i]
        solution_text = data[field][i]
        
        is_verified, discrepancy_list = verify_solution_text(names, solution, solution_text)
        
        verified.append(is_verified)
        discrepancies.append(", ".join(discrepancy_list))
    
    data = data.add_column(verified_col, verified)
    data = data.add_column(discrepancies_col, discrepancies)
    
    return data


def load_results(results_dir="./results/ood"):
    """
    Load all KnK results from the results directory.
    
    Args:
        results_dir: Directory containing results folders
        
    Returns:
        Dictionary mapping adapter names to their results
    """
    all_results = {}
    
    # Find all knk.json files
    knk_files = glob.glob(f"{results_dir}/**/knk.json", recursive=True)
    
    for file_path in knk_files:
        # Extract adapter name from path
        adapter_name = file_path.split(results_dir + '/')[1].split('/knk.json')[0]
        
        # Load the JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        all_results[adapter_name] = data
        
    return all_results


def visualize_results(all_results, output_dir="./results/visualizations"):
    """
    Create visualizations comparing the performance of different models.
    
    Args:
        all_results: Dictionary mapping adapter names to their results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract scores and model info
    model_data = []
    
    for adapter, results in all_results.items():
        model_size, model_type = extract_parts(adapter)
        
        for key in results['scores']:
            score = results['scores'][key]
            model_data.append({
                'Adapter': adapter,
                'Model Size': model_size,
                'Model Type': model_type,
                'Dataset': key,
                'Score': score
            })
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(model_data)
    
    # Create visualizations
    # 1. Bar chart comparing models across datasets
    plt.figure(figsize=(14, 8))
    
    # Group by model type and dataset
    for i, dataset in enumerate(["2ppl", "3ppl", "4ppl"]):
        dataset_df = df[df['Dataset'] == dataset]
        
        plt.subplot(1, 3, i+1)
        
        # Sort by model size, then type
        sorted_df = dataset_df.sort_values(['Model Size', 'Model Type'])
        
        # Create bar chart
        bars = plt.bar(range(len(sorted_df)), sorted_df['Score'], color='skyblue')
        
        # Add model info as labels
        plt.xticks(range(len(sorted_df)), 
                  [f"{row['Model Size']}\n{row['Model Type']}" for _, row in sorted_df.iterrows()], 
                  rotation=90)
        
        plt.title(f"Knights and Knaves - {dataset}")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)  # Leave some space at the top
        
        # Add score values on top of bars
        for j, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 1, 
                    f"{sorted_df['Score'].iloc[j]:.1f}%", 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/knk_comparison_by_dataset.png", dpi=300, bbox_inches='tight')
    
    # 2. Heatmap of model performance
    plt.figure(figsize=(12, 8))
    
    # Pivot data for heatmap
    heatmap_data = df.pivot_table(
        index=['Model Size', 'Model Type'], 
        columns='Dataset', 
        values='Score'
    )
    
    # Sort by model size
    model_sizes = ['0.5B', '1.5B']
    heatmap_data = heatmap_data.reindex(
        pd.MultiIndex.from_product([model_sizes, heatmap_data.index.get_level_values('Model Type').unique()]), 
        names=['Model Size', 'Model Type']
    )
    
    # Create heatmap
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    
    # Add labels
    plt.colorbar(label='Accuracy (%)')
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(range(len(heatmap_data.index)), 
              [f"{idx[0]} - {idx[1]}" for idx in heatmap_data.index])
    
    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not pd.isna(value):
                plt.text(j, i, f"{value:.1f}%", ha='center', va='center', 
                        color='white' if value < 50 else 'black')
    
    plt.title('Knights and Knaves Performance by Model and Dataset')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/knk_heatmap.png", dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    print("\nModel Performance Summary:")
    print("-------------------------")
    print(df.groupby(['Model Size', 'Model Type'])['Score'].mean().sort_values(ascending=False))
    
    return df


if __name__ == "__main__":
    data_ = datasets.load_dataset("K-and-K/knights-and-knaves", name="train")
    context_len = 2048
    temperature = 0.7

    keys = ["2ppl", "3ppl", "4ppl"]
    results = {}
    results['trajectories'] = {}
    results['scores'] = {}

    for adapter in [
            # "chloeli/qwen-2.5-1.5B-instruct-sft-lora-countdown-deepseek-correct-seq8k-1k",
            # "chloeli/qwen-2.5-1.5B-instruct-sft-lora-countdown-deepseek-5k",
            # "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-optimal-seq8k-5k",
        
            "chloeli/qwen-2.5-0.5B-instruct-sft-lora-countdown-search-seq8k-5k",
            "chloeli/qwen-2.5-0.5B-instruct-sft-lora-countdown-search-react-correct-seq10k-5k",
            "chloeli/qwen-2.5-0.5B-instruct-sft-lora-countdown-optimal-seq8k-5k",
            "chloeli/qwen-2.5-0.5B-instruct-sft-lora-countdown-optimal-1k",
            "chloeli/qwen-2.5-0.5B-instruct-sft-lora-countdown-o3-5k",
            "chloeli/qwen-2.5-0.5B-instruct-sft-lora-countdown-o3-1k"
        ]:
        
        # if model: del model
        # if tokenizer: del tokenizer
        batch_size=16
        if "Qwen" in adapter:
            model, tokenizer = load_model_from_hub(adapter)
        else:
            model, tokenizer = load_model(adapter)
        model.eval()
        model.cuda()
        tokenizer.pad_token = tokenizer.eos_token

        def message_template(example_question):
            return [{ "content": f"{example_question}.\nConclude with the final result in EXACTLY this format:\n```\nSOLUTION: YES/NO\ \nRESULT: final_value\n```\nThe final_value should be statements separated by commas. For example, 'Michael is a knight, Zoey is a knight, and Ethan is a knight.'", "role": "user" }]

        for key in keys:
            output_texts_concat = []

            data = data_[key]
            data = data.map(lambda x: {
                "test_prompt": message_template(x['quiz']) 
            })
            
            # Generate completions for this batch
            for i, data_batch in tqdm(enumerate(data.iter(batch_size=batch_size)), total=len(data)//batch_size):   
                chat_inputs = tokenizer.apply_chat_template(data_batch["test_prompt"], return_tensors="pt", padding=True, truncation=True, max_length=context_len, return_length=True, tokenize=False)
                outputs = generate_batch(model, tokenizer, chat_inputs, max_new_tokens=context_len, temperature=temperature)
                output_texts_concat.extend(outputs)

            # Add completions column to dataset
            column_name = f"completions_{key}"
            data = data.add_column(column_name, output_texts_concat)
            
            # Evaluate completions
            verified_column = f"verified_{key}"
            discrepancies_column = f"discrepancies_{key}"
            data = eval_dataset(data, column_name, verified_column, discrepancies_column)
            
            # Calculate score
            score = data[verified_column].count(True) / len(data) * 100
            print(f"{key} score: {score:.2f}%")
            
            # Store score and trajectories
            results['scores'][key] = score
            results['trajectories'][key] = []
            
            # Create trajectory data using the correct column names for each key
            for i in range(len(data)):
                results['trajectories'][key].append({
                    'completions': data[column_name][i],
                    'verified': data[verified_column][i],
                    'discrepancies': data[discrepancies_column][i]
                })

        savepath = f"./results/ood/{adapter}/knk.json"
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, 'w') as f:
            json.dump(results, f, indent=4)
    
    # After all models have been evaluated, visualize the results
    # print("\nGenerating visualizations...")
    # all_results = load_results()
    # visualize_results(all_results)
    # print("\nVisualizations saved to ./results/visualizations/")

