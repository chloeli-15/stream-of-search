# qwen-2.5-0.5B-instruct-sft-lora-countdown-search-1k
import sys
sys.path.append('/cs/student/msc/ml/2024/ycheah/projects/sos/stream-of-search')
from finetune.run_adapter_model import load_model, generate, generate_batch, load_model_from_hub

from tqdm import tqdm
import datasets
data_ = datasets.load_dataset("K-and-K/knights-and-knaves", name="train")

def message_template(example_question):
    return [{ "content": f"{example_question}.\nConclude with the final result in EXACTLY this format:\n```\nSOLUTION: YES/NO\ \nRESULT: final_value\n```\nThe final_value should be statements separated by commas. For example, 'Michael is a knight, Zoey is a knight, and Ethan is a knight.'", "role": "user" }]

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

# use the results to update the verified and discrepancies column of the data_[key]_[key]set
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

def visualize_example_count_performance():
    """
    Visualize model performance by number of examples (3num vs 5num),
    clustered by example count with model types within each cluster
    """
    # Store data by example count and model type
    data_by_example_count = {}
    
    # Find all relevant JSON files in qwen folders
    for folder in glob.glob("./results/qwen*".lower()):
        folder_name = os.path.basename(folder)
        
        # Extract model size
        model_size = folder_name.split("-instruct")[0].split("2.5-")[-1]
        
        # Extract approach type (search-seq8k vs search-react-seq8k)
        approach_match = re.search(r'countdown-(search-[^_-]+)', folder_name)
        if approach_match:
            approach_type = approach_match.group(1)
        else:
            # Fallback to a more general pattern
            approach_match = re.search(r'countdown-([^_]+)', folder_name)
            approach_type = approach_match.group(1) if approach_match else "unknown"
            
        model_key = f"{model_size}-{approach_type}"
        
        for file in glob.glob(f"{folder}/countdown_*.json"):
            with open(file, "r") as f:
                data = json.load(f)
            
            hyperparams = data[0]['hyperparams']
            mean_success_rate = data[1]['mean']
            sample_size = hyperparams['num']
            
            # Extract example count from filename
            filename = os.path.basename(file)
            example_count_match = re.search(r'countdown_(\d+)num_(\d+)', filename)
            
            if example_count_match:
                example_count = int(example_count_match.group(1))
                dataset_size = int(example_count_match.group(2))
                
                # Store data grouped by example count first
                example_key = f"{example_count}num"
                if example_key not in data_by_example_count:
                    data_by_example_count[example_key] = {}
                
                if model_key not in data_by_example_count[example_key]:
                    data_by_example_count[example_key][model_key] = []
                
                data_by_example_count[example_key][model_key].append({
                    'success_rate': mean_success_rate,
                    'sample_size': sample_size,
                    'dataset_size': dataset_size,
                    'file': filename
                })
    
    # Create visualization if we have data
    if not data_by_example_count:
        print("No example count data found.")
        return
    
    # Sort example counts and model types
    example_counts = sorted(data_by_example_count.keys())
    all_model_types = sorted(set(model for example_data in data_by_example_count.values() 
                                for model in example_data.keys()))
    
    # Create figure for bar chart
    plt.figure(figsize=(14, 8))
    
    # Set up the bar positions
    x = np.arange(len(example_counts))
    bar_width = 0.8 / len(all_model_types)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_model_types)))
    
    # Plot bars grouped by example count, with model types within each cluster
    for i, model_type in enumerate(all_model_types):
        avg_rates = []
        sample_sizes = []
        
        for count in example_counts:
            if count in data_by_example_count and model_type in data_by_example_count[count]:
                runs = data_by_example_count[count][model_type]
                avg_rate = np.mean([run['success_rate'] for run in runs])
                avg_rates.append(avg_rate)
                max_sample = max([run['sample_size'] for run in runs])
                sample_sizes.append(max_sample)
            else:
                avg_rates.append(0)
                sample_sizes.append(0)
        
        # Position bars within each cluster
        offset = i - (len(all_model_types) - 1) / 2
        x_pos = x + offset * bar_width
        
        # Plot the bars
        bars = plt.bar(x_pos, avg_rates, width=bar_width, label=model_type, color=colors[i])
        
        # Add value labels
        for bar, value, sample in zip(bars, avg_rates, sample_sizes):
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f"{value:.2f}\nn={sample}", ha='center', va='bottom', 
                        fontsize=8)
    
    # Configure the plot
    plt.xlabel('Number of Examples in Input', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Model Performance by Number of Examples', fontsize=14)
    plt.xticks(x, example_counts)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    # Add legend
    plt.legend(title='Model Types', loc='upper center', bbox_to_anchor=(0.5, -0.1), 
              ncol=min(3, len(all_model_types)))
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("./results/example_count_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    adapters= [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "chloeli/qwen-2.5-1.5B-instruct-sft-lora-countdown-search-1k",
            "chloeli/qwen-2.5-1.5B-instruct-sft-lora-countdown-search-react-1k"
        ]
        
    CTX = 512
    for adapter in adapters:
        if 'react' in adapter:
            context_len = CTX*2
        else:
            context_len = CTX
            
        batch_size=32
        if "chloeli" in adapter:
            model, tokenizer = load_model(adapter)
        else:
            model, tokenizer = load_model_from_hub(adapter)
        model.eval()
        model.cuda()
        tokenizer.pad_token = tokenizer.eos_token

        temperature = 0.7

        keys = ["2ppl", "3ppl"]
        results = {}
        results['trajectories'] = {}
        results['scores'] = {}

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

        import json, os
        savepath = f"./results/ood/{adapter}/knk.json"
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, 'w') as f:
            json.dump(results, f, indent=4)
            
    
    # Visualize example count performance
    visualize_example_count_performance()