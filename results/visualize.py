import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os
import re


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

def visualize_results():
    # Dictionary to store data by model and folder
    data_by_model = {}
    # Store individual run data
    run_details = {}
    # Dictionary to store sample sizes by folder
    folder_sample_sizes = {}

    for folder in glob.glob("./results/qwen*".lower()):
        folder_name = os.path.basename(folder)
        
        for file in glob.glob(f"{folder}/test*.json"):
            with open(file, "r") as f:
                data = json.load(f)
            
            hyperparams = data[0]['hyperparams']
            mean = data[1]['mean']
            model_name = hyperparams['adapter'].split("instruct")[0].split("2.5-")[-1]
            sample_size = hyperparams['num']
            experiment_name = hyperparams.get('experiment_name', os.path.basename(file))
            print(experiment_name)
            experiment_name = "_".join(extract_parts(experiment_name))
            # Store folder sample size
            folder_sample_sizes[folder_name] = sample_size
            
            run_id = f"{model_name}_{folder_name}_{os.path.basename(file)}"
            run_details[run_id] = {
                "success_rate": mean,
                "model": model_name,
                "folder": folder_name,
                "file": os.path.basename(file),
                "sample_size": sample_size,
                "experiment_name": experiment_name.split(".json")[0]
            }
            
            if model_name not in data_by_model:
                data_by_model[model_name] = {}
            
            data_by_model[model_name][folder_name] = mean

    # 1. Overall Comparison Plot
    # Get unique model names and folder names
    models = sorted(data_by_model.keys())
    folder_names = sorted(set(folder for model_data in data_by_model.values() for folder in model_data.keys()))

    # Create a figure with 3 subplots (1 row, 3 columns)
    fig = plt.figure(figsize=(6, 6))

    # 1. Overall Comparison Plot (first subplot)
    ax_main = plt.subplot(1, 1, 1)

    # Set up the bar chart
    x = np.arange(len(models))
    width = 0.8 / len(folder_names)  # Width of each bar
    # Use a colormap with more distinct colors to avoid similar colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(folder_names)))

    # Plot bars for each folder
    for i, folder in enumerate(folder_names):
        values = [data_by_model[model].get(folder, 0) for model in models]
        sample_size = folder_sample_sizes.get(folder, "Unknown")
        bars = ax_main.bar(x + i*width - (len(folder_names)-1)*width/2, values, 
                        width=width, label=f"{folder.split('-countdown-')[1].split('_Melina')[0]} (n={sample_size})", color=colors[i])
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height > 0:  # Only label non-zero bars
                ax_main.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                            f"{value:.2f}", ha='center', va='bottom', 
                            fontsize=8)

    # Calculate average success rate per model
    model_avg = {model: np.mean([value for value in data_by_model[model].values()]) 
                for model in models}

    # Plot average line
    avg_values = [model_avg[model] for model in models]
    for i, (x_pos, y_pos) in enumerate(zip(x, avg_values)):
        ax_main.plot([x_pos-0.4, x_pos+0.4], [y_pos, y_pos], color='black', linestyle='-', linewidth=2)
        ax_main.text(x_pos, y_pos + 0.02, f"avg: {y_pos:.2f}", ha='center', fontsize=9)

    # Configure the plot
    ax_main.set_xlabel('Models', fontsize=12)
    ax_main.set_ylabel('Success Rate', fontsize=12)
    ax_main.set_title('Overall Model Performance Comparison', fontsize=14)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax_main.set_ylim(0, 1)
    ax_main.grid(axis='y', linestyle='--', alpha=0.7)
    ax_main.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label="50% success")
    
    # Move legend outside the plot to prevent it from covering the bars
    ax_main.legend(
        title='Folders (Sample Size)', 
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8
    )

    # Adjust layout to make room for the legend
    plt.subplots_adjust(bottom=0.3)
    
    plt.savefig("./results/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Individual Runs Scatter Plot (subsequent subplots)
    # Group run details by folder
    runs_by_folder = {}
    for run_id, details in run_details.items():
        folder = details['folder']
        if folder not in runs_by_folder:
            runs_by_folder[folder] = []
        runs_by_folder[folder].append(details)

    # Map model names to numeric indices for scatter plot x-positions
    model_indices = {model: i for i, model in enumerate(models)}

    # Calculate number of plots: one per folder
    num_plots = len(folder_names)

    # Determine layout: use 3 columns maximum
    num_cols = min(3, num_plots)
    num_rows = (num_plots + num_cols - 1) // num_cols  # Ceiling division

    # Create figure with appropriate size
    fig = plt.figure(figsize=(7*num_cols, 6*num_rows))
    
    # Create scatter plot for each folder
    folder_experiment_names = []
    for i, folder in enumerate(folder_names):
        # Create subplot, starting from position 1
        ax_folder = plt.subplot(num_rows, num_cols, i+1)
        
        if folder in runs_by_folder:
            folder_runs = runs_by_folder[folder]
            
            # Prepare data for scatter plot
            x_positions = []
            y_values = []
            colors_list = []
            experiment_names = []
            
            for run in folder_runs:
                model = run['model']
                if model in model_indices:
                    x_pos = model_indices[model]
                    x_positions.append(x_pos)
                    y_values.append(run['success_rate'])
                    colors_list.append(plt.cm.tab10(model_indices[model] % 10))
                    experiment_names.append(run['experiment_name'])
                    folder_experiment_names.append(run['experiment_name'])
            # Add some jitter to x positions
            x_jittered = [x + np.random.uniform(-0.3, 0.3) for x in x_positions]
            
            # Create scatter plot
            ax_folder.scatter(x_jittered, y_values, c=colors_list, alpha=0.7, s=80)
            
            # Annotate with experiment names
            for j in range(len(x_jittered)):
                ax_folder.annotate(experiment_names[j], (x_jittered[j], y_values[j]), 
                                textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            
            # Configure the plot
            ax_folder.set_ylabel('Success Rate', fontsize=12)
            sample_size = folder_sample_sizes.get(folder, "Unknown")
            ax_folder.set_title(f"Individual Runs with n={sample_size} in \n" + "\n".join(folder.split('-countdown-')), fontsize=14)
            ax_folder.set_xticks(range(len(models)))
            ax_folder.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
            ax_folder.set_ylim(0, 1.0)
            ax_folder.grid(axis='y', linestyle='--', alpha=0.7)
            ax_folder.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            
        else:
            ax_folder.text(0.5, 0.5, f"No run details available for {folder}", 
                        ha='center', va='center')
            ax_folder.set_title(f'{folder}', fontsize=14)
            ax_folder.axis('off')

    plt.savefig("./results/all_trials.png", dpi=300, bbox_inches='tight')

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
        
        # Extract approach type using extract_parts
        parts = extract_parts(folder_name)
        approach_type = parts[1]  # Fix: Use the second element of the list

        model_key = f"{model_size}-{approach_type}"
        print(f"Processing folder: {folder_name}, model_key: {model_key}")
        
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
                print(f"Added data point: {example_key}, {model_key}, success_rate: {mean_success_rate}")
    
    # Create visualization if we have data
    if not data_by_example_count:
        print("No example count data found.")
        return
    
    # Print the collected data for debugging
    print("\nCollected data:")
    for example_key, models in data_by_example_count.items():
        print(f"{example_key}:")
        for model_key, runs in models.items():
            rates = [run['success_rate'] for run in runs]
            print(f"  {model_key}: {rates}")
    
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
        
        # Add value labels - modified to label all bars, even zero height ones
        for bar, value, sample in zip(bars, avg_rates, sample_sizes):
            height = bar.get_height()
            # Label all bars, including zero-height ones
            y_pos = max(height, 0.01)  # Slightly above zero for zero-height bars
            plt.text(bar.get_x() + bar.get_width()/2, y_pos + 0.01,
                    f"{value:.2f}\nn={sample}", ha='center', va='bottom', 
                    fontsize=8)
    
    # Configure the plot
    plt.xlabel('Number of Examples', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Model Performance by Example Count', fontsize=14)
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
    visualize_results()
    visualize_example_count_performance()