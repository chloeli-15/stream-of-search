import os
import json
import re
import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

def parse_filename(filename):
    """Parse experiment filename to extract metadata."""
    basename = os.path.basename(filename)
    # Extract pattern like 'countdown_5num_32_20250403-025812.json'
    match = re.match(r'(\w+)_(\d+)num_(\d+)_(\d{8}-\d{6})\.json', basename)
    
    if match:
        problem_type = match.group(1)  # countdown
        num_items = int(match.group(2))  # 5
        sample_size = int(match.group(3))  # 32
        timestamp = match.group(4)  # 20250403-025812
        
        return {
            'problem_type': problem_type,
            'num_items': num_items,
            'sample_size': sample_size,
            'timestamp': timestamp,
            'filename': basename
        }
    return None

def load_experiment_data(filepath):
    """Load experiment data from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract relevant information
        hyperparams = data[0].get('hyperparams', {})
        results = data[1] if len(data) > 1 else {}
        
        return {
            'hyperparams': hyperparams,
            'results': results,
            'metadata': parse_filename(filepath)
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def find_experiment_files(base_dir, pattern="**/*countdown*.json"):
    """Find all experiment result files in the given directory."""
    return glob.glob(os.path.join(base_dir, pattern), recursive=True)

def create_results_dataframe(experiments):
    """Create a pandas DataFrame from experiment results."""
    rows = []
    
    for exp in experiments:
        if not exp:
            continue
            
        metadata = exp['metadata']
        results = exp['results']
        
        row = {
            'problem_type': metadata['problem_type'],
            'num_items': metadata['num_items'],
            'sample_size': metadata['sample_size'],
            'timestamp': metadata['timestamp'],
            'success_rate': results.get('mean', results.get('best_of_n', 0)),
            'model': exp['hyperparams'].get('adapter', '').split('/')[-1]
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

def visualize_experiment_results(df):
    """Create visualizations for experiment results."""
    plt.figure(figsize=(12, 8))
    
    # Group by problem complexity (num_items) and sample size
    sns.barplot(
        data=df, 
        x='num_items', 
        y='success_rate',
        hue='sample_size',
        palette='viridis'
    )
    
    plt.title('Success Rate by Problem Complexity and Sample Size')
    plt.xlabel('Number of Items in Problem')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1.0)
    plt.legend(title='Sample Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('experiment_results.png')
    plt.close()
    
    # Plot by model if multiple models exist
    if len(df['model'].unique()) > 1:
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=df, 
            x='num_items', 
            y='success_rate',
            hue='model',
            palette='Set2'
        )
        plt.title('Success Rate by Problem Complexity and Model')
        plt.xlabel('Number of Items in Problem')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1.0)
        plt.legend(title='Model')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
    
    # Create a heatmap of success rates
    if len(df) > 2:
        pivot_data = df.pivot_table(
            values='success_rate',
            index='num_items',
            columns='sample_size'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_data, 
            annot=True, 
            cmap='YlGnBu', 
            vmin=0, 
            vmax=1.0,
            fmt='.3f',
            linewidths=.5
        )
        plt.title('Success Rate Heatmap by Problem Complexity and Sample Size')
        plt.tight_layout()
        plt.savefig('success_rate_heatmap.png')
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize experiment results.')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing experiment result files')
    parser.add_argument('--pattern', type=str, default="**/*countdown*.json", help='Glob pattern to match files')
    
    args = parser.parse_args()
    
    # Find all experiment files
    files = find_experiment_files(args.dir, pattern=args.pattern)
    print(f"Found {len(files)} experiment files")
    
    # Load experiment data
    experiments = [load_experiment_data(f) for f in files]
    experiments = [e for e in experiments if e]  # Filter out None values
    
    if not experiments:
        print("No valid experiment data found")
        return
    
    # Create DataFrame
    df = create_results_dataframe(experiments)
    print(f"Created DataFrame with {len(df)} rows")
    print(df.head())
    
    # Visualize results
    visualize_experiment_results(df)
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    main()
