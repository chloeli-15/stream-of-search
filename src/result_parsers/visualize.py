#!/usr/bin/env python3
"""
Visualization script for countdown evaluation results.
This script loads result files from the 'results' directory and creates 
visualizations to compare model performance across different configurations.
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_results(results_dir="results", pattern="*.json"):
    """Load all result files matching the pattern from the results directory"""
    results_files = glob.glob(os.path.join(results_dir, "**", pattern), recursive=True)
    
    all_results = []
    for file_path in results_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract metadata from filename
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                
                # Basic metadata extraction
                metadata = {
                    'file_path': file_path,
                    'filename': filename,
                }
                
                # Extract model name and dataset from either filename or hyperparams
                if len(data) > 0 and "hyperparams" in data[0]:
                    hyperparams = data[0]["hyperparams"]
                    metadata['model'] = hyperparams.get("adapter", "").split("/")[-1]
                    metadata['dataset'] = hyperparams.get("data", "")
                    metadata['num_samples'] = hyperparams.get("num", 0)
                
                # Get performance metrics
                if len(data) > 1:
                    metrics = data[1]
                    metadata['mean_success'] = metrics.get("mean", 0)
                    metadata['best_of_n_success'] = metrics.get("best_of_n", 0)
                    metadata['trial_success_rates'] = metrics.get("trial_success_rates", [])
                
                all_results.append({
                    'metadata': metadata,
                    'full_data': data
                })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_results

def plot_success_rates(results, output_dir="visualizations"):
    """Plot success rates across different models and configurations"""
    if not results:
        print("No results to visualize")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    plot_data = []
    for result in results:
        metadata = result['metadata']
        plot_data.append({
            'Model': metadata.get('model', 'Unknown'),
            'Dataset': metadata.get('dataset', 'Unknown').split('/')[-1],
            'Samples': metadata.get('num_samples', 0),
            'Mean Success Rate': metadata.get('mean_success', 0),
            'Best-of-N Success Rate': metadata.get('best_of_n_success', 0)
        })
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(plot_data)
    
    # Sort by performance
    df = df.sort_values('Mean Success Rate', ascending=False)
    
    # 1. Bar chart of mean success rates
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Model', y='Mean Success Rate', data=df, hue='Dataset')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.title('Mean Success Rate by Model and Dataset')
    plt.ylim(0, max(df['Mean Success Rate']) * 1.2)  # Add some padding at the top
    
    # Add value labels on top of bars
    for i, v in enumerate(df['Mean Success Rate']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.savefig(os.path.join(output_dir, 'mean_success_rates.png'), dpi=300, bbox_inches='tight')
    
    # 2. Mean vs Best-of-N comparison
    plt.figure(figsize=(14, 8))
    
    x = range(len(df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x, df['Mean Success Rate'], width, label='Mean Success Rate')
    ax.bar([i + width for i in x], df['Best-of-N Success Rate'], width, label='Best-of-N Success Rate')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Success Rate')
    ax.set_title('Mean vs Best-of-N Success Rate by Model')
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_vs_best_of_n.png'), dpi=300, bbox_inches='tight')
    
    # 3. Dataset comparison
    if len(df['Dataset'].unique()) > 1:
        plt.figure(figsize=(14, 8))
        dataset_pivot = df.pivot_table(
            values='Mean Success Rate', 
            index='Model', 
            columns='Dataset', 
            aggfunc='mean'
        )
        
        dataset_pivot.plot(kind='bar', figsize=(14, 8))
        plt.title('Model Performance by Dataset')
        plt.ylabel('Mean Success Rate')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset_comparison.png'), dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {output_dir}")

def analyze_individual_problems(results, output_dir="visualizations"):
    """Analyze which problems are most difficult across models"""
    if not results:
        return
    
    # Extract individual problem results
    problem_results = {}
    
    for result in results:
        model_name = result['metadata'].get('model', 'Unknown')
        full_data = result['full_data']
        
        if len(full_data) <= 2:  # No problem data
            continue
        
        for i in range(2, len(full_data)):
            problem_data = full_data[i]
            
            if 'prompt' not in problem_data or 'parsed_results' not in problem_data:
                continue
                
            prompt = problem_data['prompt']
            # Create a simplified prompt for matching across models
            simple_prompt = prompt.strip().split("\n")[0]  # Just take first line
            
            if simple_prompt not in problem_results:
                problem_results[simple_prompt] = {
                    'prompt': prompt,
                    'target': problem_data['parsed_results'].get('target'),
                    'initial_numbers': problem_data['parsed_results'].get('initial_numbers'),
                    'models_solved': [],
                    'models_failed': []
                }
            
            if problem_data['parsed_results'].get('solved', False):
                problem_results[simple_prompt]['models_solved'].append(model_name)
            else:
                problem_results[simple_prompt]['models_failed'].append(model_name)
    
    # Calculate difficulty score (percentage of models that failed)
    for prompt, data in problem_results.items():
        total_attempts = len(data['models_solved']) + len(data['models_failed'])
        if total_attempts > 0:
            data['difficulty'] = len(data['models_failed']) / total_attempts
        else:
            data['difficulty'] = 0
    
    # Sort by difficulty
    sorted_problems = sorted(problem_results.items(), key=lambda x: x[1]['difficulty'], reverse=True)
    
    # Output the most difficult problems
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'problem_difficulty.json'), 'w') as f:
        json.dump({
            'most_difficult': sorted_problems[:10],
            'least_difficult': sorted_problems[-10:],
            'all_problems': problem_results
        }, f, indent=2)
    
    # Plot difficulty distribution
    difficulties = [data['difficulty'] for _, data in problem_results.items()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(difficulties, bins=20)
    plt.title('Distribution of Problem Difficulty')
    plt.xlabel('Difficulty Score (% of models that failed)')
    plt.ylabel('Number of Problems')
    plt.savefig(os.path.join(output_dir, 'difficulty_distribution.png'), dpi=300, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(description='Visualize countdown evaluation results')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory containing result files')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save visualizations')
    args = parser.parse_args()
    
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} result files")
    
    if not results:
        print("No results found")
        return
    
    print("Generating visualizations...")
    plot_success_rates(results, args.output_dir)
    analyze_individual_problems(results, args.output_dir)
    
    print(f"Visualizations saved to {args.output_dir}")
    
    # Print summary table
    print("\nSummary of Results:")
    print("-" * 80)
    print(f"{'Model':<30} {'Dataset':<25} {'Samples':<8} {'Mean':<8} {'Best-of-N':<8}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x['metadata'].get('mean_success', 0), reverse=True):
        metadata = result['metadata']
        model = metadata.get('model', 'Unknown')
        dataset = metadata.get('dataset', 'Unknown').split('/')[-1]
        samples = metadata.get('num_samples', 0)
        mean = metadata.get('mean_success', 0)
        best_of_n = metadata.get('best_of_n_success', 0)
        
        print(f"{model[:30]:<30} {dataset[:25]:<25} {samples:<8} {mean:.4f}  {best_of_n:.4f}")

if __name__ == "__main__":
    main()