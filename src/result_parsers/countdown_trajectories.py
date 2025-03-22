from tqdm import tqdm
# from datasets import load_dataset
import re
import json, sys, os, glob
import numpy as np

import matplotlib.pyplot as plt
import re
def evaluate_countdown_trajectory(prompt, trajectory):
    """
    Evaluates a countdown puzzle solver trajectory to determine if it correctly found a solution.
    
    Args:
        trajectory (str): A string containing the full trajectory of a countdown solver run
    
    Returns:
        dict: A dictionary containing evaluation results with keys:
            - 'solved': Boolean indicating if a solution was found
            - 'target': The target number that the solver was trying to reach
            - 'initial_numbers': List of initial numbers available for the calculation
            - 'operations': List of operations performed to reach the solution (if solved)
            - 'final_value': The final value reached by the solver
    """
    lines = trajectory.strip().split('\n')
    
    # Extract initial state from the first line
    # initial_line = lines[0]
    # target_and_numbers = initial_line.split("Current State: ")[1].split(", Operations:")[0]
    
    problem_match = re.search(r"Make (\d+) with the numbers \[([0-9, ]+)\]", prompt)
    target = int(problem_match.group(1))
    initial_numbers = [int(num.strip()) for num in problem_match.group(2).split(',')]
    

    # target = int(target_and_numbers.split(":")[0])
    # initial_numbers_str = target_and_numbers.split(":")[1].strip()
    # initial_numbers = [int(num) for num in initial_numbers_str.strip("[]").split(", ")]
    
    # Check if a solution was found
    solved = False
    operations = []
    final_value = None
    try:
        for line in lines:
            # Check for "Goal Reached" message
            if "Goal Reached" in line:
                solved = True
                # Extract the final value from the comparison
                comparison_part = line.split("equal:")[0].strip()
                final_value = int(comparison_part.split(",")[0])
                
                # Find the operations from the last state before "Goal Reached"
                for i in range(lines.index(line) - 1, -1, -1):
                    if "Operations:" in lines[i]:
                        operations_str = lines[i].split("Operations: ")[1].strip()
                        # Handle empty operations list
                        if operations_str == "[]":
                            operations = []
                        else:
                            # Remove brackets and split by comma
                            operations_str = operations_str.strip("[]'")
                            operations = operations_str.split("', '")
                        break
                break
            
            # Check for explicit "No solution found" message
            if "No solution found" in line:
                solved = False
                final_value = None
                operations = []
                break
    
        return {
            'solved': solved,
            'target': target,
            'initial_numbers': initial_numbers,
            'operations': operations,
            'final_value': final_value
        }

    except Exception as e:
        # print(f"Error evaluating trajectory: {e}")
        return {
            'solved': False,
            'target': target,
            'initial_numbers': initial_numbers,
            'operations': [],
            'final_value': None
        }
        
        

def evaluate_countdown_trajectories(results_all_trials):
    """
    Evaluates a dataset of countdown problem solver trajectories.
    data should have format [
        [ # trial 0
        {'prompt': str, 'completion': str},
        {'prompt': str, 'completion': str},
        ...
        ],
        [ # trial 1
        {'prompt': str, 'completion': str},
        {'prompt': str, 'completion': str},
        ...
        ],
    ]
    """
    question_solved = {i: False for i in range(len(results_all_trials[0]))}
    trial_success_rates = []
    for i in range(len(results_all_trials)): # number of trials
        for j in range(len(results_all_trials[i])): # number of samples in each trial
            problem_text = results_all_trials[i][j].get("prompt", "")
            solution_text = results_all_trials[i][j].get("completion", "")
            results_all_trials[i][j]['parsed_results'] = evaluate_countdown_trajectory(problem_text, solution_text)
            
            # Update whether this question was solved in any trial
            question_solved[j] = question_solved[j] or results_all_trials[i][j]['parsed_results']['solved']
    # list of success rates in each trial
    trial_success_rates = [sum([r['parsed_results']['solved'] for r in trial]) / len(trial) for trial in results_all_trials]
    print(f"Success rate for each trial: {trial_success_rates}")
    # Calculate best-of-n success rate (across all trials for each question)
    best_of_n_successes = sum(question_solved.values())
    best_of_n_rate = best_of_n_successes / len(question_solved)
    
    # Calculate mean success rate across trials
    mean_of_n_trials = sum(trial_success_rates) / len(trial_success_rates)
    
    # Store results
    aggregated_results = {
            'trial_success_rates': trial_success_rates,
            'best_of_n': best_of_n_rate,
            'mean': mean_of_n_trials
    }
    results_all_trials.insert(0, aggregated_results)
    
    print(f"\nSummary:")
    print(f"  Best-of-{len(trial_success_rates)} success rate: {best_of_n_rate:.4f} ({best_of_n_successes}/{len(question_solved)})")
    print(f"  Mean success rate across trials: {mean_of_n_trials:.4f}\n")
    
    return results_all_trials
def plot_results(results_all_trials):
    # Plotting
    plt.figure(figsize=(12, 8))

    # Sort files by best-of-n performance
    sorted_files = sorted(file_results.keys(), key=lambda x: file_results[x]['best_of_n'], reverse=True)

    x = np.arange(len(sorted_files))
    width = 0.35

    # Create the plot
    ax = plt.subplot(111)
    best_bars = ax.bar(x - width/2, [file_results[f]['best_of_n'] for f in sorted_files], width, 
                    label='Best-of-3 (Question Level)')
    mean_bars = ax.bar(x + width/2, [file_results[f]['mean'] for f in sorted_files], width, 
                    label='Mean Across Trials')

    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0, fontsize=8)

    add_labels(best_bars)
    add_labels(mean_bars)

    # Customize the plot
    ax.set_ylabel('Success Rate')
    ax.set_title('Countdown Problem Success Rates by Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in sorted_files], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max([file_results[f]['best_of_n'] for f in sorted_files]) * 1.1)

    plt.tight_layout()
    plt.savefig('countdown_success_rates.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    

# if __name__ == "__main__":
#     # Example usage
#     # %% Testing on the generated trajectories
#     data = load_dataset("chloeli/stream-of-search-countdown-10k")
#     data.keys()

#     # load
#     for split in data.keys():
#         prompts= data[split]['messages']
#         # do this with map function
#         results = list(map(lambda x: {
#             'prompt': x[0]['content'],
#             'completion': x[1]['content']
#         }, prompts))
        
        # res = []
        # for i in range(len(results)):
        #     problem_text = results[i].get("prompt", "")
        #     solution_text = results[i].get("completion", "")
        #     res.append(evaluate_countdown_trajectory(problem_text, solution_text))

#         # aggregate for accuracy
#         successes = [r["solved"] for r in res]
#         success_rate = sum(successes) / len(successes)
#         print(f"Success rate for file {split}: {success_rate:.4f} ({sum(successes)}/{len(successes)})")
#         assert success_rate>0.3, f"sanity check failed training dataset is poor: success_rate {success_rate} < 0.3"
        
        
#     #%% Evaluation on LLM generated trajectories
#     files = glob.glob("/cs/student/msc/ml/2024/ycheah/projects/sos/stream-of-search/results/chloeli_stream-of-search-countdown-10k/*")
#     for file in files:
#         # Extract a friendly name for display
#         file_name = file.split('qwen-2.5-0.5b-instruct-sft-qlora-countdown-search-1k_')[-1]

        
#         # Load the data
#         with open(file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
            
#         # Evaluate the data
#         file_results = evaluate_countdown_trajectories(data)
        