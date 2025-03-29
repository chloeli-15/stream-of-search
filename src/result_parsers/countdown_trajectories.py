from tqdm import tqdm
# from datasets import load_dataset
import re
import json, sys, os, glob
import numpy as np

import matplotlib.pyplot as plt
import re
import re
import ast

def validate_operations(initial_numbers, operations, target):
    """
    Validates if the operations use all initial numbers exactly once and reach the target.
    """
    # Make a copy of initial numbers to track usage
    available_numbers = initial_numbers.copy()
    # Dictionary to store intermediate results
    intermediate_results = {}
    
    for i, operation in enumerate(operations):
        # Parse the operation (format like '9-5=4')
        parts = operation.split('=')
        if len(parts) != 2:
            return False
        
        result_str = parts[1].strip()
        expression = parts[0].strip()
        
        # Find operator (+, -, *, /)
        op_match = re.search(r'[+\-*/]', expression)
        if not op_match:
            return False
        
        operator = op_match.group(0)
        operands = expression.split(operator)
        if len(operands) != 2:
            return False
        
        left_operand = operands[0].strip()
        right_operand = operands[1].strip()
        
        # Try to get values for operands
        left_val = None
        if left_operand.isdigit() or (left_operand.startswith('-') and left_operand[1:].isdigit()):
            left_val = int(left_operand)
            if left_val in available_numbers:
                available_numbers.remove(left_val)
            elif left_val not in intermediate_results.values():
                return False  # Not an available number or intermediate result
        else:
            return False  # Non-numeric operand
        
        right_val = None
        if right_operand.isdigit() or (right_operand.startswith('-') and right_operand[1:].isdigit()):
            right_val = int(right_operand)
            if right_val in available_numbers:
                available_numbers.remove(right_val)
            elif right_val not in intermediate_results.values():
                return False  # Not an available number or intermediate result
        else:
            return False  # Non-numeric operand
        
        # Calculate the result
        calculated_result = None
        if operator == '+':
            calculated_result = left_val + right_val
        elif operator == '-':
            calculated_result = left_val - right_val
        elif operator == '*':
            calculated_result = left_val * right_val
        elif operator == '/':
            if right_val == 0:
                return False  # Division by zero
            calculated_result = left_val / right_val
        
        # Verify the result matches what's stated
        result_val = int(result_str) if result_str.isdigit() else None
        if result_val is None or abs(calculated_result - result_val) > 1e-10:
            return False
        
        # Store intermediate result for future operations
        intermediate_results[f"step{i}"] = result_val
    
    # Check if all initial numbers were used and final result matches target
    return len(available_numbers) == 0 and result_val == target

def evaluate_countdown_trajectory(ds_entry):
    """
    Evaluates a countdown puzzle solver trajectory to determine if it correctly found a solution.
    
    Args:
        ds_entry (dict): Dictionary containing problem data and model response
    
    Returns:
        dict: A dictionary containing evaluation results
    """
    import re
    import ast
    
    # Extract the target and initial numbers from the entry
    target = ds_entry.get('target')
    initial_numbers = ds_entry.get('nums', [])
    
    # Extract just the assistant's response
    completion = ds_entry.get('completion', '')
    parts = completion.split("\nassistant\n")
    assistant_response = parts[-1] if len(parts) > 1 else completion
    
    # Initialize default values
    solved = False
    operations = []
    final_value = None
    
    # First check for the explicit SOLUTION: YES/NO format, even without line breaks
    solution_match = re.search(r"SOLUTION:\s*(YES|NO)", assistant_response, re.IGNORECASE)
    if solution_match:
        solved = solution_match.group(1).upper() == "YES"
        
        # Extract operations list - handle both with and without line breaks
        operations_match = re.search(r"OPERATIONS:\s*(\[.*?\])", assistant_response, re.DOTALL)
        if operations_match:
            ops_str = operations_match.group(1)
            try:
                # Clean up the string before parsing
                cleaned_ops = ops_str.replace("'", '"').replace('\n', '').strip()
                operations = ast.literal_eval(cleaned_ops)
            except:
                # Try extracting individual operations using regex
                op_matches = re.findall(r"['\"]([^'\"]*?=[^'\"]*?)['\"]", ops_str)
                if op_matches:
                    operations = op_matches
        
        # Extract the result value
        result_match = re.search(r"RESULT:\s*(\d+\.?\d*)", assistant_response)
        if result_match:
            try:
                final_value = float(result_match.group(1))
                if final_value.is_integer():
                    final_value = int(final_value)
            except:
                final_value = None
    else:
        # Check for operations pattern even without SOLUTION header
        op_matches = re.findall(r"(\d+\s*[\+\-\*/]\s*\d+\s*=\s*\d+)", assistant_response)
        if op_matches:
            operations = op_matches
            
            # Check for a clear confirmation message
            if re.search(r"that works|success!|solved|got it", assistant_response, re.IGNORECASE):
                solved = True
                
                # Try to extract the result value
                result_match = re.search(r"(\d+)\s*$", assistant_response)
                if result_match:
                    try:
                        final_value = int(result_match.group(1))
                    except:
                        final_value = None
                else:
                    # If no explicit result, check if target appears near the end
                    if str(target) in assistant_response[-100:]:
                        final_value = target
            
        # Check for explicit failure messages
        if re.search(r"not possible|cannot|impossible|no solution", assistant_response, re.IGNORECASE):
            solved = False
    
    # If the operations are valid and the final operation results in the target, mark as solved
    if operations and not final_value:
        for op in operations:
            result_match = re.search(r"=\s*(\d+)$", op)
            if result_match and int(result_match.group(1)) == target:
                final_value = target
                solved = True
    
    # Validate operations if they exist
    if solved and operations:
        valid = validate_operations(initial_numbers, operations, target)
        solved = valid and (final_value == target)
    
    
    return {
        'solved': solved,
        'target': target,
        'initial_numbers': initial_numbers,
        'operations': operations,
        'final_value': final_value
    }
    
        
def evaluate_countdown_trajectories(results_all_trials):
    """
    Evaluates a dataset of countdown problem solver trajectories and returns a flattened
    list of results in the format expected by visualization tools and wandb logging.
    """
    # Create a dict to track which questions were solved in any trial
    question_solved = {i: False for i in range(len(results_all_trials[0]))}
    trial_success_rates = []
    
    # Process each trial
    for i in range(len(results_all_trials)):
        successes_in_trial = 0
        for j in range(len(results_all_trials[i])):
            # problem_text = results_all_trials[i][j].get("prompt", "")
            # solution_text = results_all_trials[i][j].get("completion", "")
            results_all_trials[i][j]['parsed_results'] = evaluate_countdown_trajectory(results_all_trials[i][j])
            
            # Update success counts
            if results_all_trials[i][j]['parsed_results']['solved']:
                successes_in_trial += 1
                question_solved[j] = True
                
        # Calculate success rate for this trial
        trial_success_rates.append(successes_in_trial / len(results_all_trials[i]))
    
    # Calculate aggregate statistics
    best_of_n_successes = sum(question_solved.values())
    best_of_n_rate = best_of_n_successes / len(question_solved)
    mean_of_n_trials = sum(trial_success_rates) / len(trial_success_rates)
    
    # Print summary statistics
    print(f"Success rate for each trial: {trial_success_rates}")
    print(f"\nSummary:")
    print(f"  Best-of-{len(trial_success_rates)} success rate: {best_of_n_rate:.4f} ({best_of_n_successes}/{len(question_solved)})")
    print(f"  Mean success rate across trials: {mean_of_n_trials:.4f}\n")
    
    # Create the properly formatted results array
    # Start with the metrics summary (which will be at index 1 after hyperparams are added)
    final_results = [{
        'trial_success_rates': trial_success_rates,
        'best_of_n': best_of_n_rate,
        'mean': mean_of_n_trials
    }]
    
    # Add individual trajectory results (using best results from any trial)
    for j in range(len(results_all_trials[0])):
        # Find the best result for this problem across all trials
        best_result = None
        for i in range(len(results_all_trials)):
            result = results_all_trials[i][j]
            if best_result is None or (result['parsed_results']['solved'] and not best_result['parsed_results']['solved']):
                best_result = result
        
        # Add this result to our final list
        if best_result:
            final_results.append(best_result)
    
    return final_results


# def plot_results(results_all_trials):
#     # Plotting
#     plt.figure(figsize=(12, 8))

#     # Sort files by best-of-n performance
#     sorted_files = sorted(file_results.keys(), key=lambda x: file_results[x]['best_of_n'], reverse=True)

#     x = np.arange(len(sorted_files))
#     width = 0.35

#     # Create the plot
#     ax = plt.subplot(111)
#     best_bars = ax.bar(x - width/2, [file_results[f]['best_of_n'] for f in sorted_files], width, 
#                     label='Best-of-3 (Question Level)')
#     mean_bars = ax.bar(x + width/2, [file_results[f]['mean'] for f in sorted_files], width, 
#                     label='Mean Across Trials')

#     # Add value labels on top of bars
#     def add_labels(bars):
#         for bar in bars:
#             height = bar.get_height()
#             ax.annotate(f'{height:.3f}',
#                         xy=(bar.get_x() + bar.get_width() / 2, height),
#                         xytext=(0, 3),  # 3 points vertical offset
#                         textcoords="offset points",
#                         ha='center', va='bottom', rotation=0, fontsize=8)

#     add_labels(best_bars)
#     add_labels(mean_bars)

#     # Customize the plot
#     ax.set_ylabel('Success Rate')
#     ax.set_title('Countdown Problem Success Rates by Configuration')
#     ax.set_xticks(x)
#     ax.set_xticklabels([f.replace('_', '\n') for f in sorted_files], rotation=45, ha='right')
#     ax.legend(loc='upper right')
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     ax.set_ylim(0, max([file_results[f]['best_of_n'] for f in sorted_files]) * 1.1)

#     plt.tight_layout()
#     plt.savefig('countdown_success_rates.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
    

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
        