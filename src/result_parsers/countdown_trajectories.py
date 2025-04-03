from tqdm import tqdm
# from datasets import load_dataset
import re
import json, sys, os, glob
import numpy as np
from typing import List, Tuple
import math
import matplotlib.pyplot as plt
import re
import re
import ast
import tiktoken

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

# def evaluate_countdown_trajectory(ds_entry):
#     """
#     Evaluates a countdown puzzle solver trajectory to determine if it correctly found a solution.
    
#     Args:
#         ds_entry (dict): Dictionary containing problem data and model response
    
#     Returns:
#         dict: A dictionary containing evaluation results
#     """
#     import re
#     import ast
    
#     # Extract the target and initial numbers from the entry
#     target = ds_entry.get('target')
#     initial_numbers = ds_entry.get('nums', [])
    
#     # Extract just the assistant's response
#     completion = ds_entry.get('completion', '')
#     parts = completion.split("\nassistant\n")
#     assistant_response = parts[-1] if len(parts) > 1 else completion
    
#     # Initialize default values
#     solved = False
#     operations = []
#     final_value = None
    
#     # First check for the explicit SOLUTION: YES/NO format, even without line breaks
#     solution_match = re.search(r"SOLUTION:\s*(YES|NO)", assistant_response, re.IGNORECASE)
#     if solution_match:
#         solved = solution_match.group(1).upper() == "YES"
        
#         # Extract operations list - handle both with and without line breaks
#         operations_match = re.search(r"OPERATIONS:\s*(\[.*?\])", assistant_response, re.DOTALL)
#         if operations_match:
#             ops_str = operations_match.group(1)
#             try:
#                 # Clean up the string before parsing
#                 cleaned_ops = ops_str.replace("'", '"').replace('\n', '').strip()
#                 operations = ast.literal_eval(cleaned_ops)
#             except:
#                 # Try extracting individual operations using regex
#                 op_matches = re.findall(r"['\"]([^'\"]*?=[^'\"]*?)['\"]", ops_str)
#                 if op_matches:
#                     operations = op_matches
        
#         # Extract the result value
#         result_match = re.search(r"RESULT:\s*(\d+\.?\d*)", assistant_response)
#         if result_match:
#             try:
#                 final_value = float(result_match.group(1))
#                 if final_value.is_integer():
#                     final_value = int(final_value)
#             except:
#                 final_value = None
#     else:
#         # Check for operations pattern even without SOLUTION header
#         op_matches = re.findall(r"(\d+\s*[\+\-\*/]\s*\d+\s*=\s*\d+)", assistant_response)
#         if op_matches:
#             operations = op_matches
            
#             # Check for a clear confirmation message
#             if re.search(r"that works|success!|solved|got it", assistant_response, re.IGNORECASE):
#                 solved = True
                
#                 # Try to extract the result value
#                 result_match = re.search(r"(\d+)\s*$", assistant_response)
#                 if result_match:
#                     try:
#                         final_value = int(result_match.group(1))
#                     except:
#                         final_value = None
#                 else:
#                     # If no explicit result, check if target appears near the end
#                     if str(target) in assistant_response[-100:]:
#                         final_value = target
            
#         # Check for explicit failure messages
#         if re.search(r"not possible|cannot|impossible|no solution", assistant_response, re.IGNORECASE):
#             solved = False
    
#     # If the operations are valid and the final operation results in the target, mark as solved
#     if operations and not final_value:
#         for op in operations:
#             result_match = re.search(r"=\s*(\d+)$", op)
#             if result_match and int(result_match.group(1)) == target:
#                 final_value = target
#                 solved = True
    
#     # Validate operations if they exist
#     if solved and operations:
#         valid = validate_operations(initial_numbers, operations, target)
#         solved = valid and (final_value == target)
    
    
#     return {
#         'solved': solved,
#         'target': target,
#         'initial_numbers': initial_numbers,
#         'operations': operations,
#         'final_value': final_value
#     }

def evaluate_countdown_trajectory_claude(ds_entry) -> Tuple[bool, str]:
    """
    Given:
      - target: the desired final result.
      - nums: list of initial numbers.
      - trajectory: a string that should include lines of the form:
            SOLUTION: YES/NO
            OPERATIONS: [list of strings like 'A+B=C', ...]
            RESULT: final_value
    This function uses regex to extract the parts and then simulates the operations
    by “consuming” initial numbers (and intermediate results) from a pool, verifying:
      • each operation string has one binary operator
      • each operation is valid (left op right equals given result)
      • the operations use all initial numbers exactly once (by simulating removal from pool).
    Returns a tuple (is_valid, message) where is_valid is True only if the trajectory is correct.
    """
    # target: int, nums: List[int], trajectory: str
    trajectory = ds_entry['completion']
    target = ds_entry['target']
    nums = ds_entry['nums']
    # Extract SOLUTION (YES or NO)
    sol_match = re.search(r"SOLUTION:\s*(YES|NO)", trajectory, re.IGNORECASE)
    if not sol_match:
        return False, "Could not find SOLUTION declaration."
    sol_decl = sol_match.group(1).upper()
    if sol_decl != "YES":
        return False, "The trajectory indicates no valid solution."

    # Extract OPERATIONS list
    ops_match = re.search(r"OPERATIONS:\s*(\[[^\]]*\])", trajectory, re.DOTALL)
    if not ops_match:
        return False, "Could not find OPERATIONS list."
    try:
        operations = ast.literal_eval(ops_match.group(1))
        if not isinstance(operations, list):
            return False, "OPERATIONS is not a list."
    except Exception as e:
        return False, f"Failed to parse OPERATIONS list: {e}"
        
    # Extract final RESULT
    res_match = re.search(r"RESULT:\s*([-+.\d]+)", trajectory)
    if not res_match:
        return False, "Could not find RESULT."
    try:
        expected_final = float(res_match.group(1))
    except Exception as e:
        return False, f"Failed to parse RESULT: {e}"
    
    # Simulation: available numbers (as floats).
    available = [float(n) for n in nums]
    # We simulate the sequence of operations.
    # Each operation must be of the form "operand1 operator operand2 = result"
    op_pattern = re.compile(r"^\s*([\d.]+)\s*([\+\-\*/])\s*([\d.]+)\s*=\s*([\d.]+)\s*$")
    
    for idx, op_str in enumerate(operations):
        m = op_pattern.match(op_str)
        if not m:
            return False, f"Operation '{op_str}' does not match required pattern."
        op1_str, operator, op2_str, given_result_str = m.groups()
        try:
            op1 = float(op1_str)
            op2 = float(op2_str)
            op_result = float(given_result_str)
        except Exception as e:
            return False, f"Error converting numbers in op '{op_str}': {e}"
            
        # Check that the operation is valid:
        if operator == '+':
            computed = op1 + op2
        elif operator == '-':
            computed = op1 - op2
        elif operator == '*':
            computed = op1 * op2
        elif operator == '/':
            # Avoid division by zero
            if math.isclose(op2,0.0):
                return False, f"Division by zero in op '{op_str}'."
            computed = op1 / op2
        else:
            return False, f"Unknown operator '{operator}' in op '{op_str}'."
            
        if not math.isclose(computed, op_result, rel_tol=1e-5):
            return False, f"In op '{op_str}', computed {computed} which does not match given {op_result}."
            
        # Now simulate consumption:
        # For each operand, check if it is "available". If yes, remove one instance.
        # (We assume that if the operand equals a value in available within tolerance,
        #  it comes from the pool of that operand.)
        def consume(value: float, pool: List[float]) -> bool:
            for i, num in enumerate(pool):
                if math.isclose(num, value, rel_tol=1e-5):
                    del pool[i]
                    return True
            return False
        
        # Try to consume op1 and op2
        if not consume(op1, available):
            return False, f"Operand {op1} in op '{op_str}' not available from initial/intermediate numbers."
        if not consume(op2, available):
            return False, f"Operand {op2} in op '{op_str}' not available from initial/intermediate numbers."
        # Append current operation result to available pool
        available.append(op_result)
    
    # At the end, exactly one number should remain; it should equal the target.
    if len(available) != 1:
        return False, f"After all operations, expected one value but got {len(available)} values: {available}"
    if not math.isclose(available[0], float(target), rel_tol=1e-5):
        return False, f"Final value {available[0]} does not equal the target {target}."
    
    return True, f"Trajectory is valid with operations: {operations}"
    
#     return {
#         'solved': solved,
#         'target': target,
#         'initial_numbers': initial_numbers,
#         'operations': operations,
#         'final_value': final_value
#     }

def evaluate_countdown_trajectory(ds_entry):
    solved, remarks = evaluate_countdown_trajectory_claude(ds_entry)
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(ds_entry['completion'])

    return {
        'solved': solved,
        'target': ds_entry['target'],
        'initial_numbers': ds_entry['nums'],
        'remarks': remarks,
        'completion_length': len(tokens),
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
