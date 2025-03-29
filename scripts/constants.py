SEARCH_TYPES = {
    "bfs_1": "breadth first search",
    "bfs_2": "breadth first search",
    "bfs_3": "breadth first search",
    "bfs_4": "breadth first search",
    "bfs_5": "breadth first search",
    "dfs": "depth first search",
}

o3_batch_prompts_filename = "data/o3/batch_prompts.jsonl"
o3_results_filepath = "data/o3/results.jsonl"

merged_sos_database_filename = "data/sos_10k_b4_merged/train1_b4_t100_n10000.json"
merged_sos_o3_database_filename = "data/sos_10k_b4_merged/train1_b4_t100_n10000_with_o3.json"

o3_prompt_template = """
Hi! You'll be providing material to conduct supervised fine-tuning on a small model to learn how to play the countdown game using search strategies. In the Countdown game, we start with a set of four initial numbers (e.g., [1,2,4,5]) and a target value (e.g., 14). Our goal is to use arithmetic operations (+,-,*,/) to produce the target using all of the numbers. 

We formalize this game as a search problem:
- States: Sets of remaining numbers
- Initial State: Set of initial numbers
- Goal State: Using all numbers to reach target value
- Actions: Choose two numbers, apply an operation (+, -, *, /), replace them with the result
Constraints:
- All 4 numbers must be used exactly once

I'd like you to generate detailed reasoning traces for this game that: 
1. Uses {SEARCH_TYPE} with a heuristic. Propose a heuristic you think is promising and explain why 
2. Go through the full search trajectory in a structured manner towards the target using that strategy, with articulated thought processes 
3. Includes suboptimal trajectories (encouraged) and make explicit where you need to backtrack to a previously visited node. Brevity is not important here, rather it would be great if you can write out your entire search trajectory in a structured manner.
4. Conclude with the final result in EXACTLY this format:
```
SOLUTION: YES/NO
OPERATIONS: [list of operations performed]
RESULT: final_value
```

For example, a successful solution might end with:
```
SOLUTION: YES
OPERATIONS: ['75-72=3', '3*2=6', '66/6=11']
RESULT: 11
```
Note that each string in the list of OPERATIONS involves only 1 operation each. For example, 'A+B=C' is allowed, 'A+B/C=D' is not allowed because that involves 2 operations in the string.

An unsuccessful attempt might end with:
```
SOLUTION: NO
OPERATIONS: []
RESULT: None
```

Could you provide a detailed reasoning trace for making {TARGET} using the numbers {INITIAL_NUMBERS}? 
When you have reached a solution during the search that arrives at the target, please double check if the operations
1. use all the initial numbers once, 
2. used any numbers not in the list at all,
3. applied the arithmetics correctly. 
Otherwise, reflect on all mistakes and repeat the search process. 
Once you've reached the target and the result passes your check, report the final answer in the format given. The evaluation code will specifically look for these exact keywords and format in the final lines to extract the solution details.
"""

o3_sft_prompt_template = """
You'll be playing the countdown game using search strategies. In the Countdown game, we start with a set of four initial numbers (e.g., [1,2,4,5]) and a target value (e.g., 14). Our goal is to use arithmetic operations (+,-,*,/) to produce the target using all of the numbers. 

We formalize this game as a search problem:
- States: Sets of remaining numbers
- Initial State: Set of initial numbers
- Goal State: Using all numbers to reach target value
- Actions: Choose two numbers, apply an operation (+, -, *, /), replace them with the result
Constraints:
- All 4 numbers must be used exactly once

I'd like you to generate detailed reasoning traces for this game that: 
1. Uses {SEARCH_TYPE} with a heuristic. Propose a heuristic you think is promising and explain why 
2. Go through the full search trajectory in a structured manner towards the target using that strategy, with articulated thought processes 
3. Includes suboptimal trajectories (encouraged) and make explicit where you need to backtrack to a previously visited node. Brevity is not important here, rather it would be great if you can write out your entire search trajectory in a structured manner.
4. Conclude with the final result in EXACTLY this format:
```
SOLUTION: YES/NO
OPERATIONS: [list of operations performed]
RESULT: final_value
```

For example, a successful solution might end with:
```
SOLUTION: YES
OPERATIONS: ['75-72=3', '3*2=6', '66/6=11']
RESULT: 11
```
Note that each string in the list of OPERATIONS involves only 1 operation each. For example, 'A+B=C' is allowed, 'A+B/C=D' is not allowed because that involves 2 operations in the string.

An unsuccessful attempt might end with:
```
SOLUTION: NO
OPERATIONS: []
RESULT: None
```

Could you provide a detailed reasoning trace for making {TARGET} using the numbers {INITIAL_NUMBERS}? 
When you have reached a solution during the search that arrives at the target, please double check if the operations
1. use all the initial numbers once, 
2. used any numbers not in the list at all,
3. applied the arithmetics correctly. 
Otherwise, reflect on all mistakes and repeat the search process. 
Once you've reached the target and the result passes your check, report the final answer in the format given. The evaluation code will specifically look for these exact keywords and format in the final lines to extract the solution details.
"""

eval_prompt_template = "Make {TARGET} with the numbers {INITIAL_NUMBERS}"