HEURISTIC_DESCRIPTIONS = {
    "sum_heuristic": (
        "**sum-based** heuristic, which computes the average of the absolute differences between each number and the target number.\n"
        # "For example, for nums=[4,2,1,1] and target=10, the heuristic score is:\n"
        # "(|4-10| + |2-10| + |1-10| + |1-10|)/4 = 8\n"
        "Lower scores indicate that, on average, the numbers are closer to the target, making the state more promising for reaching the target via additive operations.\n"
    ),
    "mult_heuristic": (
        "**product-based** heuristic, which computes the minimum of all absolute differences between each number and each factor of the target number. \n"
        "For example, for nums=[4,2,1,1] and target=10, the heuristic score is:\n"
        "min(|4-2|, |4-5|, |4-10|) + min(|2-2|, |2-5|, |2-10|) + min(|1-2|, |1-5|, |1-10|) + min(|1-2|, |1-5|, |1-10|) = 1 + 0 + 1 + 1 = 3\n"
        "Lower scores indicate that the numbers are closer to the target's factors, making them more promising for reaching the target via multiplication.\n"
    ),
}
SEARCH_DESCRIPTIONS = {
    "dfs": {
        "name": "Depth-First Search (DFS)",
        "description": (
            "- **Initialization**: We create a single root node containing the initial set of numbers and compute its heuristic score. This node serves as the starting point for our search.\n"
            "- **Nodes**: Each node represents a state defined by a set of numbers and the sequence of operations applied to reach that state.\n"
            "- **Node Selection for Expansion**: We select a node from our current branch to expand by applying the next arithmetic operation, diving deeper into that branch.\n"
            "- **Expansion**: For the current node, we generate child nodes by selecting two numbers and applying an arithmetic operation, resulting in a new set of n−1 numbers. We compute the heuristic score for each new node as we explore further.\n"
            "- **Goal Check**: If any node’s set contains exactly one number that matches the target, we have found a solution. Otherwise we backtrack and re-select a node for expansion.\n"
            "- **Backtracking and Termination**: If a branch does not lead to the target, we backtrack and explore alternative operations. The search terminates when a solution is found or when no more nodes are available to explore.\n"
            )
    },
    "bfs": {
        "name": "Breadth-First Search (BFS) with Beam Size {beam_size}",
        "description": (
        "- **Initialization**: We create a single root node containing the initial set of numbers. We compute its heuristic score and place this node in our 'frontier' or 'open set' of nodes to explore.\n"
        "- **Nodes**: Each node in our search graph represents a set of n numbers (and the operations used to reach them).\n"
        "- **Node Selection for Expansion**: At each level, we use the same heuristic to rank the nodes in our frontier. We keep only a fixed number of top-scoring nodes (the “beam width”) and discard the rest.\n"
        "- **Expansion**: For each selected node, we generate child nodes by selecting two numbers and applying an arithmetic operation, resulting in a new set of n−1 numbers. Each child node’s heuristic is then computed, and we keep the top {beam_size} children\n"
        "- **Goal Check**: If any node’s set contains exactly one number that equals the target, we have found a solution. Otherwise we backtrack and re-select a node for expansion.\n"
        "- **Termination**: If we run out of nodes to explore without finding the target, no solution exists.\n"
        )
    }
}

user_prompt = ( "Combine these initial numbers {nums} using only arithmetic operations (+, -, *, /) to reach the target value {target}. All initial numbers must be used exactly once.\n"
                "Conclude with the final result in EXACTLY this format:\n"
                "```\n"
                "SOLUTION: YES/NO\n"
                "OPERATIONS: list of string of operations performed, each string involving only 1 operation. For example, ['A+B=C','C+D=E'] is allowed, ['A+B+D=E'] is not allowed\n"
                "RESULT: final_value\n"
                "```\n")

TEXT_TEMPLATES = {
    "sos": {
        "prefix": ( "Combine these initial numbers {nums} using only arithmetic operations (+, -, *, /) to reach the target value {target}. All initial numbers must be used exactly once.\n"
                    "Conclude with the final result in EXACTLY this format:\n"
                    "```\n"
                    "SOLUTION: YES/NO\n"
                    "OPERATIONS: list of string of operations performed, each string involving only 1 operation. For example, ['A+B=C','C+D=E'] is allowed, ['A+B+D=E'] is not allowed\n"
                    "RESULT: final_value\n"
                    "```\n"
                    "==="),
        "current_state": "Current State: {target}:{nums}, Operations: {operations}\n",
        "operation_selection": "Exploring Operation: {operation}, Resulting Numbers: {nums}\n",
        "node_generation": "Generated Node #{node_idx}: {target}:{nums} Operation: {operation}\n",
        "move_to_node": "Moving to Node #{next_idx}\n",
        "move_to_node_bfs": "Moving to Node #{next_idx}\n",
        "no_solution":("{nums},{target} unequal: No Solution\n\n"),
        "no_solution_backtracking":("{nums},{target} unequal: No Solution\n\n"),
        "no_solution_backtracking_bfs":("{nums},{target} unequal: No Solution\n\n"),
        "solution_summary": (
            "```\n"
            "SOLUTION: {solution_found}\n"
            "OPERATIONS: {operations_str}\n"
            "RESULT: {final_value}\n"
            "```\n"
        )
    },
    "sos_react": {
        "prefix": (
        "Combine these initial numbers {nums} using only arithmetic operations (+, -, *, /) to reach the target value {target}. All initial numbers must be used exactly once.\n"
        "Conclude with the final result in EXACTLY this format:\n"
        "```\n"
        "SOLUTION: YES/NO\n"
        "OPERATIONS: list of string of operations performed, each string involving only 1 operation. For example, ['A+B=C','C+D=E'] is allowed, ['A+B+D=E'] is not allowed\n"
        "RESULT: final_value\n"
        "```\n"
        "==="
        "We have these numbers {nums}. Our goal is to produce {target}. Let's show the search process step by step:\n"
        "One possible strategy is to use a {search_name} guided by a {heuristic_description}\n\n"
        ),
        "current_state": (
            "Current State #{node_idx}: {target}:{nums}. Operations so far : {operations}.\n\n"
        ),
        "operation_selection": (
            "Thought    : Expand Node #{node_idx}’s {child_ordinal} child to build next frontier.\n"
            "Action     : Exploring operation {operation}. \n"
            "Resulting numbers: {nums}.\n"
            "{heuristic_arithmetic_line}"
        ),
        "node_generation": "Generated Node #{node_idx}: {target}:{nums} Operation: {operation}\n\n", 
        "move_to_node": (
            "Thought    : Selecting next node to expand based on heuristic value\n"
            "Action     : Moving to Node #{next_idx} with lowest heuristic value {heuristic_value}\n" 
        ),
        "move_to_node_bfs": (
            "Thought    : Completed expansion of previous level. Selecting next node to expand based on heuristic value\n"
            "Action     : Moving to Node #{next_idx} with lowest heuristic value {heuristic_value}\n" 
        ),
        "no_solution":("{nums},{target} unequal: No Solution\n\n"),
        "no_solution_backtracking":("{nums},{target} unequal: No Solution ... backtracking\n\n"),
        "no_solution_backtracking_bfs":("{nums},{target} unequal: No solution after expanding all children at current level ... backtracking \n\n"),
        "solution_summary": (
            "```\n"
            "SOLUTION: {solution_found}\n"
            "OPERATIONS: {operations_str}\n"
            "RESULT: {final_value}\n"
            "```\n"
        )
            # "Observation: Heuristic score = {heuristic_arithmetic_string}\n\n" 

    },
    #     "sos_explained_v2": {
    #     "prefix": (
    #     "## Countdown Game Explanation\n"
    #     "In the Countdown game, we start with a set of four initial numbers (e.g., [4,2,1,1]) and a target value (e.g., 10). Our goal is to use arithmetic operations (+,–,*,/) to produce the target. Each time we apply an operation to two numbers, we remove those two numbers from the set and replace them with the operation’s result. We continue until we:\n"
    #     "1. Obtain the target (success) and can provide a valid formula (e.g., 4*2+1+1=10), or\n"
    #     "2. Exhaust all possibilities without finding the target (failure).\n"
    #     "\n"
    #     "## {search_name}\n"
    #     "You will be using a {search_name} guided by a {heuristic_description}\n"
    #     "The search process:\n"
    #     "{search_description}\n"
    #     "\n"
    #     "Make {target} with the numbers {nums} using standard arithmetic operations, and show your search step by step\n"
    #     "==="
    #     "We are given initial numbers {nums} and target value {target}. Let's show the search process step by step:\n"
    #     ),
    #     "node_selection": (
    #         "Current State #{node_idx}: {target}:{nums}. Operations so far : {operations}.\n"
    #     ),
    #     "operation_selection": (
    #         "Generating Node #{node_idx}. Exploring operation: {operation}. Resulting numbers: {nums}.\n"
    #     ),
    #     "node_generation": "",  # Not needed in this mode.
    #     "backtracking": "Backtracking to Node #{next_index}\n"
    # }
}

SOS_TRAJECTORY_PROMPT = """
Hi! You'll be providing material to conduct supervised fine-tuning on a small model to learn how to play the countdown game using search strategies. In the Countdown game, we start with a set of four initial numbers (e.g., [1,2,4,5]) and a target value (e.g., 14). Our goal is to use arithmetic operations (+,-,*,/) to produce the target using all of the numbers. 

We formalize this game as a search problem:
- States: Sets of remaining numbers
- Initial State: Set of initial numbers
- Goal State: Using all numbers to reach target value
- Actions: Choose two numbers, apply an operation (+, -, *, /), replace them with the result
Constraints:
- All 4 numbers must be used exactly once

I'd like you to generate detailed reasoning traces for this game that: 
1. Uses breadth first search with a heuristic. Propose a heuristic you think is promising and explain why 
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

COUNTDOWN_PROMPT = """
Combine these initial numbers {INITIAL_NUMBERS}, using only arithmetic operations (+, -, *, /) to reach the target value {TARGET}. All initial numbers must be used exactly once. The correct combination must exist.

Conclude with the final result in EXACTLY this format:
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
The evaluation code will specifically look for these exact format in the final lines to extract the solution details.
"""