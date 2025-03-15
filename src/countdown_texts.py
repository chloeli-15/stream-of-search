HEURISTIC_DESCRIPTIONS = {
    "sum_heuristic": (
        "**sum-based** heuristic, which computes the average of the absolute differences between each number and the target number"
        "If there is only one number, it returns the absolute difference between that number and the target." 
        "Lower scores indicate that, on average, the numbers are closer to the target, making the state more promising for reaching the target via additive operations."
    ),
    "mult_heuristic": (
        "**product-based** heuristic, which computes the minimum of all absolute differences between each number and each factor of the target number. "
        "Lower scores indicate that the numbers are closer to the target's factors, making them more promising for reaching the target via multiplication."
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

TEXT_TEMPLATES = {
    "sos": {
        "prefix": "",
        "node_selection": "Current State: {target}:{nums}, Operations: {operations}\n",
        "operation_selection": "Exploring Operation: {operation}, Resulting Numbers: {nums}\n",
        "node_generation": "Generated Node #{node_idx}: {target}:{nums} Operation: {operation}\n",
        "backtracking": "Moving to Node #{next_index}\n"
    },
    "sos_explained_v1": {
        "prefix": (
        "## Countdown Game Explanation\n"
        "In the Countdown game, we start with a set of four initial numbers (e.g., [1,2,4,5]) and a target value (e.g., 14). Our goal is to use arithmetic operations (+,–,*,/) to produce the target. Each time we apply an operation to two numbers, we remove those two numbers from the set and replace them with the operation’s result. We continue until we:\n"
        "1. Obtain the target (success) and can provide a valid formula (e.g., 2*4+5+1=14), or\n"
        "2. Exhaust all possibilities without finding the target (failure).\n"
        "\n"
        "## {search_name}\n"
        "Our approach is a {search_name} guided by a {heuristic_description}\n"
        "The search process:\n"
        "{search_description}\n"
        "\n"
        "We are given initial numbers {nums} and target value {target}. Let's illustrate the search process step by step:\n"
        ),
        "node_selection": (
            "Current State #{node_idx}: {target}:{nums}. Operations so far : {operations}. Heuristic score: {score}, \n"
        ),
        "operation_selection": (
            "Generating Node #{node_idx}. Operation: {operation}. Resulting numbers: {nums}. Heuristic score: {score} \n"
        ),
        "node_generation": "",  # Not needed in this mode.
        "backtracking": "Backtracking to Node #{next_index}\n"
    }
}


