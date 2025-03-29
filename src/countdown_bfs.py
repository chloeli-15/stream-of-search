import heapq
import itertools
import random

import tiktoken

from countdown_utils import combine_nums, CountdownNode, sum_heuristic, mult_heuristic, metric_fn, generate_mult_heuristic_string, generate_sum_heuristic_string
from countdown_texts import TEXT_TEMPLATES, HEURISTIC_DESCRIPTIONS, SEARCH_DESCRIPTIONS
import inflect

def bfs(target, nums, beam_size, heuristic=sum_heuristic, text_template_name="sos"):
    p = inflect.engine()
    text_template = TEXT_TEMPLATES.get(text_template_name)
    heuristic_description = HEURISTIC_DESCRIPTIONS.get(heuristic.__name__)
    generate_heuristic_arithmetic_string = generate_mult_heuristic_string if heuristic.__name__ == "mult_heuristic" else generate_sum_heuristic_string
    search_name = SEARCH_DESCRIPTIONS.get("bfs")["name"]\
                    .format(beam_size=beam_size)
    search_description = SEARCH_DESCRIPTIONS.get("bfs")["description"]\
                    .format(beam_size=beam_size)

    search_trace = text_template["prefix"].format(
            target=target
            , nums=nums
            , heuristic_description=heuristic_description
            , search_name=search_name
            , search_description=search_description) 
    
    open_set = []
    # Push the initial node with its index, heuristic value, and parent index
    heapq.heappush(open_set, (heuristic(nums, target), CountdownNode(0, None, nums, [], heuristic(nums, target))))
    node_index = 1  # Initialize node indexing 

    while open_set:
        # Get the top beam width nodes based on heuristic (pruning others)
        current_nodes = [heapq.heappop(open_set) for _ in range(min(beam_size, len(open_set)))]
        if not current_nodes:
            break  # Exit if no nodes are left to expand

        for idx, (score, current_node) in enumerate(current_nodes):
            heuristic_arithmetic_string = generate_heuristic_arithmetic_string(current_node.nums, target)
            text = text_template["current_state"].format(
                target=target,
                nums=current_node.nums,
                operations=current_node.operations,
                # heuristic_arithmetic_line=f"Heuristic score : {heuristic_arithmetic_string}\n" if len(new_node.nums) > 1 else "",            # For explained mode.
                node_idx=current_node.idx  # For explained mode.
            )
            search_trace += text
            # Store nodes generated at this level for pruning
            generated_nodes = []
            # Generate successors for each node
            for i, j in itertools.combinations(range(len(current_node.nums)), 2):
                for result, operation in combine_nums(current_node.nums[i], current_node.nums[j]):
                    new_nums = [current_node.nums[k] for k in range(len(current_node.nums)) if k != i and k != j] + [result]
                    new_operations = current_node.operations + [operation]
                    new_heuristic = heuristic(new_nums, target)
                    new_node = CountdownNode(node_index, current_node, new_nums, new_operations, new_heuristic)
                    generated_nodes.append((new_heuristic, new_node))  # Add to generated nodes
                    node_index += 1

            # Explicit Pruning: Keep only top 'beam_size' nodes, prune the rest
            generated_nodes.sort()  # Sort by heuristic value
            pruned_nodes = generated_nodes[beam_size:]  # Nodes that will be pruned
            generated_nodes = generated_nodes[:beam_size]  # Nodes that will be kept

           # shuffle the generated nodes so that the first node explored is not the same every time
            random.shuffle(generated_nodes)
            # Add remaining nodes back to open_set
            node_idx = 0
            for heuristic_val, node in generated_nodes:

                node.idx = f"{node.parent.idx},{node_idx}"
                heuristic_arithmetic_string = generate_heuristic_arithmetic_string(node.nums, target)
                child_ordinal = p.ordinal(node_idx+1)
                op_text = text_template["operation_selection"].format(
                    node_idx=node.parent.idx,
                    child_ordinal=child_ordinal,
                    target=target,
                    nums=node.nums,
                    operation=node.operations[-1],
                    heuristic_arithmetic_line=f"Heuristic score : {heuristic_arithmetic_string}\n" if len(new_node.nums) > 1 else ""
                )
                search_trace += op_text 

                if len(node.nums) == 1 and node.nums[0] == target:
                    search_trace += f"{node.nums[0]},{target} equal: Goal Reached\n\n"
                    text = text_template["solution_summary"].format(solution_found="YES", operations_str=new_node.operations, final_value=new_node.nums[0])
                    search_trace += text 
                    return search_trace
                elif len(node.nums) == 1 and node_idx == len(generated_nodes)-1:
                    search_trace += text_template["no_solution_backtracking_bfs"].format(nums=node.nums[0], target=target)
                elif len(node.nums) == 1:
                    search_trace += text_template["no_solution"].format(nums=node.nums[0], target=target)
                else:
                    gen_text = text_template["node_generation"].format(
                        node_idx=node.idx,
                        target=target,
                        nums=node.nums,
                        operation=node.operations[-1]
                    )
                    search_trace += gen_text
                node_idx += 1
            generated_nodes.sort()
            for node_tuple in generated_nodes:
                heapq.heappush(open_set, node_tuple)
 
            # Note transition to the next node within the current set
            if idx < len(current_nodes) - 1:
                heuristic_value, next_node = current_nodes[idx + 1]
                text = text_template["move_to_node_bfs"].format(
                    next_idx=next_node.idx,
                    heuristic_value=heuristic_value
                )
                search_trace += text
            

        # Backtracking trace
        if current_nodes:
            # Find the next node to backtrack to (the next in the open set)
            next_node_to_explore = open_set[0] if open_set else None
            if next_node_to_explore:
                heuristic_value, next_node = next_node_to_explore
                text = text_template["move_to_node_bfs"].format(
                    next_idx=next_node.idx,
                    heuristic_value=heuristic_value
                )
                search_trace += text

    # search_trace += "No solution found."
    text = text_template["solution_summary"].format(solution_found="NO", operations_str=[], final_value=None)
    search_trace += text 
    return search_trace

if __name__ == "__main__":
    # Example usage
    random.seed(4)
    target =  16
    nums = [48, 15, 5]
    search_path = bfs(target, nums, 5, heuristic=sum_heuristic, text_template_name="sos")
    print(search_path)
    print(len(search_path))
    print(metric_fn(search_path))

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(search_path)
    print(f"token length: {len(tokens)}")