import itertools
import tiktoken

from countdown_utils import combine_nums, CountdownNode, sum_heuristic, mult_heuristic, metric_fn, generate_sum_heuristic_string, generate_mult_heuristic_string
from countdown_texts import TEXT_TEMPLATES, HEURISTIC_DESCRIPTIONS, SEARCH_DESCRIPTIONS


def dfs(target, nums, heuristic=sum_heuristic, threshold=None, open_set=[], search_trace="", text_template_name="sos"):
    
    text_template = TEXT_TEMPLATES.get(text_template_name)
    generate_heuristic_arithmetic_string = generate_mult_heuristic_string if heuristic.__name__ == "mult_heuristic" else generate_sum_heuristic_string
    if search_trace == "":
        search_name = SEARCH_DESCRIPTIONS.get("bfs")["name"]
        search_description = SEARCH_DESCRIPTIONS.get("bfs")["description"]
        heuristic_description = HEURISTIC_DESCRIPTIONS.get(heuristic.__name__)
        search_trace = text_template["prefix"].format(
            target=target
            , nums=nums
            , heuristic_description=heuristic_description
            , search_name=search_name
            , search_description=search_description) + "\n"
    
    if len(open_set) == 0:
        # Push the initial node with its index, heuristic value, and parent index
        open_set.append((heuristic(nums, target), CountdownNode(0, None, nums, [], heuristic(nums, target))))

    while open_set:
        # Sort open_set by heuristic value, then pop the best node (lowest heuristic)
        open_set.sort(key=lambda x: -x[0])
        score, current_node = open_set.pop()
        heuristic_arithmetic_string = generate_heuristic_arithmetic_string(current_node.nums, target)
        selection_text = text_template["node_selection"].format(
                target=target,
                nums=current_node.nums,
                operations=current_node.operations,
                heuristic_arithmetic_string=heuristic_arithmetic_string,             # For explained mode.
                node_idx=current_node.idx  # For explained mode.
            )
        search_trace += selection_text

        # Generate successors for the current node
        generated_nodes = []
        for i, j in itertools.combinations(range(len(current_node.nums)), 2):
            node_index = 0
            for result, operation in combine_nums(current_node.nums[i], current_node.nums[j]):
                new_nums = [current_node.nums[k] for k in range(len(current_node.nums)) if k != i and k != j] + [result]
                new_operations = current_node.operations + [operation]
                new_heuristic = heuristic(new_nums, target)
                new_node = CountdownNode(node_index, current_node, new_nums, new_operations, new_heuristic)
                generated_nodes.append((new_heuristic, new_node))  # Add to generated nodes

        kept_nodes = []
        for g in generated_nodes:
            if threshold is None or g[0] <= threshold:
                kept_nodes.append(g)
            else:
                continue
        generated_nodes = kept_nodes
        generated_nodes.sort()

        node_index = 0
        for g, (heuristic_val, new_node) in enumerate(generated_nodes):
            new_node.idx = f"{new_node.parent.idx},{node_index}"
            heuristic_arithmetic_string = generate_heuristic_arithmetic_string(new_node.nums, target)
            op_text = text_template["operation_selection"].format(
                node_idx=new_node.idx,
                target=target,
                nums=new_node.nums,
                operation=new_node.operations[-1],
                heuristic_arithmetic_string=heuristic_arithmetic_string
            )
            search_trace += op_text 

            if len(new_node.nums) == 1 and new_node.nums[0] == target:
                search_trace += f"{new_node.nums[0]},{target} equal: Goal Reached\n"
                return search_trace
            elif len(new_node.nums) == 1:
                search_trace += f"{new_node.nums[0]},{target} unequal: No Solution\n"
            else:
                gen_text = text_template["node_generation"].format(
                        node_idx=new_node.idx,
                        target=target,
                        nums=new_node.nums,
                        operation=new_node.operations[-1]
                    )
                search_trace += gen_text
                # search_trace += f"Generated Node #{new_node.idx}: {target}:{new_node.nums} Operation: {new_node.operations[-1]}\n"
                new_set = [(new_heuristic, new_node)]
                backtracking_text = text_template["backtracking"].format(
                    next_index=new_node.idx
                )
                search_trace += backtracking_text
                # search_trace += f"Moving to Node #{new_node.idx}\n"
                search_trace = dfs(target, nums, heuristic=heuristic, threshold=threshold, search_trace=search_trace, open_set=new_set)
                if "Goal Reached" in search_trace:
                    return search_trace
            node_index += 1
            if g < len(generated_nodes) - 1:
                next_index = new_node.parent.idx
                backtracking_text = text_template["backtracking"].format(
                    next_index=next_index
                )
                search_trace += backtracking_text
                heuristic_arithmetic_string = generate_heuristic_arithmetic_string(new_node.nums, target)
                op_text = text_template["operation_selection"].format(
                    node_idx=next_index,
                    target=target,
                    nums=new_node.nums,
                    operation=new_node.parent.operations,
                    heuristic_arithmetic_string=heuristic_arithmetic_string
                )
                search_trace += op_text 

                # search_trace += f"Moving to Node #{next_index}\n"
                # search_trace += f"Current State: {target}:{new_node.parent.nums}, Operations: {new_node.parent.operations}\n"

        # Backtracking trace
        if open_set:  # If there are still nodes to explore
            next_node = open_set[-1][1]  # Get the index of the next node to be explored
            next_index = next_node.idx
            search_trace += f"Moving to Node #{next_index}\n"

    return search_trace


if __name__ == "__main__":
    # Example usage
    target = 10
    nums = [80,4,1,1]
    search_path = dfs(target, nums, heuristic=mult_heuristic, threshold=target, text_template_name="sos_explained_v1")
    print(search_path)
    print(len(search_path))
    print(metric_fn(search_path))
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(search_path)
    print(f"token length: {len(tokens)}")