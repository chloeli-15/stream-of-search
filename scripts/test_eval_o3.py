import json
import pandas as pd
import constants


def main(num_o3_trajectories = 1000):
    o3_trajectories = []
    with open(constants.o3_results_filepath) as file:
        for line in file:
            response = json.loads(line.rstrip())
            o3_trajectory = response["response"]["body"]["output"][-1]["content"][0]["text"]
            o3_trajectories.append(o3_trajectory)
            
    with open(constants.merged_sos_database_filename) as file: 
        hf_database = json.load(file)


    results = []
    for i in range(num_o3_trajectories):
        nums = hf_database[i]["nums"]
        target = hf_database[i]["target"]
        search_type = constants.SEARCH_TYPES[hf_database[i]["search_type"]]
        user_prompt = constants.eval_prompt_template.format(
            INITIAL_NUMBERS=nums,
            TARGET=target)
        # old_trajectory = hf_database[i]["messages_sos"][1]["content"]
        o3_trajectory = o3_trajectories[i]
        results.append(evaluate_countdown_trajectory(user_prompt, o3_trajectory))
    
    results = pd.DataFrame(results)
    return results.solved.sum() / len(results)
    