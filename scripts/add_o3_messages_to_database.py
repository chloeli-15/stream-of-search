import json
import constants


SEARCH_TYPES = constants.SEARCH_TYPES

o3_trajectories = []

def main():
    with open(constants.o3_results_filepath) as file:
        for line in file:
            response = json.loads(line.rstrip())
            o3_trajectory = response["response"]["body"]["output"][-1]["content"][0]["text"]
            o3_trajectories.append(o3_trajectory)
            
    with open(constants.merged_sos_database_filename) as file: 
        hf_database = json.load(file)
        
    for i in range(len(hf_database)):
        nums = hf_database[i]["nums"]
        target = hf_database[i]["target"]
        search_type = SEARCH_TYPES[hf_database[i]["search_type"]]
        user_prompt = constants.o3_sft_prompt_template.format(
            SEARCH_TYPE=search_type, 
            INITIAL_NUMBERS=nums,
            TARGET=target)
        
        # We've only generated 1000 o3 responses
        if i < 1000:
            o3_trajectory = o3_trajectories[i]
        else: 
            o3_trajectory = ""
            
        hf_database[i]["messages_o3"] = [
            {"role": "user", "content": user_prompt}, 
            {"role": "assistant", "content": o3_trajectory}
        ]
        
    with open(constants.merged_sos_o3_database_filename, "w+") as file:
        json.dump(hf_database, file, indent=4)