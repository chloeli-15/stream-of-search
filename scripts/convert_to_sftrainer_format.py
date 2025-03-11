import json
import os
import argparse
from tqdm import tqdm

# should only taken 30s if you have the files
def convert_to_sftrainer_format(input_json_path, output_dir):
    with open(input_json_path, "r") as f:
        data = json.load(f)
    
    formatted_data_search = []
    formatted_data_optimal = []
    
    for entry in tqdm(data, desc=f"Processing {os.path.basename(input_json_path)}"):
        user_prompt = f"Make {entry['target']} with the numbers {entry['nums']} using standard arithmetic operations."

        formatted_entry_search = {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": entry['search_path']}
            ]
        }
        
        formatted_entry_optimal = {
            "messages": [
                {"role": "user", "content": user_prompt}, 
                {"role": "assistant", "content": entry['optimal_path']}
            ]
        }
        
        formatted_data_search.append(formatted_entry_search)
        formatted_data_optimal.append(formatted_entry_optimal)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(input_json_path)
    output_json_path_search = os.path.join(output_dir, f"{base_filename}_search.json")
    output_json_path_optimal = os.path.join(output_dir, f"{base_filename}_optimal.json")
    
    with open(output_json_path_search, "w+") as f:
        json.dump(formatted_data_search, f, indent=4)
    
    with open(output_json_path_optimal, "w+") as f:
        json.dump(formatted_data_optimal, f, indent=4)

def convert_directory(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_json_path = os.path.join(input_dir, filename)
            convert_to_sftrainer_format(input_json_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON data to SFTrainer format.")
    parser.add_argument("input_path", help="Path to the input JSON file or directory")
    parser.add_argument("output_dir", help="Directory to save the output files")
    args = parser.parse_args()
    
    if os.path.isdir(args.input_path):
        convert_directory(args.input_path, args.output_dir)
    else:
        raise ValueError("Not a directory")

