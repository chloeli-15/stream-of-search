import json
import os
import argparse
from tqdm import tqdm

# should only taken 30s if you have the files
def convert_to_sftrainer_format(file_prefix, input_dir, output_dir):
    data = {}
    
    template_names = os.listdir(input_dir)
    
    for template_name in template_names:
        file_path = os.path.join(input_dir, template_name, f"{file_prefix}_random_{template_name}.json")
        with open(file_path, "r") as f:
            data[template_name] = json.load(f)
    
    formatted_data = []
    
    n_samples = len(data["sos"])
    for i in tqdm(range(n_samples), desc=f"Processing {os.path.basename(file_path)}"):
        base_entry = data["sos"][i]
        
        user_prompt = f"Make {base_entry["target"]} with the numbers {base_entry["nums"]} using standard arithmetic operations.\n"
        base_entry["messages_optimal"] = [
                {"role": "user", "content": user_prompt}, 
                {"role": "assistant", "content": base_entry.pop('optimal_path')}
            ]
        
        for template_name in template_names:
            entry = data[template_name][i]
            user_prompt, search_path = entry.pop("search_path").split("===")
            base_entry[f"messages_{template_name}"] = [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": search_path}
                ]
        
        formatted_data.append(base_entry)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_json_path_search = os.path.join(output_dir, f"{file_prefix}.json")
    
    with open(output_json_path_search, "w+") as f:
        json.dump(formatted_data, f, indent=4)
    

def convert_to_hf(input_dir, output_dir):
    template_names = os.listdir()
    file_names = os.listdir(os.path.join(input_dir,"sos"))
    file_prefixes = [f.split("_random_")[0] for f in file_names]
    
    for file_prefix in file_prefixes:
        # if filename.endswith(".json"):
        # input_json_path = os.path.join(input_dir, filename)
        convert_to_sftrainer_format(file_prefix, input_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON data to SFTrainer format.")
    parser.add_argument("input_path", help="Path to the input JSON file or directory")
    parser.add_argument("output_dir", help="Directory to save the output files")
    args = parser.parse_args()
        
    if os.path.isdir(args.input_path):
        convert_to_hf(args.input_path, args.output_dir)
    else:
        raise ValueError("Not a directory")

