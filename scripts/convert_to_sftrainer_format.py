#%%
import json
import os
import argparse
from tqdm import tqdm

#%%
# should only taken 30s if you have the files
def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)
    
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
    
    base_filename = os.path.splitext(os.path.basename(input_json_path))[0]
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

#%%
def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

#%%
original_data=load_json("/Users/chloeli/Library/CloudStorage/GoogleDrive-chloeli561@gmail.com/My Drive/UCL/Term 2/Openendedness/stream-of-search/scripts/train1_b4_t100_n500000_random.json")[:10000]

search_data=load_json("/Users/chloeli/Library/CloudStorage/GoogleDrive-chloeli561@gmail.com/My Drive/UCL/Term 2/Openendedness/stream-of-search/scripts/train1_b4_t100_n500000_random_optimal.json")[:10000]

for i in range(len(original_data)):
    original_data[i]['messages'] = search_data[i]['messages']

#%%
with open("/Users/chloeli/Library/CloudStorage/GoogleDrive-chloeli561@gmail.com/My Drive/UCL/Term 2/Openendedness/stream-of-search/scripts/train1_b4_t100_n10000_random_optimal.json", "w+") as f:
    json.dump(original_data, f, indent=4)

#%%
train_search_data=load_json("/Users/chloeli/Library/CloudStorage/GoogleDrive-chloeli561@gmail.com/My Drive/UCL/Term 2/Openendedness/stream-of-search/scripts/train1_b4_t100_n10000_random_optimal.json")

#%%
# Make a HF dataset
from datasets import Dataset, DatasetDict
train_search =  Dataset.from_dict({k: [d[k] for d in train_search_data] for k in train_search_data[0].keys()})

# %%
# dataset_dict = DatasetDict({
#     "train_optimal": train_search
# })

dataset_dict["train_optimal"] = train_search

dataset_dict.push_to_hub("chloeli/stream-of-search-countdown-10k")

# %%
original_test_data = load_json("/Users/chloeli/Library/CloudStorage/GoogleDrive-chloeli561@gmail.com/My Drive/UCL/Term 2/Openendedness/stream-of-search/scripts/val1_b4_t100_n500000_random.json")

test_search_data = load_json("/Users/chloeli/Library/CloudStorage/GoogleDrive-chloeli561@gmail.com/My Drive/UCL/Term 2/Openendedness/stream-of-search/scripts/val1_b4_t100_n500000_random_search.json")

for i in range(len(original_test_data)):
    original_test_data[i]['messages'] = test_search_data[i]['messages']
#%%
test_search =  Dataset.from_dict({k: [d[k] for d in original_test_data] for k in original_test_data[0].keys()})
dataset_dict["val_search"] = test_search
# %%
dataset_dict.push_to_hub("chloeli/stream-of-search-countdown-10k")

# %%
