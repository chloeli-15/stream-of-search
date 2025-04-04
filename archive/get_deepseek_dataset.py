import time
import sys
import os
import concurrent.futures
import argparse
from typing import Optional, List, Dict, Any, Tuple
from openai import OpenAI
from huggingface_hub import HfApi
from tqdm import tqdm
import json
from datasets import load_dataset
import backoff  # Add backoff for robust API error handling
import threading
import queue
import uuid
from collections import defaultdict

class OnlineLM:
    """Online language model using API services."""
    
    def __init__(self, model_name: str, api_token: str, api_base_url: str, **kwargs):
        self.model = model_name  # Store model name as the model identifier
        self.temperature = kwargs.get("temperature", 0)  # Default temperature
        self.api_token = api_token
        self.api_base_url = api_base_url
        self._initialize()
        
    def _initialize(self):
        """Initialize the OpenAI client."""
        self.openai = OpenAI(
            api_key=self.api_token,
            base_url=self.api_base_url,
        )
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def _fetch_response(self, message, max_tokens):
        """Fetch a response from the API with exponential backoff retry."""
        try:
            chat_completion = self.openai.chat.completions.create(
                model=self.model,
                messages=message,
                max_tokens=max_tokens,
                temperature=self.temperature if self.temperature > 0 else 0     
            )
            reasoning_content = chat_completion.choices[0].message.reasoning_content
            content = chat_completion.choices[0].message.content
            
            return reasoning_content, content
        except Exception as e:
            print(f"API error: {str(e)}")
            raise  # Let backoff handle the retry

def worker(task_queue, output_file, result_dict, pbar, model, max_tokens, lock):
    """Worker function for processing tasks"""
    while True:
        try:
            # Get a task from the queue
            task = task_queue.get(block=False)
            if task is None:  # None is our signal to stop
                task_queue.put(None)  # Put it back for other workers
                break
                
            idx, message = task
            message_id = str(uuid.uuid4())  # Generate a unique ID for this message
            
            # Process the request
            lm = model  # Use the shared model instance
            try:
                reasoning_content, content = lm._fetch_response(message, max_tokens)
                
                # Create response with original message
                new_message = message.copy()
                response_message = {
                    'role': 'assistant',
                    'content': content,
                    'reasoning_content': reasoning_content
                }
                result = {
                    'id': message_id, 
                    'original_idx': idx,
                    'messages': new_message + [response_message]
                }
                
                # Write to file immediately
                with lock:
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(result) + '\n')
                
                # Store mapping for later alignment
                with lock:
                    result_dict[idx] = message_id
                    pbar.update(1)
                
            except Exception as e:
                print(f"Error processing task {idx}: {str(e)}")
                # Put failed tasks back in the queue after a delay
                time.sleep(1)
                task_queue.put(task)
                
            # Small delay to control API rate
            time.sleep(0.1)
            
        except queue.Empty:
            break
            
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate responses from DeepSeek API')
    parser.add_argument('--api_token', type=str, default="sk-9a60aa6a5edf47488fc809f7afce5a99",
                        help='API token for authentication')
    parser.add_argument('--output_file', type=str, default="./res/output.jsonl",
                        help='Path to output file')
    parser.add_argument('--max_workers', type=int, default=100,
                        help='Number of concurrent workers')
    parser.add_argument('--max_tokens', type=int, default=8192,
                        help='Maximum tokens for generation')
    parser.add_argument('--api_base_url', type=str, default="https://api.deepseek.com/v1",
                        help='Base URL for API')
    parser.add_argument('--model', type=str, default="deepseek-reasoner",
                        help='Model name')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Temperature for generation')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a checkpoint file to resume from')
    
    args = parser.parse_args()

    # Initialize the model
    model = OnlineLM(args.model, args.api_token, args.api_base_url, temperature=args.temperature)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Create a mapping file path for remapping results later
    mapping_file = os.path.join(os.path.dirname(args.output_file), "mapping.json")
    
    # Load the dataset
    print("Loading dataset from Hugging Face...")
    data = load_dataset("MelinaLaimon/stream-of-search", split="train[0%:80%]")

    # Prepare the messages
    print("Preparing messages...")
    messages = []
    for idx, example in enumerate(data):
        message = [{
            'content': example['messages_sos'][0]['content'] + 
                     "\nNote that the solution does exist. Verify your solutions before your present your final results and backtrack to correct mistakes from before your mistakes if you have to.",
            'role': 'user'
        }]
        messages.append((idx, message))
    
    # Resume logic
    completed_tasks = {}
    if args.resume_from and os.path.exists(args.resume_from) and os.path.exists(mapping_file):
        print(f"Resuming from {args.resume_from}")
        with open(mapping_file, 'r') as f:
            completed_tasks = json.load(f)
        print(f"Found {len(completed_tasks)} completed tasks")
    
    # Create a task queue
    task_queue = queue.Queue()
    
    # Add tasks to the queue (skip completed ones if resuming)
    skipped = 0
    for idx, message in messages:
        if str(idx) not in completed_tasks:
            task_queue.put((idx, message))
        else:
            skipped += 1
            
    print(f"Skipped {skipped} already completed tasks")
    print(f"Added {task_queue.qsize()} tasks to the queue")
    
    # Add sentinel values to stop workers
    for _ in range(args.max_workers):
        task_queue.put(None)
    
    # Create a thread lock for file access
    lock = threading.Lock()
    
    # Setup progress bar
    total_tasks = len(messages) - skipped
    pbar = tqdm(total=total_tasks, desc="Processing requests")
    
    # Store results for remapping
    result_dict = defaultdict(str)
    # Add already completed tasks
    for idx, message_id in completed_tasks.items():
        result_dict[int(idx)] = message_id
    
    # Create and start worker threads
    print(f"Starting {args.max_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for _ in range(args.max_workers):
            future = executor.submit(
                worker, task_queue, args.output_file, result_dict, 
                pbar, model, args.max_tokens, lock
            )
            futures.append(future)
        
        # Wait for all workers to complete
        concurrent.futures.wait(futures)
    
    pbar.close()
    
    # Save the mapping for potential future resumption
    with open(mapping_file, 'w') as f:
        json.dump(result_dict, f)
    
    print(f"Done! All tasks processed and saved to {args.output_file}")
    print(f"Mapping saved to {mapping_file} for future reference")
    
    # Optional: Create an aligned version with original dataset order
    print("Creating aligned dataset...")
    aligned_file = args.output_file.replace('.jsonl', '_aligned.jsonl')
    
    # Load all results
    results = {}
    with open(args.output_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            results[item['id']] = item
    
    # Write aligned results
    with open(aligned_file, 'w') as f:
        for idx in range(len(messages)):
            if str(idx) in result_dict:
                message_id = result_dict[str(idx)]
                if message_id in results:
                    f.write(json.dumps(results[message_id]) + '\n')
    
    print(f"Aligned dataset saved to {aligned_file}")

if __name__ == '__main__':
    main()