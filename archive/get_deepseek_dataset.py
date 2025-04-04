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
    def _fetch_response(self, message_data):
        """Fetch a response from the API with exponential backoff retry."""
        messages, max_tokens = message_data
        
        try:
            chat_completion = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature if self.temperature > 0 else 0     
            )
            reasoning_content = chat_completion.choices[0].message.reasoning_content
            content = chat_completion.choices[0].message.content
            
            return reasoning_content, content
        except Exception as e:
            print(f"API error: {str(e)}")
            raise  # Let backoff handle the retry
    
    def generate(self, 
                input_messages: List[Dict[str, Any]], 
                max_new_tokens: int = 100, 
                repeat_input: bool = False) -> Tuple[List[Any], Any]:
        """Generate text using the API."""
        
        # Prepare batch of requests
        request_data = []
        for messages in input_messages:
            request_data.append((messages, max_new_tokens))
        
        # Process in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            api_results = list(executor.map(self._fetch_response, request_data))
            
            for i, content in enumerate(api_results):
                if repeat_input:
                    # Append generation to the original message
                    new_message = input_messages[i].copy()
                    new_message[-1] = new_message[-1].copy()
                    new_message[-1]['content'] += content
                    results.append(new_message)
                else:
                    results.append(content)
        
        # For API compatibility, return both results and a metadata object
        metadata = {"model": self.model, "online": True}
        # Add proper rate limiting
        time.sleep(0.5)  # More reasonable rate limiting for API calls
        return results, metadata    

def process_and_save_batch(model, batch, output_file, max_tokens):
    """Process a batch of inputs and save results"""
    responses, metadata = model.generate(
        input_messages=batch['messages_deepseek'],
        max_new_tokens=max_tokens,
        repeat_input=True,
    )
    
    # Save responses to file
    with open(output_file, 'a') as f:
        for response in responses:
            f.write(json.dumps(response) + '\n')
    
    return len(batch['messages_deepseek'])

def main():
    parser = argparse.ArgumentParser(description='Generate responses from DeepSeek API')
    parser.add_argument('--api_token', type=str, default="sk-9a60aa6a5edf47488fc809f7afce5a99",
                        help='API token for authentication')
    parser.add_argument('--output_file', type=str, default="./res/output.jsonl",
                        help='Path to output file')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size for API requests')
    parser.add_argument('--max_tokens', type=int, default=8192,
                        help='Maximum tokens for generation')
    parser.add_argument('--api_base_url', type=str, default="https://api.deepseek.com/v1",
                        help='Base URL for API')
    parser.add_argument('--model', type=str, default="deepseek-reasoner",
                        help='Model name')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Temperature for generation')
    
    args = parser.parse_args()

    # Initialize the model
    model = OnlineLM(args.model, args.api_token, args.api_base_url, temperature=args.temperature)
    api = HfApi()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load the dataset
    print("Loading dataset from Hugging Face...")
    data = load_dataset("MelinaLaimon/stream-of-search", split="train[90%:100%]")

    data = data.map(lambda message:{
            'messages_deepseek': [{'content': message['messages_sos'][0]['content'] + 
                                 "\nNote that the solution does exist. Verify your solutions before your present your final results and backtrack to correct mistakes from before your mistakes if you have to.", 
                                 'role': 'user'}]
        }
    )

    # Process batches with progress bar
    total_processed = 0
    with tqdm(total=len(data)) as progress_bar:
        for batch in data.iter(batch_size=args.batch_size):
            batch_size = process_and_save_batch(model, batch, args.output_file, args.max_tokens)
            total_processed += batch_size
            progress_bar.update(batch_size)
            print(f"Progress: {total_processed}/{len(data)} messages processed.")

    print(f"Done! All {total_processed} messages processed and saved to {args.output_file}")

if __name__ == '__main__':
    main()