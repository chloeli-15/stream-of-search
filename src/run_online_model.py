import time, sys, os
import concurrent.futures
import numpy as np
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from openai import OpenAI


class OnlineLM:
    """Online language model using API services."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model = model_name  # Store model name as the model identifier
        self.temperature = kwargs.get("temperature", 0.7)  # Default temperature
        self._initialize()
        
    def _initialize(self):
        """Initialize the OpenAI client."""
        if "DEEPINFRA_TOKEN" not in os.environ:
            raise ValueError("DEEPINFRA_TOKEN environment variable is not set")
            
        self.openai = OpenAI(
            api_key=os.environ["DEEPINFRA_TOKEN"],
            base_url="https://api.deepinfra.com/v1/openai",
        )
    
    def _fetch_response(self, message_data):
        """Fetch a response from the API."""
        messages, max_tokens= message_data
        
        try:
            chat_completion = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature if self.temperature > 0 else 0,
                # top_k=20 if self.temperature>0.0 else None,
                # top_p=0.8 if self.temperature>0.0 else None,                
            )
            content = chat_completion.choices[0].message.content
            
            return content
        except Exception as e:
            return f"Error: {str(e)}", None
    
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
            
            for i, (content, logits) in enumerate(api_results):
                if repeat_input:
                    # Append generation to the original message
                    new_message = input_messages[i].copy()
                    new_message[-1] = new_message[-1].copy()
                    new_message[-1]['content'] += content
                    results.append((new_message, logits))
                else:
                    results.append((content, logits))
        
        # For API compatibility, return both results and a metadata object
        metadata = {"model": self.model, "online": True}
        time.sleep(0.02)
        return results, metadata


if __name__ == "__main__":
    # Example usage
    # model = OnlineLM("deepseek-ai/DeepSeek-R1")
    model = OnlineLM("DeekSeek-R1-Distill-Llama-70B")
    from datasets import load_dataset
    data_all = load_dataset("MelinaLaimon/stream-of-search")
    data = data_all["test"].select(range(100))
    
    data = data.map(lambda x: { # type: ignore
        'test_prompt': [
            # {'role': 'system', 'content': SYSTEM_PROMPT},
            x["messages_sos"]["role"=="user"]
        ],
        # 'answer': extract_hash_answer(x['answer'])
    })
    
    results, metadata = model.generate(data['test_prompt'], max_new_tokens=10)
    
    # for ds_batched in data.iter(batch_size=100):
        # results, metadata = model.generate(ds_batched['test_prompt'], max_new_tokens=4096)
    
    for result in results:
        print(result)  # Print the generated content