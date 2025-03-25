
from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl
import torch

import os
import sys

from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime
import json
import logging

# Add the project root to the path (two levels up from scripts)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now this will find src/result_parsers/countdown_trajectories.py at the project root
from src.result_parsers.countdown_trajectories import evaluate_countdown_trajectories

class DebugCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:  # Just check on first step
            print("Available in kwargs:", list(kwargs.keys()))
            trainer = kwargs.get("trainer")
            if trainer:
                print("Trainer attributes:", [attr for attr in dir(trainer) 
                                             if not attr.startswith('_')])
            return control
        
class CountdownEvalCallback(TrainerCallback):
    "A callback that runs the Countdown game eval"
    def __init__(self, 
                tokenizer,
                eval_steps: int=50,
                num: int=20, 
                temperature: float=0.0, 
                batch_size: int=10, 
                ctx: int=4096, 
                gens: int=1, 
                chat_template: bool=True):
        
        self.eval_steps = eval_steps
        self.tokenizer=tokenizer
        self.gens = gens # best of gen trials
        self.num = num 
        self.temperature = temperature # temperature for the model
        self.batch_size = batch_size
        self.ctx = ctx
        self.chat_template = chat_template

        self.results = None

    def generate_batch(self, model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
        """
        Generate text using the loaded model    
        Takes input str after chat template has been applied  
        """
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, padding_side='left').to(model.device)
        
        # Generate with sampling
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature>0.0
            )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def eval_batch(self, model, tokenizer, data, batch_size=128, context_len=4096, temperature=0.0, n=1):
        """
        Evaluate the model on the data using a sliding window so that the context length is not exceeded
        """
        output_texts_concat = []    

        for i, data_batch in tqdm(enumerate(data.iter(batch_size=batch_size)), total=len(data)//batch_size):   
            # tokenize and generate data_batch['test_prompt']. Input is a list of dicts with role
            if self.chat_template:
                chat_inputs = tokenizer.apply_chat_template(data_batch["test_prompt"], return_tensors="pt", padding=True, truncation=True, max_length=context_len, return_length=True, tokenize=False)
            else:
                chat_inputs = data_batch["test_prompt"]["content"] # if no chat template 
            outputs = self.generate_batch(model, tokenizer, chat_inputs, max_new_tokens=context_len, temperature=temperature)
            output_texts_concat.extend(outputs)

        return output_texts_concat

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Evaluate the model on the data after every n steps
        """
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
                
            # Get model directly from kwargs, not from trainer
            model = kwargs.get("model")
            if model is None:
                print("Model not found in kwargs")
                return control
            
            # Use the tokenizer that was passed during initialization
            tokenizer = self.tokenizer

            try:
                logger = logging.getLogger(__name__)
                logger.info(f"Running Countdown evaluation at step {state.global_step}")
                
                data_all = load_dataset("chloeli/stream-of-search-countdown-10k", split="val_search")
                results_all_trials = []

                for trial in range(self.gens):
                    data = data_all.select(range(self.num))
                    tokenizer.padding_side = "left"

                    data = data.map(lambda x: { # type: ignore
                        'test_prompt': [
                            # {'role': 'system', 'content': SYSTEM_PROMPT},
                            x['messages']["role"=="user"]
                        ],
                        # 'answer': extract_hash_answer(x['answer'])
                    })

                    results = []
                    completions = self.eval_batch(model, tokenizer, data, batch_size=self.batch_size, context_len=self.ctx, temperature=self.temperature, n=self.gens)
                    for i in range(len(data['test_prompt'])):
                        results.append({
                            'prompt': data['test_prompt'][i][0]['content'],
                            'completion': completions[i]
                        })
                    results_all_trials.append(results)

                eval_results = evaluate_countdown_trajectories(results_all_trials)
                self.results = eval_results
                logger.info(f"Countdown results:\n{eval_results[0]}")

                # Save results to disk
                save_path = os.path.join("results", "countdown_eval")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # Get model name from output dir
                model_name = os.path.basename(args.output_dir)
                timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
                results_file = os.path.join(
                    save_path, 
                    f"{model_name}_step{state.global_step}_{timenow}.json"
                )
                
                with open(results_file, "w") as f:
                    json.dump(eval_results, f, indent=4)

                logger.info(f"Countdown evaluation complete. Results saved to {results_file}")
                
            except Exception as e:
                import traceback
                print(f"Error in Countdown evaluation: {str(e)}")
                print(traceback.format_exc())
        
        return control
        
