# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer
from transformers.trainer_utils import get_last_checkpoint

from accelerate import Accelerator
from huggingface_hub import list_repo_files
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from peft import LoraConfig, PeftConfig
from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl

from .configs import DataArguments, DPOConfig, ModelArguments, SFTConfig
from .data import DEFAULT_CHAT_TEMPLATE
from src.result_parsers.countdown_trajectories import evaluate_countdown_trajectories
from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime
import json
import logging
def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Dict[str, int] | None:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None


def get_quantization_config(model_args: ModelArguments) -> BitsAndBytesConfig | None:
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        if model_args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
            bnb_4bit_quant_storage=model_args.bnb_4bit_quant_storage,
        ).to_dict()
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        ).to_dict()
    else:
        quantization_config = None

    return quantization_config


def get_tokenizer(
    model_args: ModelArguments, data_args: DataArguments, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.model_name_or_path
            if model_args.tokenizer_name_or_path is None
            else model_args.tokenizer_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def get_peft_config(model_args: ModelArguments) -> PeftConfig | None:
    if model_args.use_peft is False:
        return None

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config


def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    try:
        # Try first if model on a Hub repo
        repo_files = list_repo_files(model_name_or_path, revision=revision)
    except (HFValidationError, RepositoryNotFoundError):
        # If not, check local repo
        repo_files = os.listdir(model_name_or_path)
    return "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files


def get_checkpoint(training_args: SFTConfig | DPOConfig) -> Path | None:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

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
                eval_steps: int=40,
                num: int=50, 
                temperature: float=0.0, 
                batch_size: int=64, 
                ctx: int=4096, 
                gens: int=3, 
                chat_template: bool=True):
        
        self.eval_steps = eval_steps

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
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, padding_side='right').to(model.device)
        
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

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, tokenizer, **kwargs):
        """
        Evaluate the model on the data after every n steps
        """
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            
            # Get model and tokenizer from the trainer
            trainer = kwargs.get("trainer")
            if trainer is None:
                print("Trainer not found in kwargs")
                return control
                
            model = trainer.model
            tokenizer = trainer.tokenizer

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

                # Log metrics to the trainer
                for metric, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        trainer.log({f"eval/countdown_{metric}": value})
                        print(f"eval/countdown_{metric}: {value}")

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
        
