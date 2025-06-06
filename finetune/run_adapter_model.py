#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import logging
import os, glob

from accelerate import infer_auto_device_map

#%%
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(adapter_path, base_model=None, use_quantization=False):
    """Load a QLoRA fine-tuned model from Hugging Face"""

    # Get base model name from adapter config if not provided
    # Look for model in local directory
    # if glob.glob(f"{adapter_path}") and glob.glob(f"{adapter_path}/adapter_config.json") == []:
    #     adapter_path = glob.glob(f"{adapter_path}/*/*/adapter_config.json")[0].split("/adapter_config.json")[0]
        
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model = base_model or peft_config.base_model_name_or_path
    logger.info(f"Using base model: {base_model}")
    
    # Load base model with or without quantization
    if use_quantization:
        logger.info("Loading base model with quantization...")
        # Set up 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,            
        )
        
        
    else:
        # Load base model without quantization
        logger.info("Loading base model without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Load and apply adapter weights
    logger.info("Applying LoRA adapters...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=1024, temperature=1.0):
    """Generate text using the loaded model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with sampling
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature>0.0 else None,
            top_p=0.9 if temperature>0.0 else None,
            top_k=20 if temperature>0.0 else None,
            do_sample=temperature>0.0
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_batch(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    """
    Generate text using the loaded model    
    Takes input str after chat template has been applied  
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, padding_side='right').to(model.device)
    
    # Generate with sampling
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature>0.0 else None,
            top_p=0.9 if temperature>0.0 else None,
            top_k=20 if temperature>0.0 else None,
            do_sample=temperature>0.0,
        )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def download_models():
    paths = [
        "chloeli/qwen-2.5-0.5b-instruct-sft-lora-countdown-search-1k",
        "chloeli/qwen-2.5-1.5b-instruct-sft-lora-countdown-search-1k",
        "chloeli/qwen-2.5-7b-instruct-sft-lora-countdown-search-1k",
    ]
    for path in paths:
        # load_model(path)
        # tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
            
    # for adapter_path in paths:
    #     peft_config = PeftConfig.from_pretrained(adapter_path)
    #     base_model = base_model or peft_config.base_model_name_or_path
    #     logger.info(f"Using base model: {base_model}")
        
    #     # Set up 4-bit quantization (required for QLoRA compatibility)
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_use_double_quant=True
    #     )
        
    #     # Load base model with quantization
    #     logger.info("Loading base model...")
    #     model = AutoModelForCausalLM.from_pretrained(
    #         base_model,
    #         quantization_config=quantization_config,
    #         device_map="auto",
    #         trust_remote_code=True
    #     )
        
    #     # Load and apply adapter weights
    #     logger.info("Applying QLoRA adapters...")
    #     model = PeftModel.from_pretrained(model, adapter_path)
        
    #     # Load tokenizer
    #     tokenizer = AutoTokenizer.from_pretrained(base_model)
    #     if tokenizer.pad_token is None:
    #         tokenizer.pad_token = tokenizer.eos_token
            
    #     del model
    #     del tokenizer
#%%
if __name__ == "__main__":
    # download_models()
    # Example usage
    adapter_path = "chloeli/qwen-2.5-1.5B-instruct-sft-lora-countdown-search-5k" #"chloeli/qwen-2.5-1.5B-instruct-sft-lora-countdown-search-react-5k"  # Directory with adapter_model.safetensors
    
    # Load model
    model, tokenizer = load_model(adapter_path)
        
    # model.eval()
    # model.bfloat16().cuda()
    # print(tokenizer.chat_template)
#%%
    # Generate text
    prompts = [
            # "Make 10 with the numbers [2,4,1,1] using standard arithmetic operations.",
               "Make 62 with the numbers [20, 54, 64, 44] using standard arithmetic operations.",
               "Make 96 with the numbers [50, 2, 48, 87] using standard arithmetic operations.",
               "Make 62 with the numbers [20, 54, 64, 44] using standard arithmetic operations.",
               "Make 87 with the numbers [50, 2, 48, 87] using standard arithmetic operations.",
                ]
    for prompt in prompts:
        message = [{"role": "user", "content": prompt}]
        
        formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        print(f"\nPrompt: {formatted_prompt}")
        response = generate(model, tokenizer, formatted_prompt, max_new_tokens=1024)
        print(f"\nResponse: {response}")
#%%
        
# # %%


# %%
