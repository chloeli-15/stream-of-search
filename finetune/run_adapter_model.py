#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import logging

#%%
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(adapter_path, base_model=None):
    """Load a QLoRA fine-tuned model from Hugging Face"""
    # Get base model name from adapter config if not provided
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model = base_model or peft_config.base_model_name_or_path
    logger.info(f"Using base model: {base_model}")
    
    # Set up 4-bit quantization (required for QLoRA compatibility)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Load base model with quantization
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load and apply adapter weights
    logger.info("Applying QLoRA adapters...")
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
            # temperature=temperature,
            # top_p=0.9,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

#%%
if __name__ == "__main__":
    # Example usage
    adapter_path = "chloeli/qwen-2.5-7b-instruct-sft-qlora-countdown-search-1k"  # Directory with adapter_model.safetensors
    
    # Load model
    model, tokenizer = load_model(adapter_path)
    
    # print(tokenizer.chat_template)
#%%
    # Generate text
    prompt = "Make 11 with the numbers [75, 2, 72, 66] using standard arithmetic operations."
    response = generate(model, tokenizer, prompt)
    
    print(f"\nPrompt: {prompt}")
    print(f"\nResponse: {response}")

# %%
