export task=sft
export model_name=qwen-2.5
export ACCELERATE_LOG_LEVEL=info 

accelerate launch --report_to 1b --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_{task}.py recipes/{model_name}/{task}/config_qlora.yaml