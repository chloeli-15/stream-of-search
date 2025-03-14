#!/bin/bash

conda activate sos

cd src

# model_name = "Qwen/Qwen-2.5-0.5B-instruct"
adapter_name = "~/.cache/huggingface/hub/models--chloeli--qwen-2.5-0.5b-instruct-sft-qlora-countdown-search-1k/"
dataset_path = "chloeli/stream-of-search-countdown-10k"


python eval_custom.py --adapter ${adapter_name}-n 1000 -o 0 -d ${dataset_path} --temperature 0. --batch_size 64 --data_dir "" --gens 1