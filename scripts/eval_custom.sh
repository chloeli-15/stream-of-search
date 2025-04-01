#!/bin/bash

# This is just as example
cd ~/projects/sos/stream-of-search &&\
/cs/student/msc/ml/2024/ycheah/disk/miniconda3/envs/sos1/bin/python ./src/eval_custom.py\
    --adapter models/qwen-2.5-1.5B-instruct-sft-lora-countdown-search-1k\
    -n 8\
    --messages_field messages_sos\
    --batch_size 8\
    --ctx 4096\
    --gens 1\
    --chat_template True\
    --upload_results False