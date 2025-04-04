#!/usr/bin/env python3

import json
import subprocess
import concurrent.futures
import argparse
import time
import os
from datetime import datetime
from archive.ssh_test import look_for_gpu

def run_training(hostname, config_file, task="sft", model_name="qwen-2.5", 
                num_processes=1, experiment_name=None, wandb_project="stream-of-search", 
                wandb_entity="yeok-c", log_dir="training_logs"):
    """
    Run training for a specific model config on a specific host
    """
    if experiment_name is None:
        experiment_name = f"{config_file.split('/')[-1].replace('.yaml','')}-training"
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{config_file.split('/')[-1]}_{hostname}.log")
    
    print(f"Starting training on {hostname} with config: {config_file}")
    
    try:
        # Open log file for writing
        with open(log_file, 'w') as f:
            # First, run nvidia-smi to capture GPU state before training
            f.write(f"=== GPU INFO BEFORE TRAINING ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
            nvidia_cmd = f"nvidia-smi"
            if hostname != "local":
                nvidia_cmd = f"ssh {hostname} nvidia-smi"
            else:
                nvidia_cmd = f"nvidia-smi"
            
            nvidia_process = subprocess.run(
                nvidia_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Write nvidia-smi output to log
            f.write(nvidia_process.stdout)
            f.write("\n\n=== STARTING TRAINING ===\n\n")
            f.flush()
            
            cmd = [
                "ssh",
                "-o", "ServerAliveInterval=60",
                hostname,
                "bash -c '"
                f"cd ~/projects/sos/stream-of-search && "
                f"source /cs/student/msc/ml/2024/ycheah/disk/miniconda3/bin/activate sos1 && "
                f"export task={task} && "
                f"export model_name={model_name} && "
                f"export ACCELERATE_LOG_LEVEL=info && "
                f"export config_file_path={config_file} && "
                f"export WANDB_PROJECT={wandb_project} && "
                f"export WANDB_ENTITY={wandb_entity} && "
                f"accelerate launch --config_file ./finetune/recipes/accelerate_configs/multi_gpu.yaml "
                f"--num_processes={num_processes} ./finetune/scripts/run_${{task}}.py ${{config_file_path}}"
                "'"
            ]
            
            if hostname == "local":
                # For local execution, use shell=True with a properly formatted command
                cmd = " ".join(cmd[4:])
                
                # Run the training command and capture output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    shell=True  # This is key for interpreting the bash command
                )
            else:
                print("Command:", " ".join(cmd))
                
                # Run the training command and capture output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
            
            # Stream and log output in real-time
            for line in process.stdout:
                f.write(line)
                f.flush()
                
            process.wait()
            
        if process.returncode == 0:
            print(f"✅ Training completed for config {config_file} on {hostname}")
            return True, hostname, config_file
        else:
            print(f"❌ Error during training for config {config_file} on {hostname}, return code: {process.returncode}. Logfile: {log_file}")
            return False, hostname, config_file
            
    except Exception as e:
        print(f"❌ Exception during training for config {config_file} on {hostname}: {str(e)}. Logfile: {log_file}")
        return False, hostname, config_file
import glob
def main():
    parser = argparse.ArgumentParser(description="Distributed model training across hosts")
    parser.add_argument("--config_folder", type=str, default="./finetune/recipes/qwen-2.5/sft", help="Path to folder containing all the json files")
    parser.add_argument("--task", type=str, default="sft", help="Training task (sft, dpo, etc.)")
    parser.add_argument("--model-name", type=str, default="qwen-2.5", help="Base model name")
    parser.add_argument("--num-processes", type=int, default=1, help="Number of processes for accelerate")
    parser.add_argument("--project", type=str, default="stream-of-search-train", help="W&B project name")
    parser.add_argument("--entity", type=str, default="yeokch", help="W&B entity name")
    parser.add_argument("--train_locally", type=str, default=False, help="Run training locally instead of on remote hosts")
    
    args = parser.parse_args()
    
    if args.train_locally:
        print("Running training locally...")
        hostnames = ["local"]
    else:
        # Get available hosts with GPUs
        hostnames = look_for_gpu("1")
        print(f"Found {len(hostnames)} hosts with GPUs: {', '.join(hostnames)}")
    
    config_files = glob.glob(os.path.join(args.config_folder, "*.yaml"))
    # rearrange the config_files so all the ones containing "1k" are first, then the rest
    config_files = sorted(config_files, key=lambda x: "1k" in x, reverse=True)
    print(f"Found {len(config_files)} training configs to run")
    print("Training configs:", config_files)
    
    # # Load the configuration file
    # with open(args.config, 'r') as f:
    #     config = json.load(f)
    
    # config_files = config.get('training_configs', [])
    # print(f"Found {len(config_files)} training configs to run")
    
    # Create a timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = f"{timestamp}-distributed-training"
    
    # Run training concurrently
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hostnames)) as executor:
        futures = {}
        host_index = 0
        
        # Submit initial batch of tasks
        for i in range(min(len(hostnames), len(config_files))):
            config_file = config_files[i]
            hostname = hostnames[host_index]
            futures[executor.submit(
                run_training, 
                hostname, 
                config_file,
                args.task,
                args.model_name,
                args.num_processes,
                f"{experiment_name}-{config_file.split('/')[-1].replace('.yaml','')}",
                args.project,
                args.entity
            )] = (hostname, config_file)
            host_index = (host_index + 1) % len(hostnames)
        
        # Process remaining queue as hosts become available
        queue_index = len(hostnames)
        
        while futures:
            # Wait for the next task to complete
            done, not_done = concurrent.futures.wait(
                futures, 
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                success, hostname, config_file = future.result()
                results.append((success, hostname, config_file))
                del futures[future]
                
                # If there are more tasks in the queue, assign them to the free host
                if queue_index < len(config_files):
                    next_config = config_files[queue_index]
                    futures[executor.submit(
                        run_training, 
                        hostname, 
                        next_config,
                        args.task,
                        args.model_name,
                        args.num_processes,
                        f"{experiment_name}-{next_config.split('/')[-1].replace('.yaml','')}",
                        args.project,
                        args.entity
                    )] = (hostname, next_config)
                    queue_index += 1
    
    # Print summary
    print("\n=== Training Summary ===")
    print(f"Total configs: {len(config_files)}")
    successful = sum(1 for success, _, _ in results if success)
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    
    if len(results) - successful > 0:
        print("\nFailed training jobs:")
        for success, hostname, config_file in results:
            if not success:
                print(f"  - {config_file} (on {hostname})")
    
    print("\nResults will be available in W&B project:")
    print(f"https://wandb.ai/{args.entity}/{args.project}")

if __name__ == "__main__":
    main()