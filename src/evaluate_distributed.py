#!/usr/bin/env python3

import json
import subprocess
import concurrent.futures
import argparse
import time
import os
from datetime import datetime
from archive.ssh_test import look_for_gpu, stop_all_processes

def run_evaluation(hostname, model_name, messages_field, num_samples=256, 
                   dataset="MelinaLaimon/stream-of-search", temp=0.7, 
                   batch_size=64, ctx=8192, gens=1, experiment_name=None, wandb_project="stream-of-search", 
                   wandb_entity="yeok-c", log_dir="logs"):
    """
    Run evaluation for a specific model on a specific host
    """
    if experiment_name is None:
        experiment_name = f"{model_name.split('/')[-1]}-eval"
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{model_name.split('/')[-1]}_{hostname}.log")
    
    print(f"Starting evaluation on {hostname}: {model_name} with {num_samples} samples and datsaet {dataset}")
    
    try:
        # Open log file for writing
        with open(log_file, 'w') as f:
            # First, run nvidia-smi to capture GPU state before evaluation
            f.write(f"=== GPU INFO BEFORE EVALUATION ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
            nvidia_cmd = f"ssh {hostname} nvidia-smi"
            nvidia_process = subprocess.run(
                nvidia_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Write nvidia-smi output to log
            f.write(nvidia_process.stdout)
            f.write("\n\n=== STARTING EVALUATION ===\n\n")
            f.flush()
            
            # Construct the SSH command for evaluation
            # Use ServerAliveInterval to keep the connection alive and remove -t flag
            # Also add nohup to ensure the command continues to run even if the SSH session is terminated
            cmd = [
                "ssh",
                "-o", "ServerAliveInterval=60",
                hostname,
                f"cd ~/projects/sos/stream-of-search && "
                f"/cs/student/msc/ml/2024/ycheah/disk/miniconda3/envs/sos1/bin/python ./src/eval_custom.py "
                f"--adapter '{model_name}' "
                f"-n {num_samples} "
                f"--dataset_name '{dataset}' "
                f"--messages_field '{messages_field}' "
                f"--temperature {temp} "
                f"--batch_size {batch_size} "
                f"--ctx {ctx} "
                f"--gens {gens} "
                f"--chat_template True "
                f"--experiment_name '{experiment_name}' "
                f"--upload_results True "
                f"--wandb_project '{wandb_project}' "
                f"--wandb_entity '{wandb_entity}'"
            ]
            
            print("Command:", " ".join(cmd))
            
            # Write the start time, hostname and cmd to log
            f.write(f"=== START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"=== HOSTNAME: {hostname} ===\n")
            f.write(f"=== COMMAND: {' '.join(cmd)} ===\n")
            
            # Run the evaluation command and capture output
            process = subprocess.Popen(
                cmd,  # Use list format instead of shell=True for better control
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
            print(f"✅ Evaluation completed for {model_name} on {hostname}")
            return True, hostname, model_name
        else:
            print(f"❌ Error during evaluation for {model_name} on {hostname}, return code: {process.returncode}. Logfile: {log_file}")
            return False, hostname, model_name
            
    except Exception as e:
        print(f"❌ Exception during evaluation for {model_name} on {hostname}: {str(e)}. Logfile: {log_file}")
        return False, hostname, model_name
    
def main():
    parser = argparse.ArgumentParser(description="Distributed model evaluation across hosts")
    parser.add_argument("--config", type=str, default="scripts/eval_config.json", help="Path to config JSON file")
    # parser.add_argument("--n", type=int, default=8, help="Number of samples to evaluate")
    # parser.add_argument("--dataset", type=str, default="MelinaLaimon/stream-of-search", help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--ctx", type=int, default=16384, help="Context length")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    parser.add_argument("--gens", type=int, default=1, help="Number of generations")
    parser.add_argument("--project", type=str, default="stream-of-search", help="W&B project name")
    parser.add_argument("--entity", type=str, default="yeokch", help="W&B entity name")
    parser.add_argument("--hosts", type=str, default="1", help="Host flag in look_for_gpus")
    parser.add_argument("--kill", type=bool, default=False, help="Kill all processes on all hosts")
    args = parser.parse_args()
    
    stop_all_processes(args.hosts)
    # Get available hosts with GPUs
    hostnames = look_for_gpu(args.hosts)
    
    print(f"Found {len(hostnames)} hosts with GPUs: {', '.join(hostnames)}")
    
    # Load the configuration file
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    models_data = config.get('models_messages_field_pairs', {})
    
    print(f"Found {len(models_data)} models to evaluate")
    
    # Create a queue of evaluation tasks
    evaluation_queue = []
    for i, (model_name, messages_field, nums, dataset) in models_data.items():
        evaluation_queue.append(((model_name, messages_field, nums, dataset)))
    
    # Create a timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = f"{timestamp}-distributed-eval"
    
    # Run evaluations concurrently
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hostnames)) as executor:
        futures = {}
        host_index = 0
        
        # Submit initial batch of tasks
        for i in range(min(len(hostnames), len(evaluation_queue))):
            model_name, messages_field, nums, dataset = evaluation_queue[i]
            hostname = hostnames[host_index]
            futures[executor.submit(
                run_evaluation, 
                hostname, 
                model_name, 
                messages_field,
                nums, # args.n,
                dataset, # args.dataset,
                args.temp,
                args.batch_size,
                args.ctx,
                args.gens,
                f"{experiment_name}-{model_name.split('/')[-1]}",
                args.project,
                args.entity
            )] = (hostname, model_name)
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
                success, hostname, model_name = future.result()
                results.append((success, hostname, model_name))
                del futures[future]
                
                # If there are more tasks in the queue, assign them to the free host
                if queue_index < len(evaluation_queue):
                    next_model, next_prompt = evaluation_queue[queue_index]
                    futures[executor.submit(
                        run_evaluation, 
                        hostname, 
                        next_model, 
                        next_prompt,
                        args.n,
                        args.dataset,
                        args.temp,
                        args.batch_size,
                        args.ctx,
                        args.gens,
                        f"{experiment_name}-{next_model.split('/')[-1]}",
                        args.project,
                        args.entity
                    )] = (hostname, next_model)
                    queue_index += 1
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Total models: {len(evaluation_queue)}")
    successful = sum(1 for success, _, _ in results if success)
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    
    if len(results) - successful > 0:
        print("\nFailed models:")
        for success, hostname, model_name in results:
            if not success:
                print(f"  - {model_name} (on {hostname})")
    
    print("\nResults will be available in W&B project:")
    print(f"https://wandb.ai/{args.entity}/{args.project}")
    

if __name__ == "__main__":
    main()
