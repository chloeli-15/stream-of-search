#!/usr/bin/env python3

import json
import subprocess
import concurrent.futures
import argparse
import time
import os
from datetime import datetime
from archive.ssh_test import look_for_gpu

def run_evaluation(hostname, model_name, messages_field, num_samples=256, 
                   dataset="MelinaLaimon/stream-of-search", temp=0.7, 
                   batch_size=64, ctx=8192, gens=1, experiment_name=None, wandb_project="stream-of-search", 
                   wandb_entity="yeok-c", log_dir="logs"):
    """
    Run evaluation for a specific model on a specific host
    """
    if experiment_name is None:
        experiment_name = f"{model_name.split('/')[-1]}-eval"
        # experiment_name = f"{model_name.split('/')[-1]}-eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{hostname}_{model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    
    # Construct the SSH command
    cmd = [
        "ssh", "-t", hostname,
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
    
    print(f"Starting evaluation on {hostname}: {model_name}")
    print("Command:", " ".join(cmd))
    
    try:
        # Run the command and capture output
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                " ".join(cmd), 
                shell=True,
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
            print(f"❌ Error during evaluation for {model_name} on {hostname}")
            return False, hostname, model_name
            
    except Exception as e:
        print(f"❌ Exception during evaluation for {model_name} on {hostname}: {str(e)}")
        return False, hostname, model_name

def main():
    parser = argparse.ArgumentParser(description="Distributed model evaluation across hosts")
    parser.add_argument("--config", type=str, default="scripts/eval_config.json", help="Path to config JSON file")
    parser.add_argument("--n", type=int, default=32, help="Number of samples to evaluate")
    parser.add_argument("--dataset", type=str, default="MelinaLaimon/stream-of-search", help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--ctx", type=int, default=8192, help="Context length")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    parser.add_argument("--gens", type=int, default=1, help="Number of generations")
    parser.add_argument("--project", type=str, default="stream-of-search", help="W&B project name")
    parser.add_argument("--entity", type=str, default="yeokch", help="W&B entity name")
    
    args = parser.parse_args()
    
    # Get available hosts with GPUs
    hostnames = look_for_gpu()
    print(f"Found {len(hostnames)} hosts with GPUs: {', '.join(hostnames)}")
    
    # Load the configuration file
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    models_data = config.get('models_messages_field_pairs', {})
    print(f"Found {len(models_data)} models to evaluate")
    
    # Create a queue of evaluation tasks
    evaluation_queue = []
    for model_name, messages_field in models_data.items():
        evaluation_queue.append((model_name, messages_field))
    
    # Create a timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = f"distributed-eval-{timestamp}"
    
    # Run evaluations concurrently
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hostnames)) as executor:
        futures = {}
        host_index = 0
        
        # Submit initial batch of tasks
        for i in range(min(len(hostnames), len(evaluation_queue))):
            model_name, messages_field = evaluation_queue[i]
            hostname = hostnames[host_index]
            futures[executor.submit(
                run_evaluation, 
                hostname, 
                model_name, 
                messages_field,
                args.n,
                args.dataset,
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