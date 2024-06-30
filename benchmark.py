import os
import time
import torch
import gc
import subprocess
import logging
import threading
from torch import autocast
from diffusers import StableDiffusion3Pipeline
from mlperf_logging.mllog import mllogger
from mlperf_logging import mllog
import argparse
from huggingface_hub import login

# Log in to Hugging Face
login()

# Function to use the authentication token
def use_auth_token():
    return "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"

# Function to calculate cost
def calculate_cost(duration_seconds, cost):
    return (duration_seconds / 3600) * cost

# Function to fetch the total GPU memory using nvidia-smi
def fetch_total_memory():
    gpu_output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
        stderr=subprocess.STDOUT
    ).decode('utf-8').strip().split('\n')
    return int(gpu_output[0])

# Function to continuously log GPU metrics in a separate thread
class GPUMonitor(threading.Thread):
    def __init__(self, total_memory):
        super().__init__()
        self.running = True
        self.gpu_utilization = 0
        self.gpu_memory_utilization = 0
        self.gpu_memory_used = 0
        self.gpu_memory_free = 0
        self.gpu_temperature = 0
        self.gpu_power_draw = 0
        self.total_memory = total_memory

    def run(self):
        while self.running:
            gpu_output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
                stderr=subprocess.STDOUT
            ).decode('utf-8').strip().split('\n')
            for gpu in gpu_output:
                metrics = [x.strip() for x in gpu.split(',')]
                gpu_util = int(metrics[0])
                mem_used = int(metrics[1])
                mem_free = int(metrics[2])
                temp = float(metrics[3])
                power_draw = float(metrics[4])

                mem_util = (mem_used / self.total_memory) * 100

                if gpu_util > self.gpu_utilization:
                    self.gpu_utilization = gpu_util
                if mem_util > self.gpu_memory_utilization:
                    self.gpu_memory_utilization = mem_util
                if mem_used > self.gpu_memory_used:
                    self.gpu_memory_used = mem_used
                if mem_free > self.gpu_memory_free:
                    self.gpu_memory_free = mem_free
                if temp > self.gpu_temperature:
                    self.gpu_temperature = temp
                if power_draw > self.gpu_power_draw:
                    self.gpu_power_draw = power_draw

            time.sleep(1)

    def stop(self):
        self.running = False

# Function to log monitoring information
def log_monitoring_info(start_time, count, image_size, costs, server_name, gpu_monitor):
    end_time = time.time()
    duration = end_time - start_time
    ondemand_cost = calculate_cost(duration, costs["ondemand"])
    reserved_one_year_cost = calculate_cost(duration, costs["reserved_one_year"])
    reserved_three_year_cost = calculate_cost(duration, costs["reserved_three_year"])
    spot_cost = calculate_cost(duration, costs["spot"])
    metrics = {
        "duration": duration,
        "gpu_utilization": gpu_monitor.gpu_utilization,
        "gpu_memory_utilization": gpu_monitor.gpu_memory_utilization,
        "gpu_memory_used": gpu_monitor.gpu_memory_used,
        "gpu_memory_free": gpu_monitor.gpu_memory_free,
        "gpu_temperature": gpu_monitor.gpu_temperature,
        "gpu_power_draw": gpu_monitor.gpu_power_draw,
        "ondemand_cost": ondemand_cost,
        "reserved_one_year_cost": reserved_one_year_cost,
        "reserved_three_year_cost": reserved_three_year_cost,
        "spot_cost": spot_cost,
        "image_size": image_size,
        "instance_type": server_name
    }
    logging.info(f"Metrics for image {count}: {metrics}")
    mllogger.end(key='image_generation', value=metrics)

# Function to generate an image from text
def text2Image(model, prompt, count, server_name, costs, total_memory):
    if not os.path.exists(server_name):
        os.makedirs(server_name)
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(model, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Start monitoring before image generation
    start_time = time.time()

    # Start GPU monitoring thread
    gpu_monitor = GPUMonitor(total_memory)
    gpu_monitor.start()

    try:
        inference_start_time = time.time()
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=28,
                height=1024,
                width=1024,
                guidance_scale=7.0,
            ).images[0]
        inference_duration = time.time() - inference_start_time
        image_size = image.size
    except Exception as e:
        mllogger.end(key='image_generation', value={"error": str(e)})
        print(f"Error generating image: {e}")
        gpu_monitor.stop()
        return

    image_path = os.path.join(server_name, f"{count}.png")
    image.save(image_path)

    # Stop GPU monitoring thread
    gpu_monitor.stop()
    gpu_monitor.join()

    # Log monitoring info after image generation
    log_monitoring_info(start_time, count, image_size, costs, server_name, gpu_monitor)
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description='Stable Diffusion Benchmarking')
    parser.add_argument('--server_name', type=str, required=True, help='Name of the server')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--ondemand_cost', type=float, required=True, help='On-demand hosting cost per hour')
    parser.add_argument('--reserved_one_year_cost', type=float, required=True, help='Reserved one-year hosting cost per hour')
    parser.add_argument('--reserved_three_year_cost', type=float, required=True, help='Reserved three-year hosting cost per hour')
    parser.add_argument('--spot_cost', type=float, required=True, help='Spot hosting cost per hour')
    args = parser.parse_args()

    server_name = args.server_name
    model_name = args.model_name
    costs = {
        "ondemand": args.ondemand_cost,
        "reserved_one_year": args.reserved_one_year_cost,
        "reserved_three_year": args.reserved_three_year_cost,
        "spot": args.spot_cost
    }
    
    # Fetch total memory dynamically
    total_memory = fetch_total_memory()

    if not os.path.exists(server_name + '/log'):
        os.makedirs(server_name + '/log')
    mllog.config(filename=server_name + '/log/mlperf_log.txt')
    logging.basicConfig(filename=server_name + '/image_generation.log', level=logging.INFO)

    # Write the PID to a file in /tmp
    with open('/tmp/stable_diffusion.pid', 'w') as f:
        f.write(str(os.getpid()))

    prompts = [
        "A beautiful landscape with mountains and rivers",
        "A futuristic cityscape at night",
        "A serene beach with crystal clear water",
        "A majestic forest in autumn",
        "A vibrant market street in a foreign country",
        "A sci-fi spaceship exploring a distant galaxy",
        "A mystical castle surrounded by clouds",
        "A cute puppy playing in the park",
        "A bustling city street at rush hour",
        "A tranquil Japanese garden with cherry blossoms",
        "A sunset over the ocean with sailboats",
        "A magical forest with glowing plants",
        "A dragon flying over a medieval town",
        "A bustling market in an ancient city",
        "A spaceship landing on an alien planet",
        "A portrait of a futuristic cyborg",
        "A fantasy warrior in shining armor",
        "A peaceful village in the mountains",
        "A steampunk city with flying machines",
        "A cozy cottage in a snowy forest",
        "A vast desert with towering sand dunes",
        "A lively street fair with colorful stalls",
        "A knight fighting a dragon in a fiery battle",
        "A serene lake surrounded by autumn trees",
        "A futuristic train traveling through a neon-lit city",
        "A mysterious forest with hidden treasures",
        "A majestic eagle soaring over the Grand Canyon",
        "A charming small town during Christmas",
        "A sci-fi laboratory with advanced technology",
        "A vibrant coral reef teeming with sea life",
        "A peaceful meadow with wildflowers in bloom",
        "A majestic lion roaming the savannah",
        "A rustic farmhouse in a rolling countryside",
        "A bustling harbor with ships and seagulls",
        "A mystical cave with glowing crystals",
        "A colorful hot air balloon festival",
        "A tranquil river winding through a forest",
        "A futuristic robot city",
        "A serene monastery in the mountains",
        "A beautiful butterfly garden",
        "A grand cathedral with stained glass windows",
        "A tropical rainforest with exotic animals",
        "A snowy mountain peak under a starry sky",
        "A quaint village by the sea",
        "A magnificent palace with lush gardens",
        "A wild west town with cowboys",
        "A magical unicorn in an enchanted forest",
        "A bustling train station in the 1920s",
        "A serene lagoon with tropical fish",
        "A dramatic volcanic eruption with lava flows"
    ]

    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)} for prompt: '{prompt}'")
        try:
            text2Image(model_name, prompt, i+1, server_name, costs, total_memory)
            print(f"Image {i+1} saved.")
        except RuntimeError as e:
            print(f"Failed to generate image {i+1}: {e}")
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    main()
