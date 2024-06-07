import os
import time
import torch
import gc
import subprocess
import logging
from torch import autocast
from diffusers import StableDiffusionPipeline
from mlperf_logging.mllog import mllogger
from mlperf_logging import mllog
import argparse

def use_auth_token():
    return "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"

def calculate_cost(duration_seconds, cost):
    return (duration_seconds / 3600) * cost

def log_gpu_metrics():
    try:
        gpu_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
            stderr=subprocess.STDOUT
        ).decode('utf-8').strip().split('\n')

        gpu_metrics = []
        for gpu in gpu_output:
            metrics = [x.strip() for x in gpu.split(',')]
            gpu_metrics.append({
                'gpu_utilization': metrics[0],
                'gpu_memory_utilization': metrics[1],
                'gpu_memory_total': metrics[2],
                'gpu_memory_used': metrics[3],
                'gpu_memory_free': metrics[4],
                'gpu_temperature': metrics[5],  # GPU temperature
                'gpu_power_draw': metrics[6]  # Power draw in watts
            })
        return gpu_metrics
    except subprocess.CalledProcessError as e:
        logging.error(f"Error collecting GPU metrics: {e.output.decode('utf-8')}")
        return None
    except Exception as e:
        logging.error(f"Error collecting GPU metrics: {e}")
        return None

def log_monitoring_info(start_time, start_gpu_metrics, count, image_size, costs):
    end_time = time.time()

    gpu_metrics = log_gpu_metrics()
    if gpu_metrics is None:
        gpu_metrics = start_gpu_metrics

    duration = end_time - start_time

    ondemand_cost = calculate_cost(duration, costs["ondemand"])
    reserved_one_year_cost = calculate_cost(duration, costs["reserved_one_year"])
    reserved_three_year_cost = calculate_cost(duration, costs["reserved_three_year"])
    spot_cost = calculate_cost(duration, costs["spot"])

    metrics = {
        "duration": duration,
        "gpu_metrics": gpu_metrics,
        "ondemand_cost": ondemand_cost,
        "reserved_one_year_cost": reserved_one_year_cost,
        "reserved_three_year_cost": reserved_three_year_cost,
        "spot_cost": spot_cost,
        "image_size": image_size
    }

    mllogger.end(key='image_generation', value=metrics)

    logging.info(f"Image {count} generated in {duration:.2f} seconds. Metrics: {metrics}")

    print(f"Image {count} generated in {duration:.2f} seconds.")
    print(f"GPU metrics: {gpu_metrics}")
    print(f"On Demand Cost of image generation: ${ondemand_cost:.6f}")
    print(f"Reserved for one year Cost of image generation: ${reserved_one_year_cost:.6f}")
    print(f"Reserved for three year Cost of image generation: ${reserved_three_year_cost:.6f}")
    print(f"Spot instance Cost of image generation: ${spot_cost:.6f}")
    print(f"Image size: {image_size}")

def text2Image(model, prompt, count, server_name, costs):
    if not os.path.exists(server_name):
        os.makedirs(server_name)
    
    mllogger.start(key='model_loading', value={"model": model})
    start_model_loading_time = time.time()
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            use_auth_token=use_auth_token()
        ).to("cuda")
    except Exception as e:
        mllogger.end(key='model_loading', value={"duration": time.time() - start_model_loading_time, "error": str(e)})
        print(f"Error loading model: {e}")
        return
    
    model_loading_duration = time.time() - start_model_loading_time
    mllogger.end(key='model_loading', value={"duration": model_loading_duration})
    print(f"Model loaded in {model_loading_duration:.2f} seconds.")
    
    # Start monitoring before image generation
    start_time = time.time()
    start_gpu_metrics = log_gpu_metrics()
    
    mllogger.start(key='image_generation', value={"model": model, "prompt": prompt, "count": count})
    
    try:
        inference_start_time = time.time()
        with autocast("cuda"):
            image = pipe(prompt)["images"][0]
        inference_duration = time.time() - inference_start_time
        image_size = image.size
    except Exception as e:
        mllogger.end(key='image_generation', value={"error": str(e)})
        print(f"Error generating image: {e}")
        return
    
    image_path = os.path.join(server_name, f"{count}.png")
    image.save(image_path)
    
    # Log monitoring info after image generation
    log_monitoring_info(start_time, start_gpu_metrics, count, image_size, costs)
    
    mllogger.end(key='inference_latency', value={"duration": inference_duration})
    logging.info(f"Inference latency for image {count}: {inference_duration:.2f} seconds.")

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
        "A vibrant coral reef teeming with sea life"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)} for prompt: '{prompt}'")
        try:
            text2Image(model_name, prompt, i+1, server_name, costs)
            print(f"Image {i+1} saved.")
        except RuntimeError as e:
            print(f"Failed to generate image {i+1}: {e}")
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()
