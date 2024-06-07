import os
import time
import torch
import gc
import psutil
import pynvml
from torch import autocast
from diffusers import StableDiffusionPipeline
from mlperf_logging.mllog import mllogger
from mlperf_logging import mllog

def use_auth_token():
    return "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"

SERVER_NAME = "g4dn.2xlarge"
if not os.path.exists(SERVER_NAME + '/log'):
    os.makedirs(SERVER_NAME + '/log')
mllog.config(filename=SERVER_NAME + '/log/mlperf_log.txt')

# Hosting cost per hour in dollars
ONDEMAND_HOSTING_COST_PER_HOUR = 0.7520
RESERVED_ONE_YEAR_HOSTING_COST_PER_HOUR = 0.4740
RESERVED_THREE_YEAR_HOSTING_COST_PER_HOUR = 0.3250
SPOT_HOSTING_COST_PER_HOUR = 0.3201

def init_gpu_monitoring():
    pynvml.nvmlInit()
    device_handles = []
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        device_handles.append(handle)
        print(f"Device {i}: {pynvml.nvmlDeviceGetName(handle)}")
    return device_handles

def calculate_cost(duration_seconds, cost):
    return (duration_seconds / 3600) * cost

def log_monitoring_info(start_time, start_cpu_usage, start_memory_usage, start_gpu_usage, start_gpu_temp, start_disk_io, start_net_io, device_handles, count, image_size):
    end_time = time.time()
    end_cpu_usage = psutil.cpu_percent(interval=None)
    end_memory_usage = psutil.virtual_memory().percent

    gpu_usages = []
    gpu_temps = []
    gpu_power_usages = []

    for handle in device_handles:
        gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts

        gpu_usages.append(gpu_usage.gpu)
        gpu_temps.append(gpu_temp)
        gpu_power_usages.append(gpu_power)

    avg_gpu_usage = sum(gpu_usages) / len(gpu_usages)
    avg_gpu_temp = sum(gpu_temps) / len(gpu_temps)
    avg_gpu_power_usage = sum(gpu_power_usages) / len(gpu_power_usages)

    cpu_usage_change = end_cpu_usage - start_cpu_usage
    memory_usage_change = end_memory_usage - start_memory_usage
    gpu_usage_change = avg_gpu_usage - start_gpu_usage
    duration = end_time - start_time

    disk_read_bytes = psutil.disk_io_counters().read_bytes - start_disk_io.read_bytes
    disk_write_bytes = psutil.disk_io_counters().write_bytes - start_disk_io.write_bytes
    net_sent_bytes = psutil.net_io_counters().bytes_sent - start_net_io.bytes_sent
    net_recv_bytes = psutil.net_io_counters().bytes_recv - start_net_io.bytes_recv

    ondemand_cost = calculate_cost(duration, ONDEMAND_HOSTING_COST_PER_HOUR)
    reserved_one_year_cost = calculate_cost(duration, RESERVED_ONE_YEAR_HOSTING_COST_PER_HOUR)
    reserved_three_year_cost = calculate_cost(duration, RESERVED_THREE_YEAR_HOSTING_COST_PER_HOUR)
    spot_cost = calculate_cost(duration, SPOT_HOSTING_COST_PER_HOUR)

    metrics = {
        "duration": duration, 
        "cpu_usage": max(cpu_usage_change, 0), 
        "memory_usage": max(memory_usage_change, 0), 
        "gpu_usage": max(gpu_usage_change, 0), 
        "gpu_temp_change": max(avg_gpu_temp - start_gpu_temp, 0),
        "disk_read_bytes": disk_read_bytes,
        "disk_write_bytes": disk_write_bytes,
        "net_sent_bytes": net_sent_bytes,
        "net_recv_bytes": net_recv_bytes,
        "gpu_power_usage": max(avg_gpu_power_usage, 0),
        "ondemand_cost": ondemand_cost,
        "reserved_one_year_cost": reserved_one_year_cost,
        "reserved_three_year_cost": reserved_three_year_cost,
        "spot_cost": spot_cost,
        "image_size": image_size
    }

    mllogger.end(key='image_generation', value=metrics)

    print(f"Image {count} generated in {duration:.2f} seconds.")
    print(f"CPU usage change: {cpu_usage_change:.2f}%")
    print(f"Memory usage change: {memory_usage_change:.2f}%")
    print(f"GPU usage change: {gpu_usage_change:.2f}%")
    print(f"GPU temperature change: {avg_gpu_temp - start_gpu_temp:.2f}°C")
    print(f"Disk read bytes: {disk_read_bytes}")
    print(f"Disk write bytes: {disk_write_bytes}")
    print(f"Network sent bytes: {net_sent_bytes}")
    print(f"Network received bytes: {net_recv_bytes}")
    print(f"GPU power usage: {avg_gpu_power_usage:.2f} watts")
    print(f"On Demand Cost of image generation: ${ondemand_cost:.6f}")
    print(f"Reserved for one year Cost of image generation: ${reserved_one_year_cost:.6f}")
    print(f"Reserved for three year Cost of image generation: ${reserved_three_year_cost:.6f}")
    print(f"Spot instance Cost of image generation: ${spot_cost:.6f}")
    print(f"Image size: {image_size}")

def text2Image(model, prompt, count, device_handles):
    if not os.path.exists(SERVER_NAME):
        os.makedirs(SERVER_NAME)
    
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
    start_cpu_usage = psutil.cpu_percent(interval=None)
    start_memory_usage = psutil.virtual_memory().percent
    start_gpu_usage = sum([pynvml.nvmlDeviceGetUtilizationRates(handle).gpu for handle in device_handles]) / len(device_handles)
    start_gpu_temp = sum([pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU) for handle in device_handles]) / len(device_handles)
    start_disk_io = psutil.disk_io_counters()
    start_net_io = psutil.net_io_counters()
    
    mllogger.start(key='image_generation', value={"model": model, "prompt": prompt, "count": count})
    
    try:
        with autocast("cuda"):
            image = pipe(prompt)["images"][0]
        image_size = image.size
    except Exception as e:
        mllogger.end(key='image_generation', value={"error": str(e)})
        print(f"Error generating image: {e}")
        return
    
    image_path = os.path.join(SERVER_NAME, f"{count}.png")
    image.save(image_path)
    
    # Log monitoring info after image generation
    log_monitoring_info(start_time, start_cpu_usage, start_memory_usage, start_gpu_usage, start_gpu_temp, start_disk_io, start_net_io, device_handles, count, image_size)
    
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

def main():
    model = "CompVis/stable-diffusion-v1-4"
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
    
    device_handles = init_gpu_monitoring()    
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)} for prompt: '{prompt}'")
        try:
            text2Image(model, prompt, i+1, device_handles)
            print(f"Image {i+1} saved.")
        except RuntimeError as e:
            print(f"Failed to generate image {i+1}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
    
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()
