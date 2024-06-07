import torch
from diffusers import StableDiffusionPipeline
import os

# Replace with your Hugging Face API token
API_TOKEN = "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"

def generate_image(prompt):
    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=API_TOKEN)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate an image
    with torch.no_grad():
        image = pipe(prompt).images[0]

    # Save the generated image
    image.save("generated_image.png")

if __name__ == "__main__":
    prompt = "A beautiful landscape with mountains and rivers"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    generate_image(prompt)
    print("Image generated and saved as 'generated_image.png'")
