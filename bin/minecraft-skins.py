
from diffusers import StableDiffusionPipeline
from PIL import Image

import numpy as np
import argparse
import logging
import torch
import sys

MODEL_NAME = "monadical-labs/minecraft-skin-generator"
MASK_IMAGE = "images/skin-half-mask.png"

SCALE = 12

def extract_minecraft_skin(generated_image):
    # Extract the skin portion from the top half of the image.
    width, height = generated_image.size
    generated_skin = generated_image.crop((0, 0, width, height/2))
    
    # Scale the skin down to the expected size of 64x32 pixels.
    width, height = generated_skin.size
    scaled_skin = generated_skin.resize((int(width / SCALE), int(height / SCALE)), resample=Image.NEAREST) 
    
    return scaled_skin

def restore_skin_alphachannels(image):
    # Convert the image to RGBA.
    converted_image = image.convert('RGBA')

    # Convert the image into a numpy array.
    image_data = np.array(converted_image)
    red, green, blue, alpha = image_data.T

    # Convert all of the black pixels in the skin to slightly-less-black.
    # We're going to later use (0,0,0) as our transparency color, so this
    # will prevent transparent pixels in our skin.
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    image_data[..., :-1][black_areas.T] = (1, 1, 1)

    # Convert the data back into Image format.
    converted_image = Image.fromarray(image_data)
    converted_image = converted_image.convert("P")

    # Enable transparency in the skin image.                                                                                                                                               
    converted_image = converted_image.convert("P")
    converted_image.info["transparency"] = 0
    converted_image = converted_image.convert("RGBA")
    
    # Load an imagemask of unused skin areas that shoudl be fully transparent.                                                                                          
    mask_image = Image.open(MASK_IMAGE)
    mask_image = mask_image.convert("RGBA")
    
    # Perform the alpha composite, and return the result.                                                                                                                             
    mask_image.alpha_composite(converted_image)
    
    return converted_image

def main(prompt, filename, logger):
    # Enable GPU acceleration frameworks, if enabled.
    device = "cpu"
    dtype = torch.float16
    
    if torch.cuda.is_available() and torch.backends.cuda.is_built():
        # A CUDA compatible GPU was found.
        logger.info("CUDA device found, enabling.")
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Apple M1/M2 machines have the MPS framework.
        logger.info("Apple MPS device found, enabling.")
        device = "mps"
        dtype = torch.float32
    else:
        # Else we're defaulting to CPU.
        logger.info("No CUDA or MPS devices found, running on CPU.")

    # Load (and possibly download) our Minecraft model.
    logger.info("Loading HuggingFace model: '{}'.".format(MODEL_NAME))
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    pipeline.to(device)
    
    # Generate the image given the prompt provided on the command line.
    logger.info("Generating skin with prompt: '{}'.".format(prompt))
    generated_image = pipeline(prompt=prompt).images[0]

    # Extract and scale down the Minecraft skin portion of the image.
    logger.info("Extracting and scaling Minecraft skin from generated image.")
    minecraft_skin = extract_minecraft_skin(generated_image)

    logger.info("Restoring transparency in generated skin file.")
    # Clean up any background noise in the sking and restore the alphachannel transparency.
    minecraft_skin = restore_skin_alphachannels(minecraft_skin)

    logger.info("Saving skin to: '{}'.".format(filename))
    minecraft_skin.save(filename)
    
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format='[%(asctime)s] %(levelname)s - %(message)s')

    logger = logging.getLogger("minecraft-skins")

    # Get all of the command line parameters and options passed to us.
    parser = argparse.ArgumentParser(description='Process the command line arguments.')

    parser.add_argument('filename', type=str, help='Name of the generated Minecraft skin file')
    parser.add_argument('prompt', type=str, help='Stable Diffusion prompt to be used to generate skin')
    parser.add_argument('--verbose', help='Produce verbose output while running', action='store_true', default=False)

    args = parser.parse_args()

    filename = args.filename
    verbose = args.verbose
    prompt = args.prompt
    
    if verbose:
        logger.setLevel(logging.INFO)
    
    main(prompt, filename, logger)
    
