import cupy as cp
from PIL import Image
import numpy as np
import argparse
import os
import cupy as cp
from tqdm import tqdm
import cProfile
from PIL import Image, ImageEnhance
import time

# Set random seed
RAND_SEED = 0
cp.random.seed(RAND_SEED)

# Constants
INK_COLORS = cp.array([
    [0, 255, 255, 255],  # C
    [255, 0, 255, 255],  # M
    [255, 255, 0, 255],  # Y
    [0, 0, 0, 255],      # K
    [255, 255, 255, 255],# W
    [0, 0, 0, 0]         # Cl
], dtype=cp.float32)

# CUDA setup
with open('color_matching_kernel.cu', 'r') as file:
    kernel_code = file.read()
color_matching_kernel = cp.RawKernel(kernel_code, 'color_matching', backend='nvcc')

def preprocess_image(input_img: Image, saturation_factor:float, brightness_factor:float, contrast_factor:float) -> Image:
    # # Compress the image to half its height
    input_img = input_img.resize((input_img.width, input_img.height // 2), Image.Resampling.LANCZOS)
    
    # Convert the image to RGBA if it's not already in that mode
    input_img = input_img.convert("RGBA")

    # Increase the saturation
    enhancer = ImageEnhance.Color(input_img)
    input_img = enhancer.enhance(saturation_factor)

    # Lower the brightness
    enhancer = ImageEnhance.Brightness(input_img)
    input_img = enhancer.enhance(brightness_factor)

    # Increase the contrast
    enhancer = ImageEnhance.Contrast(input_img)
    input_img = enhancer.enhance(contrast_factor)

    return input_img

def process_image_gpu(input_img:Image, inks_array:cp.array, rgba_array:cp.array, matching_weights:cp.array, mix_width_pix:int, power:int, layer_idx:int, slot_count:int) -> Image:
    # Load the image
    input_img_array = cp.array(input_img, dtype=cp.float32)

    # Skip if the image is pure transparent
    if cp.all(input_img_array[:, :, 3] == 0):
        return input_img

    origin_width = input_img_array.shape[1]
    origin_height = input_img_array.shape[0]

    # Extend the image to the nearest multiple of MIX_WIDTH_PIX
    ext_width = origin_width + (mix_width_pix - origin_width % mix_width_pix) % mix_width_pix
    ext_height = origin_height + (mix_width_pix - origin_height % mix_width_pix) % mix_width_pix
    ext_img = cp.zeros((ext_height, ext_width, 4), dtype=cp.float32)
    ext_img[:origin_height, :origin_width] = input_img_array / 255.0

    ext_img[origin_height:] = ext_img[origin_height-1:origin_height]
    ext_img[:, origin_width:] = ext_img[:, origin_width-1:origin_width]

    # Downsample the image
    downsampled_width = ext_width // mix_width_pix
    downsampled_height = ext_height // mix_width_pix
    downsampled_flat = cp.zeros((downsampled_height * downsampled_width, 4), dtype=cp.float32)

    # Downsample with Image Resampling
    downsampled_img = Image.fromarray((ext_img.get() * 255).astype(np.uint8))
    downsampled_img = downsampled_img.resize((downsampled_width, downsampled_height), Image.Resampling.LANCZOS)
    downsampled_flat = cp.array(downsampled_img, dtype=cp.float32).reshape(-1, 4) / 255.0

    # # Downsample with mean
    # for idx in range(downsampled_flat.shape[0]):
    #     y = idx // downsampled_width
    #     x = idx % downsampled_width

    #     y_start = y * mix_width_pix
    #     y_end = y_start + mix_width_pix
    #     x_start = x * mix_width_pix
    #     x_end = x_start + mix_width_pix

    #     downsampled_flat[idx] = cp.mean(ext_img[y_start:y_end, x_start:x_end], axis=(0, 1))

    # Save the downsampled image
    downsampled_flat[:, 3] = downsampled_flat[:, 3] ** power

    # alpha remapping linear -> power
    new_flat = cp.zeros((ext_height * ext_width, 4), dtype=cp.uint8)

    # Color matching kernel
    blocks_per_grid = (downsampled_width * downsampled_height + 255) // 256
    color_matching_kernel(
        (blocks_per_grid,), (256,),
        (
            downsampled_flat.ravel(), inks_array.ravel(), rgba_array.ravel(), matching_weights, INK_COLORS.ravel(), new_flat.ravel(),
            downsampled_width, downsampled_height, mix_width_pix, layer_idx,
            inks_array.shape[0], rgba_array.shape[0], slot_count
        )
    )
    new_image = new_flat.reshape((ext_height, ext_width, 4))
    new_image = new_image[:origin_height, :origin_width]

    return Image.fromarray(new_image.get())

def main():
    parser = argparse.ArgumentParser(description='Convert RGBA image to CMYKWA image')
    parser.add_argument('-i', '--input_folder', type=str, default='input', help='The folder containing the input images')
    parser.add_argument('-o', '--output_folder', type=str, default='output', help='The folder to save the output images')
    parser.add_argument('-s', '--slot_count', type=int, default=9, help='The number of slots in the palette')
    parser.add_argument('-w', '--weights', type=int, nargs=2, default=[1, 1], help='The weights for each channel in the matching process')
    parser.add_argument('-p', '--power', type=int, default=1, help='The power for alpha channel remapping')
    parser.add_argument('-pp', '--preprocess_factors', type=float, nargs=3, default=[4, 0.3, 2], help='The factors for preprocessing the image')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Input folder '{args.input_folder}' does not exist.")
        return

    PALETTE_FILE = f'palette_{args.slot_count}.npy'
    MATCHING_WEIGHTS = cp.array(args.weights, dtype=np.float32)
    MIX_WIDTH_PIX = cp.uint8(cp.sqrt(args.slot_count).get())

    # Load the palette
    palette = np.load(PALETTE_FILE, allow_pickle=True).item()
    inks_array = cp.array(list(palette.keys()), dtype=cp.float32)
    rgba_array = cp.array(list(palette.values()), dtype=cp.float32)

    # If debug mode, add an "_debug" suffix to the output folder
    if args.debug:
        args.output_folder += '_debug'

    # Folder to save the output images
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Save the mata data as a text file
    with open(args.output_folder + '/metadata.txt', 'w') as file:
        file.write(f"Process time:                                          {time.ctime()}\n")
        file.write(f"Input folder:                                          {args.input_folder}\n")
        file.write(f"Output folder:                                         {args.output_folder}\n")
        file.write(f"Palette file:                                          {PALETTE_FILE}\n")
        file.write(f"Palette slot count:                                    {args.slot_count}\n")
        file.write(f"Preprocess factors(saturation, brightness, contrast):  {args.preprocess_factors}\n")
        file.write(f"Matching weights:                                      {args.weights}\n")
        file.write(f"Alpha remapping power:                                 {args.power}\n")

    # Process all images in the input folder
    idx = 0
    for file in tqdm(sorted(os.listdir(args.input_folder))):
        # Skip if the file is not an image
        if not file.endswith('.png'):
            continue
        input_img = Image.open(args.input_folder + '/' + file)
        preprocessed_img = preprocess_image(input_img, args.preprocess_factors[0], args.preprocess_factors[1], args.preprocess_factors[2])
        processed_img = process_image_gpu(preprocessed_img, inks_array, rgba_array, MATCHING_WEIGHTS, MIX_WIDTH_PIX, args.power, idx, args.slot_count)
        
        if args.debug:
            # Stick the preprocessed and processed image to the right of the original image for DEBUG
            result = Image.new('RGBA', (input_img.width*3, input_img.height)) 
            result.paste(input_img, (0, 0))
            result.paste(preprocessed_img, (input_img.width, 0))
            result.paste(processed_img, (input_img.width*2, 0))
            result.save(args.output_folder + f'/{idx:06d}.png')
        else:
            processed_img.save(args.output_folder + f'/{idx:06d}.png')

        idx += 1
    

if __name__ == '__main__':
    cProfile.run('main()', 'gpu_profile.txt')