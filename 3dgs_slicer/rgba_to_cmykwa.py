# This python script is used to convert RGBA png image into CMYK(black)W(white)Cl(transparient) ink image.
from PIL import Image, ImageEnhance
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import time
import os

# Seed
np.random.seed(0)
    
def find_ink_combination(inks_array:np.array, rgba_array:np.array, target_rgba:np.array, weights:np.array, slot_count:int) -> np.array:
    # Find how many slots should be non-transparent according to the target alpha
    target_trans_slots = slot_count - int(np.round(target_rgba[3] * slot_count))

    # Rule out all inks that don't have the same number of transparent slots, which is the last slot in inks_array
    rgba_array = rgba_array[inks_array[:, -1] == target_trans_slots]
    inks_array = inks_array[inks_array[:, -1] == target_trans_slots]

    # Include the RGB difference
    rgb_diff = np.abs(target_rgba[:3] - rgba_array[:, :3])
    rgb_diff = weights[0] * np.linalg.norm(rgb_diff, axis=1)

    # Include the ratio difference
    epsilon = 1e-6
    target_ratios = target_rgba[:3] / (np.sum(target_rgba[:3]) + epsilon)
    rgba_ratios = rgba_array[:, :3] / (np.sum(rgba_array[:, :3], axis=1)[:, np.newaxis] + epsilon)
    
    # Squared difference
    ratio_diff = np.sum((target_ratios - rgba_ratios) ** 4, axis=1)
    ratio_diff = weights[1] * np.sqrt(ratio_diff)

    # Combine the differences
    distances = rgb_diff + ratio_diff

    # Find the index of the minimum distance
    min_index = np.argmin(distances)
    
    # Get the closest ink combination
    closest_match = inks_array[min_index]

    return closest_match


def preprocess_image(input_img: Image) -> Image:
    # # Compress the image to half its height
    # input_img = input_img.resize((input_img.width, input_img.height // 2), Image.Resampling.LANCZOS)
    
    # Convert the image to RGBA if it's not already in that mode
    input_img = input_img.convert("RGBA")
    
    # Increase the saturation
    enhancer = ImageEnhance.Color(input_img)
    input_img = enhancer.enhance(2)

    # Increase the contrast
    enhancer = ImageEnhance.Contrast(input_img)
    input_img = enhancer.enhance(2)
    
    return input_img


def process_image(input_img:Image, inks_array:np.array, rgba_array:np.array, matching_weights:np.array, mix_width_pix:int, power:int, slot_count:int) -> Image:
    # Turn image into numpy array
    input_img_array = np.array(input_img)

    # Skip if the image is pure transparent
    if np.all(input_img_array[:, :, 3] == 0):
        return input_img

    # Get the original image size
    origin_width = input_img_array.shape[1]
    origin_height = input_img_array.shape[0]

    # First , extend the image to the nearest multiple of MIX_WIDTH_PIX, fill new cols with last cols, then fill new rows with last rows
    ext_width = origin_width + (mix_width_pix - origin_width % mix_width_pix) % mix_width_pix
    ext_height = origin_height + (mix_width_pix - origin_height % mix_width_pix) % mix_width_pix
    ext_img = np.zeros((ext_height, ext_width, 4), dtype=np.float32)
    ext_img[:origin_height, :origin_width] = input_img_array
    ext_img = ext_img / 255.0

    for y in range(origin_height, ext_height):
        ext_img[y] = ext_img[y-1]
    for x in range(origin_width, ext_width):
        ext_img[:, x] = ext_img[:, x-1]

    # Down sample the image
    downsampled_width = ext_width // mix_width_pix
    downsampled_height = ext_height // mix_width_pix
    downsampled_flat = np.zeros((downsampled_height * downsampled_width, 4), dtype=np.float32)

    for idx in range(downsampled_flat.shape[0]):
        y = idx // downsampled_width
        x = idx % downsampled_width

        y_start = y * mix_width_pix
        y_end = y_start + mix_width_pix
        x_start = x * mix_width_pix
        x_end = x_start + mix_width_pix

        downsampled_flat[idx] = np.mean(ext_img[y_start:y_end, x_start:x_end], axis=(0, 1))

    # alpha remapping linear -> power
    downsampled_flat[:, 3] = downsampled_flat[:, 3] ** power

    # # Save the power remapped image, first reshape the flat image to 2D
    # remapped_img = Image.fromarray((downsampled_flat * 255).astype(np.uint8).reshape((downsampled_height, downsampled_width, 4)))
    # remapped_img.save('remapped.png')

    # Create the color map for each ink type
    ink_colors = np.array([
        [0, 255, 255, 255],
        [255, 0, 255, 255],
        [255, 255, 0, 255],
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [0, 0, 0, 0]
    ])

    # Create a new image with the same size as the down sampled image
    new_image = np.zeros((ext_height, ext_width, 4), dtype=np.uint8)

    # COLOR MATCHING
    for idx in range(downsampled_flat.shape[0]):
        # Skip transparent pixels
        if downsampled_flat[idx, 3] == 0:
            continue

        rgba = downsampled_flat[idx]
        closest_inks = find_ink_combination(inks_array, rgba_array, rgba, matching_weights, slot_count)

        ink_list = np.repeat(np.arange(6), closest_inks)
        np.random.shuffle(ink_list)

        y_start = (idx // downsampled_width) * mix_width_pix
        x_start = (idx % downsampled_width) * mix_width_pix

        for i in range(mix_width_pix * mix_width_pix):
            y = y_start + i // mix_width_pix
            x = x_start + i % mix_width_pix
            new_image[y, x] = ink_colors[ink_list[i]]

    # Crop the new image to the size of the origin image
    new_image = new_image[:origin_height, :origin_width]

    return Image.fromarray(new_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert RGBA image to CMYKWA image')
    parser.add_argument('-s', '--slot_count', type=int, default=9, help='The number of slots in the palette')
    parser.add_argument('-w', '--weights', type=int, nargs=2, default=[1, 1], help='The weights for each channel in the matching process')
    parser.add_argument('-i', '--input_folder', type=str, default='input', help='The folder containing the input images')
    parser.add_argument('-p', '--power', type=int, default=1, help='The power for alpha channel remapping')
    args = parser.parse_args()

    PALETTE_FILE = f'palette_{args.slot_count}.npy'
    MATCHING_WEIGHTS = np.array(args.weights, dtype=np.uint8)
    MIX_WIDTH_PIX = np.uint8(np.sqrt(args.slot_count))

    # Load the palette
    palette = np.load(PALETTE_FILE, allow_pickle=True).item()
    inks_array = np.array(list(palette.keys()))
    rgba_array = np.array(list(palette.values()))

    # # Save the new image
    # output_folder_name = args.input_folder + f'_s{args.slot_count}_w{MATCHING_WEIGHTS[0]}{MATCHING_WEIGHTS[1]}{MATCHING_WEIGHTS[2]}_p{args.power}'
    # if not os.path.exists(output_folder_name):
    #     os.makedirs(output_folder_name)

    start_time = time.time()

    # for all images in the input folder
    idx = 0
    for file in tqdm(sorted(os.listdir(args.input_folder)), desc='Processing images'):
        input_img = Image.open(args.input_folder + '/' + file)
        input_img = preprocess_image(input_img)
        new_image = process_image(input_img, inks_array, rgba_array, MATCHING_WEIGHTS, MIX_WIDTH_PIX, args.power, args.slot_count)
        new_image.save(f'{idx:06d}.png')
        idx += 1

    print(f'Processed {idx} images in {time.time() - start_time:.2f} seconds')
