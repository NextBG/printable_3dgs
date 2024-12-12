# This script generates mixture of CMYKWCl inks and saves the resulting image

import numpy as np
from PIL import Image

MIX_WIDTH = 3
SLOT_COUNT = np.uint8(MIX_WIDTH * MIX_WIDTH)

def cmykwa_to_rgba(cmykwa: np.array):
    """
    Convert CMYKWCl to RGBA.
    c, m, y, k, w, a: ratios of Cyan, Magenta, Yellow, Black, White, Clear
    Returns: numpy array [R, G, B, A]
    """
    
    # Calaulate the percentage of each ink
    total_ink = np.sum(cmykwa)
    cmykwa_n = cmykwa / total_ink

    C, M, Y, K, W, A = cmykwa_n

    # The initial light that is not absorbed by the inks
    result = np.array([0, 0, 0, 0], dtype=np.float32)

    result = result + C * np.array([0, 1, 1, 1], dtype=np.float32)
    result = result + M * np.array([1, 0, 1, 1], dtype=np.float32)
    result = result + Y * np.array([1, 1, 0, 1], dtype=np.float32)
    result = result + K * np.array([0, 0, 0, 1], dtype=np.float32)
    result = result + W * np.array([1, 1, 1, 1], dtype=np.float32)
    # result = result + A * np.array([0, 0, 0, 0], dtype=np.float32)

    return result

def generate_image(palette: dict, output_path: str):
    # Generate a square image with the palette
    image_size = int(np.ceil(np.sqrt(color_count)))

    # Create a new image that has COLOR_PIX_WIDTH times the width and height
    new_width = image_size
    new_height = image_size

    new_image = np.zeros((new_height, new_width, 4), dtype=np.uint8)

    # Iterate over the original image and fill in the new image
    for y in range(new_height):
        for x in range(new_width):
            idx = y * new_width + x
            if idx < color_count:
                rgba = palette[list(palette.keys())[idx]]

                new_image[y, x] = [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), int(rgba[3] * 255)]

    # Save the image
    image = Image.fromarray(new_image)
    image.save(output_path)

    # Save the subimage that is the image cut to 9 small images, each with width and height of 1/3 of the original image
    subimage_size = new_width // MIX_WIDTH
    for i in range(MIX_WIDTH):
        for j in range(MIX_WIDTH):
            subimage = image.crop((j*subimage_size, i*subimage_size, (j+1)*subimage_size, (i+1)*subimage_size))
            subimage.save(f'palette_{SLOT_COUNT}/palette_{SLOT_COUNT}_{i}{j}.png')


if __name__ == '__main__':
    # C M Y K W A(clear)
    ink_c = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    ink_m = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
    ink_y = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
    ink_k = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    ink_w = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)
    ink_a = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)

    # Slot count
    palette = {}

    ink_result = np.array([0, 0, 0, 0, 0, 0], dtype=np.uint8)
    for i in range(SLOT_COUNT+1):
        slot_mykwa = SLOT_COUNT - i
        for j in range(slot_mykwa+1):
            slot_ykwa = slot_mykwa - j
            for k in range(slot_ykwa+1):
                slot_kwa = slot_ykwa - k
                for l in range(slot_kwa+1):
                    slot_wa = slot_kwa - l
                    for m in range(slot_wa+1):
                        slot_a = slot_wa - m
                        ink_result = ink_c * i + ink_m * j + ink_y * k + ink_k * l + ink_w * m + ink_a * slot_a
                        rgba = cmykwa_to_rgba(ink_result)
                        # print(f'Ink combination: {ink_result}, RGBA: {rgba}')
                        palette[tuple(ink_result)] = rgba

    color_count = len(palette)

    # Create palette img folder if it doesn't exist
    # Save the palette
    np.save(f'palette_{SLOT_COUNT}.npy', palette)
    print(f'Generated palette with {color_count} colors')

    import os
    if not os.path.exists(f'palette_{SLOT_COUNT}'):
        os.makedirs(f'palette_{SLOT_COUNT}')

    # Generate the image
    generate_image(palette, f'palette_{SLOT_COUNT}/palette_{SLOT_COUNT}.png')

