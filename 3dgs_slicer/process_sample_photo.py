from PIL import Image
import numpy as np
import os

# Import the image
INPUT_FOLDER = 'calib_palette9'
# Select a square of 15 * 15 pixels from the image, plot on the original image the edges of the square
GRID_WIDTH = 15
SELECT_WIDTH = 120
SAMPLE_START_X = 325 #265 for photos other than 6.jpg
SAMPLE_START_Y = 665
SPACING_X = 177
SPACING_Y = 176

# Create output folders if not exist
if not os.path.exists(f'{INPUT_FOLDER}/out'):
    os.makedirs(f'{INPUT_FOLDER}/out')
if not os.path.exists(f'{INPUT_FOLDER}/out_vis'):
    os.makedirs(f'{INPUT_FOLDER}/out_vis')

# For all files in the folder
for input_img in sorted(os.listdir(INPUT_FOLDER)):

    # Skip if not a jpg file
    if not input_img.endswith('.jpg'):
        continue

    print(f'Processing {input_img}')

    file_name = input_img.split('.')[0]
    image = Image.open(f'{INPUT_FOLDER}/{file_name}.jpg')
    image_array = np.array(image)

    # Create a 10 * 10 array, to store the output image
    output_image = np.zeros((15, 15, 4), dtype=np.uint8)


    # Construct the start pixels
    start_pixels = []
    for row in range(GRID_WIDTH):
        for col in range(GRID_WIDTH):
            start_pixels.append( (SAMPLE_START_X + col * SPACING_X, SAMPLE_START_Y + row * SPACING_Y) )

    # Iterate over the start pixels
    for i in range(len(start_pixels)):
        start_x, start_y = start_pixels[i][0] - SELECT_WIDTH // 2, start_pixels[i][1] - SELECT_WIDTH // 2
        end_x, end_y = start_pixels[i][0] + SELECT_WIDTH//2, start_pixels[i][1] + SELECT_WIDTH //2

        pix_list = []
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                # Plot the edges of the square
                if x == start_x or x == end_x - 1 or y == start_y or y == end_y - 1:
                    image_array[y, x] = [255, 255, 255]
                if x == start_x + 1 or x == end_x - 2 or y == start_y + 1 or y == end_y - 2:
                    image_array[y, x] = [255, 255, 255]
                if x == start_x + 2 or x == end_x - 3 or y == start_y + 2 or y == end_y - 3:
                    image_array[y, x] = [255, 255, 255]
                if x == start_x + 3 or x == end_x - 4 or y == start_y + 3 or y == end_y - 4:
                    image_array[y, x] = [255, 255, 255]

                # Store the pixel value
                pix_list.append(image_array[y, x])

        # Get the average pixel value
        pix_list = np.array(pix_list)
        average_pixel = np.mean(pix_list, axis=0)

        # Save the average pixel value to the output image, append the alpha channel
        output_image[i // GRID_WIDTH, i % GRID_WIDTH] = np.append(average_pixel, 255)

    # Save
    # Save the image array as raw npy file
    np.save(f'{INPUT_FOLDER}/out/{file_name}.npy', output_image)
    image = Image.fromarray(image_array)
    image.save(f'{INPUT_FOLDER}/out_vis/{file_name}.png')
    output_image = output_image[:, 1:]
    output_image = Image.fromarray(output_image)
    output_image.save(f'{INPUT_FOLDER}/out/{file_name}.png')