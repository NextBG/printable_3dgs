DPI = 600
MM_PER_INCH = 25.4
PIX_PER_MM = DPI / MM_PER_INCH

import numpy as np
from PIL import Image

# Generate an RGBA color_test.png image that has a grid of colors
# Each row of the grid will have the same color, but with a linearly increasing opacity
# The columns will have the same opacity, but with different colors
# Color for each row are: Cyan, Magenta, Yellow, Black, White, 

colors = {
    'Cyan': [0, 1, 1, 1],
    'Magenta': [1, 0, 1, 1],
    'Yellow': [1, 1, 0, 1],
    'Black': [0, 0, 0, 1],
    'White': [1, 1, 1, 1],

    'Red': [1, 0, 0, 1],
    'Green': [0, 1, 0, 1],
    'Blue': [0, 0, 1, 1],
    'Gray': [0.5, 0.5, 0.5, 1],

    'Peach': [1, 0.75, 0.5, 1],
    'Orange': [1, 0.5, 0, 1],
    'Purple': [0.5, 0, 0.5, 1],
    'Teal': [0, 0.5, 0.5, 1],
    'Lime': [0.5, 1, 0, 1],
    'Pink': [1, 0.75, 0.8, 1],
    'Sky Blue': [0.5, 1, 1, 1],
    'Lavender': [0.75, 0.5, 1, 1],
}

PALETTE_W = 3
PIX_DUPL = 10
NUM_LEVELS = 9

grid_size = (len(colors), NUM_LEVELS)  # Number of colors x Number of opacity levels
image_size = (grid_size[1]*PALETTE_W*PIX_DUPL, grid_size[0]*PALETTE_W*PIX_DUPL)  # Width x Height in pixels

print(f"Image size: {image_size[0]} x {image_size[1]} pixels")
# print(f"Print size: {print_size} x {print_size} mm")
# print(f"width of each color cell: {print_size / len(colors)} mm")

# Create a blank RGBA image
image = Image.new("RGBA", image_size)
pixels = image.load()

# Calculate cell size
cell_width = image_size[0] // grid_size[1]
cell_height = image_size[1] // grid_size[0]

# Generate the color grid
for row, (color_name, color) in enumerate(colors.items()):
    for col in range(grid_size[1]):
        alpha = (col+1) / (grid_size[1])  # Linearly increasing alpha
        rgba = (np.array(color) * [1, 1, 1, alpha]).astype(np.float32)
        rgba = tuple((rgba * 255).astype(np.uint8))
        for i in range(cell_width):
            for j in range(cell_height):
                x = col * cell_width + i
                y = row * cell_height + j
                pixels[x, y] = rgba

                # if i == 0 or j == 0:
                #     pixels[x, y] = tuple((np.array([1, 1, 1, 1]) * 255).astype(np.uint8))

# Save the image
image.save("test/000000.png")

print("Color test image generated: 000000.png")