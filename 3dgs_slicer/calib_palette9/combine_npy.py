# Reads in 9 npy files named 1.npy, 2.npy, ..., 9.npy and combines them into a single npy file named combined.npy
# input files has shape [15,15,4], combine them into a single file with shape [45,45,4]

import numpy as np
from PIL import Image

combined = np.zeros((45, 45, 4), dtype=np.uint8)

for i in range(3):
    for j in range(3):
        idx = i * 3 + j + 1
        combined[i*15:i*15+15, j*15:j*15+15] = np.load(f'out/{idx}.npy')

np.save('combined.npy', combined)

# Save the image
image = Image.fromarray(combined)
image.save("combined.png")
