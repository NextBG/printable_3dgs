# Reads in the calibration palette data from the file 'palette_9_calib.npy' which is a [45, 45, 4] RGBA numpy array.
# Then Reads in the uncalibrated palette data from the file 'palette_9.npy' which is a dictionary that maps [6] CMYKWA numpy arrays to [4] RGBA numpy arrays.
# Use the calibration palette to replace the uncalibrated palette's RGBA values with the calibrated values.

import numpy as np
import os

# Load the calibration palette
calib_palette = np.load('palette_9_calib.npy')

# Flatten to 2D array
calib_palette = calib_palette.reshape(-1, 4)

# Load the uncalibrated palette
uncalib_palette = np.load('palette_9.npy', allow_pickle=True).item()

# Replace the uncalibrated palette's RGBA values with the calibrated values
idx = 0
for key, value in uncalib_palette.items():
    uncalib_palette[key][:3] = calib_palette[idx][:3]
    idx += 1

# Save the calibrated palette
np.save('palette_9_calibrated.npy', uncalib_palette)
