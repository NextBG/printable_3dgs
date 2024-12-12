#include <curand_kernel.h>

extern "C" __global__
void color_matching(
    const float* downsampled_flat, const float* inks_array, const float* rgba_array, 
    const float* MATCHING_WEIGHTS, const float* ink_colors, uint8_t* new_flat,
    const int downsampled_width, const int downsampled_height, const int MIX_WIDTH_PIX, const int layer_idx,
    const int inks_array_size, const int rgba_array_size, const int slot_count) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= downsampled_width * downsampled_height) return;

    int y = idx / downsampled_width;
    int x = idx % downsampled_width;

    // Get the RGBA values of the pixel
    float rgba[4];
    for (int i = 0; i < 4; ++i) {
        rgba[i] = downsampled_flat[idx * 4 + i];
    }

    // Skip if the pixel is transparent
    if (rgba[3] == 0) return;

    // Calculate the number of non-transparent slots
    int target_trans_slots = slot_count - int(roundf(rgba[3] * slot_count));

    // Find the closest ink color
    float min_distance = 1e20;
    int min_index = 0;

    for (int i = 0; i < rgba_array_size; ++i) {
        // Skip inks that don't have the same number of transparent slots
        if (int(inks_array[i * 6 + 5]) != target_trans_slots) continue;

        // Calculate RGB difference
        float rgb_diff = 0.0f;
        for (int j = 0; j < 3; ++j) {
            float diff = rgba[j] - rgba_array[i * 4 + j];
            rgb_diff += diff * diff;
        }
        rgb_diff = sqrtf(rgb_diff);

        // Calculate ratio difference
        float target_sum = rgba[0] + rgba[1] + rgba[2] + 1e-6;
        float candidate_sum = rgba_array[i * 4 + 0] + rgba_array[i * 4 + 1] + rgba_array[i * 4 + 2] + 1e-6;

        float ratio_diff = 0.0f;
        for (int j = 0; j < 3; ++j) {
            float target_ratio = rgba[j] / target_sum;
            float candidate_ratio = rgba_array[i * 4 + j] / candidate_sum;
            float diff = target_ratio - candidate_ratio;
            ratio_diff += diff * diff * diff * diff;  // Power of 4
        }
        ratio_diff = sqrtf(ratio_diff);

        // Calculate total distance
        float distance = MATCHING_WEIGHTS[0] * rgb_diff + MATCHING_WEIGHTS[1] * ratio_diff;

        // Find the ink with the minimum distance
        if (distance < min_distance) {
            min_distance = distance;
            min_index = i;
        }
    }

    // Get the closest inks
    int closest_inks[6];
    for (int i = 0; i < 6; ++i) {
        closest_inks[i] = inks_array[min_index * 6 + i];
    }

    // Create a list of inks to be used for mixing
    int ink_list[36];  // 6 types of inks, with max possible counts of each = 6
    int ink_list_size = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < closest_inks[i]; ++j) {
            ink_list[ink_list_size++] = i;
        }
    }

    // Shuffle ink_list using LCG
    unsigned int seed = (idx + 1) * (layer_idx + 1);
    for (int i = ink_list_size - 1; i > 0; --i) {
        seed = (1103515245 * seed + 12345) & 0x7fffffff;  // LCG
        int j = seed % (i + 1);

        // Swap ink_list[i] and ink_list[j]
        int temp = ink_list[i];
        ink_list[i] = ink_list[j];
        ink_list[j] = temp;
    }

    int y_start = y * MIX_WIDTH_PIX;
    int x_start = x * MIX_WIDTH_PIX;

    for (int i = 0; i < MIX_WIDTH_PIX * MIX_WIDTH_PIX; ++i) {
        int row = y_start + i / MIX_WIDTH_PIX;
        int col = x_start + i % MIX_WIDTH_PIX;
        int ink_idx = ink_list[i];
        for (int j = 0; j < 4; ++j) {
            new_flat[(row * downsampled_width * MIX_WIDTH_PIX + col) * 4 + j] = ink_colors[ink_idx * 4 + j];
        }
    }
}
