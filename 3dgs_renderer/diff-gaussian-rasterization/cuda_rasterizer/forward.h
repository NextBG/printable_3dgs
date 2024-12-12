#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int M,
		const float* means3D,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		int W, int H, int PPU,
		float slice_height,
		int* radii,
		float3* gs_means3D,
		float2* gs_rend_means2D,
		float* depths,
		float* cov3Ds,
		float* rgb,
		float* opas,
		const dim3 grid,
		uint32_t* tiles_touched
		);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H, int PPU,
		float slice_height,
		float slice_thickness,
		const float3* gs_means3D,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float* colors,
		const float* opas,
		float* final_T,
		float* out_color);
}


#endif