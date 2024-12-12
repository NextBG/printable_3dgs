#include "forward.h"
#include "auxiliary.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <cuda_runtime.h>
#include <stdio.h>

__device__ void computeCov3D(const glm::vec3 scale, const glm::vec4 rot, int pix_per_unit, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale.x * pix_per_unit;
	S[1][1] = scale.y * pix_per_unit;
	S[2][2] = scale.z * pix_per_unit;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix SigmapreprocessCUDA
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__device__ glm::mat3 quaternionToRotationMatrix(const glm::quat& q) {
    glm::mat3 rot_matrix;
    rot_matrix[0][0] = 1 - 2 * (q.y * q.y + q.z * q.z);
    rot_matrix[0][1] = 2 * (q.x * q.y - q.z * q.w);
    rot_matrix[0][2] = 2 * (q.x * q.z + q.y * q.w);
    rot_matrix[1][0] = 2 * (q.x * q.y + q.z * q.w);
    rot_matrix[1][1] = 1 - 2 * (q.x * q.x + q.z * q.z);
    rot_matrix[1][2] = 2 * (q.y * q.z - q.x * q.w);
    rot_matrix[2][0] = 2 * (q.x * q.z - q.y * q.w);
    rot_matrix[2][1] = 2 * (q.y * q.z + q.x * q.w);
    rot_matrix[2][2] = 1 - 2 * (q.x * q.x + q.y * q.y);
    return rot_matrix;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P,int M,
	const float* orig_points,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	int W, int H, int PPU,
	float slice_height,
	int* radii,
	float3* gs_means3D,
	float2* gs_rend_means2D,
	float* gs_depths,
	float* gs_cov3Ds,
	float* gs_rgb,
	float* gs_opacity,
	const dim3 grid,
	uint32_t* gs_tiles_touched
	)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	gs_tiles_touched[idx] = 0;

	// Transform point by projecting
	float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]}; 

	float3 p_rend = {
		p_orig.x * PPU + (W / 2.0f),
		p_orig.y,
		-p_orig.z * PPU + (H / 2.0f)
	};

	// Compute 3d coveriance from scaling and rotation parameters. 
	computeCov3D(scales[idx], rotations[idx], PPU, gs_cov3Ds + idx * 6);
	const float* cov3D;
	cov3D = gs_cov3Ds + idx * 6;

	// Compute 3D covariance matrix
	float cov_xx = cov3D[0];
	float cov_yy = cov3D[3];
	float cov_zz = cov3D[5];

	// Standard deviation
	float std_x = sqrt(cov_xx);
	float std_y = sqrt(cov_yy);
	float std_z = sqrt(cov_zz);

	// Compute radius of influence
	float radius = 4.0f * max(std_x, max(std_y, std_z));

	// If rectangle is outside screen, quit
	uint2 rect_min = {
		min(grid.x, max((int)0, (int)(p_rend.x - radius) / BLOCK_X)),
		min(grid.y, max((int)0, (int)(p_rend.z - radius) / BLOCK_Y)),
	};
	uint2 rect_max = {
		min(grid.x, max((int)0, (int)(p_rend.x + radius + BLOCK_X - 1) / BLOCK_X)),
		min(grid.y, max((int)0, (int)(p_rend.z + radius + BLOCK_Y - 1) / BLOCK_Y)),
	};
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Compute RGB color from SH coefficients 0.5 + 0.5 * SH_C0
	glm::vec3* sh = ((glm::vec3*)shs) + idx * M;
	glm::vec3 result = glm::vec3(0.5f) + 0.5f * sh[0];
	result = glm::max(result, 0.0f);

	// Store
	gs_rgb[idx * 3 + 0] = result.x;
	gs_rgb[idx * 3 + 1] = result.y;
	gs_rgb[idx * 3 + 2] = result.z;
	gs_opacity[idx] = opacities[idx];
	gs_depths[idx] = p_rend.y;
	gs_means3D[idx] = p_orig;
	gs_rend_means2D[idx] = { p_rend.x, p_rend.z };
	gs_tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	radii[idx] = radius;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, int PPU,
	float slice_height,
	float slice_thickness,
	const float3* __restrict__ gs_means3D,
    const glm::vec3* __restrict__ scales,
    const glm::vec4* __restrict__ rotations,
	const float* __restrict__ features,
	const float* __restrict__ opas,
	float* __restrict__ final_T,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_xyz[BLOCK_SIZE];
    __shared__ glm::vec3 collected_scales[BLOCK_SIZE];
    __shared__ glm::vec4 collected_rotations[BLOCK_SIZE];
	__shared__ float collected_opas[BLOCK_SIZE];

	// Initialize helper variables
	float pix_rgba[4] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xyz[block.thread_rank()] = gs_means3D[coll_id];
            collected_scales[block.thread_rank()] = scales[coll_id];
            collected_rotations[block.thread_rank()] = rotations[coll_id];
            collected_opas[block.thread_rank()] = opas[coll_id];
		}
		// Wait for all threads to finish fetching
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			float3 pc_xyz = collected_xyz[j];
            glm::vec3 scale = collected_scales[j];

			// Limit the scale so that the minimum is exp(-6), for sampling
			scale.x = max(scale.x, 0.0025f);
			scale.y = max(scale.y, 0.0025f);
			scale.z = max(scale.z, 0.0025f);

            glm::vec4 rotation = collected_rotations[j];

			float3 rend_xyz = {
				((float)pix.x - W / 2.0f) / (float)PPU,
				slice_height,
				-  ((float)pix.y - H / 2.0f) / (float)PPU
			};

			float3 d = {
				rend_xyz.x - pc_xyz.x,
				rend_xyz.y - pc_xyz.y,
				rend_xyz.z - pc_xyz.z
			};

			// Manually convert quaternion to rotation matrix
            glm::quat q(rotation.x, rotation.y, rotation.z, rotation.w);
 			glm::mat3 rot_matrix = quaternionToRotationMatrix(q);

			// Calculate the normalizing factor for the Gaussian
			// This could be precomputed
			glm::vec3 rot_unit_y = rot_matrix * glm::vec3(0, 1, 0);
			glm::vec3 rot_sigmas = glm::vec3(
				rot_unit_y.x / scale.x,
				rot_unit_y.y / scale.y,
				rot_unit_y.z / scale.z
			);
			float norm_factor = sqrt((rot_sigmas.x * rot_sigmas.x + rot_sigmas.y * rot_sigmas.y + rot_sigmas.z * rot_sigmas.z)) / sqrt(2 * M_PI);

			// Integrate over the slice
			float integration_step = slice_thickness / 10.0f;

			// Compute intensity by integrate from d.z-slice_thickness/2 to d.z+slice_thickness/2
			float intensity = 0.0f;
			for (float y = d.y - slice_thickness / 2.0f; y < d.y + slice_thickness / 2.0f; y += integration_step)
			{
				glm::vec3 d_rot = rot_matrix * glm::vec3(d.x, y, d.z); // << maybe bug here
				intensity += exp(-0.5f * (
					(d_rot.x * d_rot.x) / (scale.x * scale.x) +
					(d_rot.y * d_rot.y) / (scale.y * scale.y) +
					(d_rot.z * d_rot.z) / (scale.z * scale.z)
				)) * integration_step * norm_factor;
			}

			// Obtain opa by multiplying with Gaussian opacity
			float opa = collected_opas[j];
			opa = opa * intensity;
			if (opa < (1.0f / 255.0f))
				continue;

			// Alpha blending for RGBA voxel c_out = (c_1*a_1+c_2*a_2*(1-a_1))/a_out
			float a_1 = pix_rgba[3];
			float a_2 = opa;
			float a_out = a_1 + (1 - a_1) * a_2;
			for (int ch = 0; ch < 3; ch++)
			{
				float c_1 = pix_rgba[ch];
				float c_2 = features[collected_id[j] * 3 + ch];
				pix_rgba[ch] = (c_1 * a_1 + c_2 * a_2 * (1 - a_1)) / a_out;
			}
			pix_rgba[3] = a_out;

		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		for (int ch = 0; ch < 4; ch++)
			out_color[ch * H * W + pix_id] = pix_rgba[ch];
	}
}

void FORWARD::render(
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
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H, PPU,
		slice_height,
		slice_thickness,
		gs_means3D,
		scales,
		rotations,
		colors,
		opas,
		final_T,
		out_color);
}

void FORWARD::preprocess(int P, int M,
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
	)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, M,
		means3D,
		scales,
		rotations,
		opacities,
		shs,
		W, H, PPU,
		slice_height,
		radii,
		gs_means3D,
		gs_rend_means2D,
		depths,
		cov3Ds,
		rgb,
		opas,
		grid,
		tiles_touched
		);
}