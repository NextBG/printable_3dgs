#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int M,
			int width, int height, int pix_per_unit,
			float slice_height,
			float slice_thickness,
			const float* means3D,
			const float* shs,
			const float* opacities,
			const float* scales,
			const float* rotations,
			float* out_color,
			int* radii = nullptr
			);
	};
};

#endif