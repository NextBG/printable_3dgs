#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& means3D,
  const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
  const int image_height,
  const int image_width,
  const int pix_per_unit,
  const float slice_height,
  const float slice_thickness,
	const torch::Tensor& sh
	)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  const int PPU = pix_per_unit;
  const float SLICE_HEIGHT = slice_height;
  const float SLICE_THICKNESS = slice_thickness;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({4, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		  M = sh.size(1);
    }

	  rendered = CudaRasterizer::Rasterizer::forward(
      geomFunc,
      binningFunc,
      imgFunc,
      P, M,
      W, H, PPU,
      SLICE_HEIGHT,
      SLICE_THICKNESS,
      means3D.contiguous().data<float>(),
      sh.contiguous().data_ptr<float>(),
      opacity.contiguous().data<float>(), 
      scales.contiguous().data_ptr<float>(),
      rotations.contiguous().data_ptr<float>(),
      out_color.contiguous().data<float>(),
      radii.contiguous().data<int>()
		);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}