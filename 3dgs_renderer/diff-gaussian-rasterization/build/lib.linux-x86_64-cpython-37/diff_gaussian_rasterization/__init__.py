from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        sh,
        opacities,
        scales,
        rotations,
        slice_height,
        raster_settings
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            opacities,
            scales,
            rotations,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.pix_per_unit,
            slice_height,
            raster_settings.slice_thickness,
            sh
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for 
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        return color, radii

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    pix_per_unit: int
    slice_thickness: float

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, opacities, shs, scales, rotations, slice_height):
        
        raster_settings = self.raster_settings

        # Invoke C++/CUDA rasterization routine
        return _RasterizeGaussians.apply(
            means3D,
            shs,
            opacities,
            scales,
            rotations,
            slice_height,
            raster_settings
        )

