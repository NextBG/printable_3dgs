import os
import time
import torch
import cProfile
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser
from gaussian_model import GaussianModel
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    args = parser.parse_args()

    # CONSTANTS
    MM_PER_INCH = 25.4

    # USER SETTINGS
    zoom_factor = 0.03 / 1.99999                                      # 
    print_width_mm = 30                                         # mm
    print_height_mm = 30                                        # mm
    print_thickness_mm = 10                                     # mm

    # PRINTER SETTINGS
    dpi = 600                                                   # dpi
    print_layer_thickness_mm = 0.014                            # mm Stratasys:0.014mm, saina:  0.026mm

    model_width_m = print_width_mm / zoom_factor / 1000         # m
    model_height_m = print_height_mm / zoom_factor / 1000       # m
    model_thickness_m = print_thickness_mm / zoom_factor / 1000 # m
    model_layer_thickness_m = print_layer_thickness_mm / zoom_factor / 1000 # m

    print_width_pix = int(print_width_mm / MM_PER_INCH * dpi)   # pixel
    print_height_pix = int(print_height_mm / MM_PER_INCH * dpi) # pixel

    pix_per_m = int(print_width_pix/model_width_m)              # Pixel per m
    num_slices = int(model_thickness_m / model_layer_thickness_m)

    print(f"-------------------------- Parameters --------------------------")
    print(f"Print size: {print_width_mm} mm x {print_height_mm} mm x {print_thickness_mm} mm")
    print(f"Zoom factor: {zoom_factor}")
    print(f"Model size: {model_width_m:.3f} m x {model_height_m:.3f} m x {model_thickness_m:.3f} m")
    print(f"Pixel per meter: {pix_per_m} pix/m")
    print(f"Print layer thickness: {print_layer_thickness_mm} mm")
    print(f"Model layer thickness: {model_layer_thickness_m} m")
    print(f"dpi: {dpi}")
    print(f"Number of slices: {num_slices}")
    print(f"----------------------------------------------------------------")

    start_time = time.time()

    renderings = []
    current_slice_height = -model_thickness_m
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save the metadata as text file
    with open(args.output_dir + "/metadata.txt", "w") as file:
        file.write(f"slice_time:                {time.ctime()}\n")
        file.write(f"model_path:                {args.model_path}\n")
        file.write(f"zoom_factor:               {zoom_factor}\n")
        file.write(f"print_width_mm:            {print_width_mm}\n")
        file.write(f"print_height_mm:           {print_height_mm}\n")
        file.write(f"print_thickness_mm:        {print_thickness_mm}\n")
        file.write(f"dpi:                       {dpi}\n")
        file.write(f"print_layer_thickness_mm:  {print_layer_thickness_mm}\n")
        file.write(f"num_slices:                {num_slices}\n")

    # Load the model and slice it
    with torch.no_grad():
        gaussians = GaussianModel()
        gaussians.load_ply(args.model_path)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=print_width_pix,
            image_width=print_height_pix,
            pix_per_unit=pix_per_m,
            slice_thickness=print_layer_thickness_mm
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D = gaussians.get_xyz
        shs = gaussians.get_features
        opacity = gaussians.get_opacity
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

        for _ in tqdm(range(num_slices), desc="Slicing model"):
            rendering, _ = rasterizer(
                means3D = means3D,
                shs = shs,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                slice_height = current_slice_height
                )
            renderings.append(rendering)
            current_slice_height += model_layer_thickness_m

    rend_time = time.time()
            
    # Save the renderings as images        
    slice_idx = 0
    for rendering in tqdm(renderings, desc="Writing data to disk"):
        output_path = args.output_dir + f"/{slice_idx:06d}.png"
        slice_idx += 1
        torchvision.utils.save_image(rendering, output_path)

    write_time = time.time()

    print(f"Rendering time: {rend_time - start_time:.2f}s")
    print(f"Writing time: {write_time - rend_time:.2f}s")

if __name__ == "__main__":
    main()