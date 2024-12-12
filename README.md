# Printable 3d gaussian splatting

## 3dgs_renderer
From 3dgs to sliced RGBA continuous color images
.ply -> .png seq

### Input
.ply file in ./3dgs_renderer/models/YOUR_MODEL_NAME.ply

### Output
png sequence in ./3dgs_renderer/output/YOUR_MODEL_NAME/000000.png

## 3dgs_slicer
From RGBA images to CMYKWCl discrete color images
.png seq -> .png seq

### Input
png sequence in ./3dgs_slicer/input/YOUR_MODEL_NAME/000000.png

### Output
png sequence in ./3dgs_slicer/output/YOUR_MODEL_NAME/000000.png