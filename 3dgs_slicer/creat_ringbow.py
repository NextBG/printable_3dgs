import struct
import numpy as np


def create_colored_vertices_ply(output_file_path):
    vertices_data = []

    origin_x = -14
    origin_y = 0
    origin_z = 2
    
    spacing_x = 4
    spacing_z = 4
    '''
    Opacity from sigmoid activation 1 / (1 + exp(-vertex['opacity']))
    Sample from -10 to 10
    '''
    x = origin_x
    y = origin_y
    z = origin_z

    # ######### SLICE HEIGHT TEST #########
    # ROW_CNT = 7
    # for j in range(ROW_CNT):
    #     y = origin_y + j/(ROW_CNT-1) * 3.0 - 1.5
    #     r = 1
    #     g = -1
    #     b = -1
    #     opacity = 4
    #     scale_x = 0
    #     scale_y = 0
    #     scale_z = 0
    #     rot_w = 1.0
    #     rot_x = 0
    #     rot_y = 0
    #     rot_z = 0
    #     vertices_data.append((x, y, z, r, g, b, opacity, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z))
    #     x += 28.0/(ROW_CNT-1)
    # z += spacing_z
    # x = origin_x

    ######### OPACITY TEST #########
    ROW_CNT = 8
    y = origin_y
    # RED
    for j in range(ROW_CNT):
        r = 1
        g = -1
        b = -1
        opacity = 8 * (j/(ROW_CNT-1)) - 4
        scale_x = 0
        scale_y = 0
        scale_z = 0
        rot_w = 1.0
        rot_x = 0
        rot_y = 0
        rot_z = 0
        vertices_data.append((x, y, z, r, g, b, opacity, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z))
        x += spacing_x
    z += spacing_z
    x = origin_x

    # GREEN
    for j in range(ROW_CNT):
        r = -1
        g = 1
        b = -1
        opacity = 8 * (j/(ROW_CNT-1)) - 4
        scale_x = 0
        scale_y = 0
        scale_z = 0
        rot_w = 1.0
        rot_x = 0
        rot_y = 0
        rot_z = 0
        vertices_data.append((x, y, z, r, g, b, opacity, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z))
        x += spacing_x
    z += spacing_z
    x = origin_x

    # BLUE
    for j in range(ROW_CNT):
        r = -1
        g = -1
        b = 1
        opacity = 8 * (j/(ROW_CNT-1)) - 4
        scale_x = 0
        scale_y = 0
        scale_z = 0
        rot_w = 1.0
        rot_x = 0
        rot_y = 0
        rot_z = 0
        vertices_data.append((x, y, z, r, g, b, opacity, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z))
        x += spacing_x
    z += spacing_z
    x = origin_x

    # WHITE
    for j in range(ROW_CNT):
        r = 1
        g = 1
        b = 1
        opacity = 8 * (j/(ROW_CNT-1)) - 4
        scale_x = 0
        scale_y = 0
        scale_z = 0
        rot_w = 1.0
        rot_x = 0
        rot_y = 0
        rot_z = 0
        vertices_data.append((x, y, z, r, g, b, opacity, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z))
        x += spacing_x
    
    ######### SCALE TEST #########
    x = -13
    y = origin_y
    z = -8
    r = -1
    g = 1
    b = -1
    opacity = 4
    scale_x = -0.5
    scale_y = -0.5
    scale_z = -0.5
    vertices_data.append((x, y, z, r, g, b, opacity, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z))
    x += 4
    scale_x = 0
    scale_y = 0
    scale_z = 0
    vertices_data.append((x, y, z, r, g, b, opacity, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z))
    x += 7
    scale_x = 0.5
    scale_y = 0.5
    scale_z = 0.5
    vertices_data.append((x, y, z, r, g, b, opacity, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z))
    x += 11
    scale_x = 1
    scale_y = 1
    scale_z = 1
    vertices_data.append((x, y, z, r, g, b, opacity, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z))


    vertex_length = len(vertices_data)

    # Define the header
    header = f"""ply
format binary_little_endian 1.0
element vertex {vertex_length}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
""".encode('ascii')

    # Open the file in binary write mode
    with open(output_file_path, 'wb') as file:
        # Write the header
        file.write(header)
        
        # For each vertex, pack the data into binary format and write it
        for vertex_data in vertices_data:
            packed_vertex_data = struct.pack('<14f', *vertex_data)
            file.write(packed_vertex_data)

# Specify the path for the new PLY file
create_colored_vertices_ply('/home/caoruixiang/3dgs_renderer/models/point_cloud.ply')
