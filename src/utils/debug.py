import open3d as o3d
import torch
import numpy as np
import sys
@torch.no_grad
def plot_rays(origins, 
              directions, 
              additional_points=None,
              show_directions=False, 
              directions_scale=0.1, 
              directions_subsample=1,
              show_coordinate_frame=True,
              coordinate_frame_size=1,
              show_box=True,
              box_size=1024,
              exit_program=True):
    
    geometries = []

    if show_box:
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=box_size, height=box_size, depth=box_size)
        mesh_box.translate(np.array([-box_size / 2, -box_size / 2, -box_size / 2]))
        mesh_box.compute_vertex_normals()
        geometries.append(mesh_box)
    
    if show_coordinate_frame:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_frame_size, 
                                                                             origin=[0, 0, 0])
        geometries.append(coordinate_frame)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(origins)
    geometries.append(point_cloud)

    if additional_points != None:
        half_idx = len(additional_points)//2
        additional_o = additional_points[:half_idx, :]
        additional_d = additional_points[half_idx:, :]

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(additional_o)

        colors = np.zeros((len(additional_o), 3))
        pcd2.colors = o3d.utility.Vector3dVector(colors)

        additional_lines = [[i, i + half_idx] for i in range(half_idx)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack((additional_o, additional_d)))
        line_set.lines = o3d.utility.Vector2iVector(additional_lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(additional_lines))]) 

        geometries.append(pcd2)
        geometries.append(line_set)
    
    
    if show_directions:
        lines = []
        line_points = []

        for origin, direction in zip(origins[::directions_subsample], 
                                     directions[::directions_subsample]):
            
            end_point = origin + direction * directions_scale
            line_points.append(origin)
            line_points.append(end_point)
            lines.append([len(line_points) - 2, len(line_points) - 1])

        line_points = np.array(line_points)
        lines = np.array(lines)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries)

    if exit_program:
        sys.exit()