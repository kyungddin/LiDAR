import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt
from ouster.sdk import open_source
from ouster.sdk.core import XYZLut
from scipy.spatial import cKDTree

def get_grid_lineset(h_min_val, h_max_val, w_min_val, w_max_val, ignore_axis, grid_length=1, nth_line=5):
    if (h_min_val % 2 != 0):
        h_min_val -= 1
    if (h_max_val % 2 != 0):
        h_max_val += 1
    if (w_min_val % 2 != 0):
        w_min_val -= 1
    if (w_max_val % 2 != 0):
        w_max_val += 1
    
    num_h_grid = int(np.ceil((h_max_val - h_min_val) / grid_length))
    num_w_grid = int(np.ceil((w_max_val - w_min_val) / grid_length))
    
    grid_vertexes_order = np.zeros((num_h_grid, num_w_grid)).astype(np.int16)
    grid_vertexes = []
    vertex_order_index = 0
    
    for h in range(num_h_grid):
        for w in range(num_w_grid):
            grid_vertexes_order[h][w] = vertex_order_index
            if ignore_axis == 0:
                grid_vertexes.append([0, grid_length*w + w_min_val, grid_length*h + h_min_val])
            elif ignore_axis == 1:
                grid_vertexes.append([grid_length*h + h_min_val, 0, grid_length*w + w_min_val])
            elif ignore_axis == 2:
                grid_vertexes.append([grid_length*w + w_min_val, grid_length*h + h_min_val, 0])
            vertex_order_index += 1       
            
    next_h = [0, 1]
    next_w = [1, 0]
    grid_lines = []
    grid_nth_lines = []
    for h in range(num_h_grid):
        for w in range(num_w_grid):
            for i in range(2):
                there_h = h + next_h[i]
                there_w = w + next_w[i]   
                if (0 <= there_h < num_h_grid) and (0 <= there_w < num_w_grid):
                    if ((h % nth_line) == 0 and (w % nth_line) == 0) or \
                       ((h % nth_line) != 0 and (w % nth_line) == 0 and i == 1) or \
                       ((h % nth_line) == 0 and (w % nth_line) != 0 and i == 0):
                        grid_nth_lines.append([grid_vertexes_order[h][w], grid_vertexes_order[there_h][there_w]])
                    else:
                        grid_lines.append([grid_vertexes_order[h][w], grid_vertexes_order[there_h][there_w]])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid_vertexes),
        lines=o3d.utility.Vector2iVector(grid_lines),
    )
    line_set.colors = o3d.utility.Vector3dVector([(0.8, 0.8, 0.8)] * len(grid_lines))
    
    line_nth_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid_vertexes),
        lines=o3d.utility.Vector2iVector(grid_nth_lines),
    )
    line_nth_set.colors = o3d.utility.Vector3dVector([(1.0, 0.0, 0.0)] * len(grid_nth_lines))
    
    return line_set, line_nth_set


# 센서 연결
HOST = "192.168.0.49"
source = open_source(HOST)
metadata = source.metadata[0]
xyz_lut = XYZLut(metadata, False)

# 시각화 설정
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Live Ouster PointCloud', width=800, height=600)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
pcd.colors = o3d.utility.Vector3dVector(np.zeros((1, 3)))
vis.add_geometry(pcd)

render_opt = vis.get_render_option()
render_opt.point_size = 3.0

first_frame = True
max_distance = 10  # 최대 거리 (m)

# 변수 초기화
lineset_yz = lineset_zx = lineset_xy = None
lineset_nth_yz = lineset_nth_zx = lineset_nth_xy = None

try:
    for scans in source:
        scan = scans[0]
        xyz = xyz_lut(scan)
        intensity = scan.field("REFLECTIVITY")

        points = xyz.reshape(-1, 3)
        intensity = intensity.flatten().astype(float)

        if points.shape[0] == 0:
            continue

        distances = np.linalg.norm(points, axis=1)
        mask = distances <= max_distance
        points = points[mask]
        intensity = intensity[mask]

        if points.shape[0] == 0:
            continue

        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(points)
        pcd_temp = pcd_temp.voxel_down_sample(voxel_size=0.01)
        pcd_temp, ind = pcd_temp.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)

        # 평면 제거
        plane_model, inliers = pcd_temp.segment_plane(distance_threshold=0.05,
                                                     ransac_n=3,
                                                     num_iterations=2000)
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        dot_with_z = np.abs(normal @ np.array([0, 0, 1]))
        if dot_with_z < 0.3:
            pcd_temp = pcd_temp.select_by_index(inliers, invert=True)

        filtered_points = np.asarray(pcd_temp.points)

        tree = cKDTree(points)
        dists, idxs = tree.query(filtered_points, k=1)
        filtered_intensity = intensity[idxs]

        intensity_norm = (filtered_intensity - filtered_intensity.min()) / (np.ptp(filtered_intensity) + 1e-6)
        cmap = plt.get_cmap('jet')
        colors = cmap(intensity_norm)[:, :3]

        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis.update_geometry(pcd)

        if first_frame:
            # 그리드 범위 설정 (초기 한 번만)
            x_vals = filtered_points[:, 0]
            y_vals = filtered_points[:, 1]
            z_vals = filtered_points[:, 2]

            x_min_val, x_max_val = int(np.floor(x_vals.min())), int(np.ceil(x_vals.max()))
            y_min_val, y_max_val = int(np.floor(y_vals.min())), int(np.ceil(y_vals.max()))
            z_min_val, z_max_val = int(np.floor(z_vals.min())), int(np.ceil(z_vals.max()))

            lineset_yz, lineset_nth_yz = get_grid_lineset(z_min_val, z_max_val, y_min_val, y_max_val, 0, grid_length=0.5)
            lineset_zx, lineset_nth_zx = get_grid_lineset(x_min_val, x_max_val, z_min_val, z_max_val, 1, grid_length=0.5)
            lineset_xy, lineset_nth_xy = get_grid_lineset(y_min_val, y_max_val, x_min_val, x_max_val, 2, grid_length=0.5)

            vis.add_geometry(lineset_yz)
            vis.add_geometry(lineset_nth_yz)
            vis.add_geometry(lineset_zx)
            vis.add_geometry(lineset_nth_zx)
            vis.add_geometry(lineset_xy)
            vis.add_geometry(lineset_nth_xy)

            vis.reset_view_point(reset_bounding_box=True)
            first_frame = False

        vis.poll_events()
        vis.update_renderer()
        time.sleep(1 / 60)

finally:
    vis.destroy_window()
