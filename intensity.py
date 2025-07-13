import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt
from ouster.sdk import open_source
from ouster.sdk.core import XYZLut

def get_grid_lineset(h_min_val, h_max_val, w_min_val, w_max_val, ignore_axis, grid_length=1, nth_line=5):
    if (h_min_val%2!=0):
        h_min_val -= 1
    if (h_max_val%2!=0):
        h_max_val += 1
    if (w_min_val%2!=0):
        w_min_val -= 1
    if (w_max_val%2!=0):
        w_max_val += 1
    
    num_h_grid = int(np.ceil((h_max_val - h_min_val) / grid_length))
    num_w_grid = int(np.ceil((w_max_val - w_min_val) / grid_length))
    
    num_h_grid_mid = num_h_grid // 2
    num_w_grid_mid = num_w_grid // 2
    
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
            else:
                pass                
            vertex_order_index += 1       
            
    next_h = [0, 1]
    next_w = [1, 0]
    grid_lines = []
    for h in range(num_h_grid):
        for w in range(num_w_grid):
            here_h = h
            here_w = w
            for i in range(2):
                there_h = h + next_h[i]
                there_w = w +  next_w[i]   
                if (0 <= there_h and there_h < num_h_grid) and (0 <= there_w and there_w < num_w_grid):
                    if ((here_h % nth_line) == 0) and ((here_w % nth_line) == 0):
                        grid_nth_lines.append([grid_vertexes_order[here_h][here_w], grid_vertexes_order[there_h][there_w]])
                    elif ((here_h % nth_line) != 0) and ((here_w % nth_line) == 0) and i == 1:
                        grid_nth_lines.append([grid_vertexes_order[here_h][here_w], grid_vertexes_order[there_h][there_w]])
                    elif ((here_h % nth_line) == 0) and ((here_w % nth_line) != 0) and i == 0:
                        grid_nth_lines.append([grid_vertexes_order[here_h][here_w], grid_vertexes_order[there_h][there_w]])
                    else:
                        grid_lines.append([grid_vertexes_order[here_h][here_w], grid_vertexes_order[there_h][there_w]])

    color = (0.8, 0.8, 0.8)
    colors = [color for i in range(len(grid_lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid_vertexes),
        lines=o3d.utility.Vector2iVector(grid_lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    color = (255, 0, 0)
    colors = [color for i in range(len(grid_nth_lines))]
    line_nth_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid_vertexes),
        lines=o3d.utility.Vector2iVector(grid_nth_lines),
    )
    line_nth_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set, line_nth_set

# 센서 IP 주소
HOST = "192.168.0.49"
source = open_source(HOST)
metadata = source.metadata[0]
xyz_lut = XYZLut(metadata, False)

# Open3D 시각화 창 생성
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Live Ouster PointCloud', width=800, height=600)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # 초기 포인트 1개
pcd.colors = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # 초기 색상

vis.add_geometry(pcd)

# 바닥 평면 생성 및 시각화
ground_plane = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.01, depth=4.0)
R = ground_plane.get_rotation_matrix_from_axis_angle([np.pi / 2, 0, 0])
ground_plane.rotate(R, center=(0, 0, 0))
ground_plane.translate(np.array([-1, 2, -1.0]))
ground_plane.paint_uniform_color([0.5, 0.5, 0.5])
vis.add_geometry(ground_plane)

render_opt = vis.get_render_option()
render_opt.point_size = 3.0

first_frame = True
max_distance = 10  # 최대 거리 (m)

try:
    for scans in source:
        scan = scans[0]
        xyz = xyz_lut(scan)  # (H, W, 3)

        intensity = scan.field("REFLECTIVITY")  # (H, W)

        points = xyz.reshape(-1, 3)
        intensity = intensity.flatten().astype(float)

        if points.shape[0] == 0:
            continue

        # 거리 필터링
        distances = np.linalg.norm(points, axis=1)
        mask = distances <= max_distance
        points = points[mask]
        intensity = intensity[mask]

        if points.shape[0] == 0:
            continue

        # 다운샘플링
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(points)
        pcd_temp = pcd_temp.voxel_down_sample(voxel_size=0.01)

        # 노이즈 제거
        pcd_temp, ind = pcd_temp.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)

        # 평면 제거 (벽 제거)
        plane_model, inliers = pcd_temp.segment_plane(distance_threshold=0.05,
                                                     ransac_n=3,
                                                     num_iterations=2000)
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        dot_with_z = np.abs(normal @ np.array([0, 0, 1]))
        if dot_with_z < 0.3:
            pcd_temp = pcd_temp.select_by_index(inliers, invert=True)

        # 다운샘플링 + 노이즈 제거 후 점들 선택
        filtered_points = np.asarray(pcd_temp.points)

        # filtered_points에 대응하는 intensity 재계산
        # 원래 points 배열에 없어진 점들이 있으므로 intensity도 맞춰야 함
        # 가장 가까운 점을 매칭해서 intensity 값을 가져오는 방법 사용
        # KDTree 이용 (아래)
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, idxs = tree.query(filtered_points, k=1)
        filtered_intensity = intensity[idxs]

        # intensity 정규화 및 컬러맵 적용
        intensity_norm = (filtered_intensity - filtered_intensity.min()) / (np.ptp(filtered_intensity) + 1e-6)
        cmap = plt.get_cmap('jet')
        colors = cmap(intensity_norm)[:, :3]

        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        if first_frame:
            vis.reset_view_point(reset_bounding_box=True)
            first_frame = False

        time.sleep(1/60)

finally:
    vis.destroy_window()