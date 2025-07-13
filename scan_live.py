# Module Import
import open3d as o3d
import numpy as np
import time
from ouster.sdk import open_source
from ouster.sdk.core import XYZLut

# Get Sensor Info & MetaData
HOST = "192.168.0.49"
source = open_source(HOST)
metadata = source.metadata[0]
xyz_lut = XYZLut(metadata, False)

# Visualizer Start
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Live Ouster PointCloud', width=800, height=600)

# Visualizer Initialization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # 초기 포인트 1개
vis.add_geometry(pcd)

# 바닥 평면 생성
ground_plane = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.01, depth=4.0)
# Z축을 위로 향하게 XY → XZ 평면으로 회전
R = ground_plane.get_rotation_matrix_from_axis_angle([np.pi / 2, 0, 0])  # X축 기준 90도 회전
ground_plane.rotate(R, center=(0, 0, 0))  # 원점 기준 회전
# 위치 조정 (살짝 아래에 깔리도록)
ground_plane.translate(np.array([-1, 2, -1.0]))
# 색상 설정
ground_plane.paint_uniform_color([0.5, 0.5, 0.5])
# 시각화에 추가
vis.add_geometry(ground_plane)

# 점 크기 설정
render_opt = vis.get_render_option()
render_opt.point_size = 3.0  # 여기서 조정

# first_frame
first_frame = True

# max_distance 설정
max_distance = 5  # 최대 거리 제한 (단위: 미터)

# 반복문을 돌며 Scan 시작
try:
    for scans in source:
        scan = scans[0]
        xyz = xyz_lut(scan)  # (H, W, 3) 배열
        
        points = xyz.reshape(-1, 3)

        if points.shape[0] == 0:
            continue  # 포인트가 없으면 다음 프레임으로 (필수는 아님)

        # 거리 필터링: 원점 기준 max_distance 이상 점 제거
        distances = np.linalg.norm(points, axis=1)
        points = points[distances <= max_distance]

        if points.shape[0] == 0:
            continue  # 필터링 후 포인트 없으면 다음 프레임

        # 다운샘플링 추가
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(points)
        pcd_temp = pcd_temp.voxel_down_sample(voxel_size=0.01)  # 크기 조절 여기서!

        # 노이즈 제거 추가 (Statistical Outlier Removal)
        pcd_temp, ind = pcd_temp.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)

        # 평면 제거 (벽 탐지)
        # threshold 줄이거나 iteration 늘려서 더 넓은 거리 벽 제거
        plane_model, inliers = pcd_temp.segment_plane(distance_threshold=0.05,
                                                    ransac_n=3,
                                                    num_iterations=2000)

        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])

        # 평면의 법선이 거의 수평이라면 → 벽일 가능성 있음
        # 기준: Z축(상방향)과의 내적이 작으면 → 수직 방향
        dot_with_z = np.abs(normal @ np.array([0, 0, 1]))

        # 수직 방향 (즉, 벽이라 판단되면 제거)
        if dot_with_z < 0.3:  # z축과 거의 수직
            # 벽 인라이어 제외한 점만 사용
            pcd_temp = pcd_temp.select_by_index(inliers, invert=True)

        pcd.points = pcd_temp.points

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Error 방지를 위한 first_frame
        if first_frame:
            vis.reset_view_point(True)  # ✔️ 올바른 사용법
            first_frame = False

        # FPS 설정 (60프레임)
        time.sleep(1/60)

finally:
    vis.destroy_window()
