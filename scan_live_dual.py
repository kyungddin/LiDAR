# Module Import
import open3d as o3d
import numpy as np
import time
from ouster.sdk import open_source
from ouster.sdk.core import XYZLut
from ouster.sdk.examples.client import configure_dual_returns  # 듀얼 리턴 설정

# Sensor Setup
HOST = "192.168.0.49"
configure_dual_returns(HOST)  # 센서를 듀얼 리턴 모드로 설정

# Get Metadata
source = open_source(HOST)
metadata = source.metadata[0]
xyz_lut = XYZLut(metadata, False)

# Visualizer Start
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Live Ouster PointCloud (Dual Return)', width=800, height=600)

# Visualizer Initialization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # 초기 포인트 1개
vis.add_geometry(pcd)

# 바닥 평면 생성
ground_plane = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.01, depth=4.0)
R = ground_plane.get_rotation_matrix_from_axis_angle([np.pi / 2, 0, 0])
ground_plane.rotate(R, center=(0, 0, 0))
ground_plane.translate(np.array([-1, 2, -1.0]))
ground_plane.paint_uniform_color([0.5, 0.5, 0.5])
vis.add_geometry(ground_plane)

# 렌더 옵션
render_opt = vis.get_render_option()
render_opt.point_size = 3.0

first_frame = True
max_distance = 5.0  # 최대 거리 제한 (단위: 미터)

try:
    for scans in source:
        # Dual Return 지원 여부 확인
        if len(scans) == 1:
            scan = scans[0]
            points = xyz_lut(scan).reshape(-1, 3)
        elif len(scans) == 2:
            scan0 = scans[0]
            scan1 = scans[1]
            points0 = xyz_lut(scan0).reshape(-1, 3)
            points1 = xyz_lut(scan1).reshape(-1, 3)
            points = np.vstack((points0, points1))  # 두 리턴 병합
        else:
            print(f"⚠️ 예상치 못한 리턴 수: {len(scans)}")
            continue

        # 거리 필터링
        distances = np.linalg.norm(points, axis=1)
        points = points[distances <= max_distance]
        if points.shape[0] == 0:
            continue

        # 다운샘플링
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(points)
        pcd_temp = pcd_temp.voxel_down_sample(voxel_size=0.01)

        # 노이즈 제거
        pcd_temp, ind = pcd_temp.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)

        # 평면 제거 (벽 감지)
        plane_model, inliers = pcd_temp.segment_plane(distance_threshold=0.05,
                                                      ransac_n=3,
                                                      num_iterations=2000)
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        dot_with_z = np.abs(normal @ np.array([0, 0, 1]))

        if dot_with_z < 0.3:  # Z축과 거의 수직 → 벽
            pcd_temp = pcd_temp.select_by_index(inliers, invert=True)

        # 포인트클라우드 업데이트
        pcd.points = pcd_temp.points
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # 첫 프레임에만 초기 시점 설정 (reset_view_point → reset_view_point_camera)
        if first_frame:
            ctr = vis.get_view_control()
            ctr.set_front([0.0, 0.0, -1.0])
            ctr.set_lookat([0.0, 0.0, 0.0])
            ctr.set_up([0.0, -1.0, 0.0])
            ctr.set_zoom(0.5)
            first_frame = False


        time.sleep(1 / 60)  # 60 FPS 제한

finally:
    vis.destroy_window()
