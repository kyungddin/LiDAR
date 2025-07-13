# 일단은 폐기

import time
import numpy as np
import open3d as o3d

from ouster.sdk.core import XYZLut, UDPProfileLidar
from ouster.sdk.sensor import get_config, set_config, SensorPacketSource
from ouster.sdk.util import parsing

sensor_hostname = "192.168.0.49"  # ← LiDAR IP

def main():
    # 1. 센서 설정 가져오기
    cfg = get_config(sensor_hostname, active=True)

    # 2. 듀얼 리턴 UDP 프로파일 설정
    cfg.udp_profile_lidar = UDPProfileLidar.PROFILE_LIDAR_RNG19_RFL8_SIG16_NIR16_DUAL

    # 3. 설정 적용
    set_config(sensor_hostname, cfg, persist=False)

    # 4. 센서 패킷 소스 생성
    source = SensorPacketSource([sensor_hostname], config_timeout=10.0)

    # 5. 메타데이터(SensorInfo) 단일 객체로 추출
    sensor_infos = source.sensor_info
    if isinstance(sensor_infos, list):
        metadata = sensor_infos[0]
    else:
        metadata = sensor_infos

    # 6. LUT 생성
    xyz_lut = XYZLut(metadata, use_extrinsics=False)

    print("Sensor stream started... Press Ctrl+C to stop.")

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Dual Return LiDAR Stream")
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        first_frame = True  # 첫 프레임 여부 플래그

        for packets in source:
            if isinstance(packets, tuple):
                packets = [packets]

            # 듀얼 리턴 분리
            return0_pkts = [pkt for return_idx, pkt in packets if return_idx == 0]
            return1_pkts = [pkt for return_idx, pkt in packets if return_idx == 1]

            # 리턴 0 데이터 처리
            if return0_pkts:
                scan0 = parsing.packets_to_scan(return0_pkts, metadata)
                xyz0 = xyz_lut(scan0).reshape(-1, 3)
                color0 = np.tile([1, 0, 0], (xyz0.shape[0], 1))  # 빨간색
            else:
                xyz0 = np.empty((0, 3))
                color0 = np.empty((0, 3))

            # 리턴 1 데이터 처리
            if return1_pkts:
                scan1 = parsing.packets_to_scan(return1_pkts, metadata)
                xyz1 = xyz_lut(scan1).reshape(-1, 3)
                color1 = np.tile([0, 0, 1], (xyz1.shape[0], 1))  # 파란색
            else:
                xyz1 = np.empty((0, 3))
                color1 = np.empty((0, 3))

            # 두 리턴 데이터 합치기
            points = np.vstack((xyz0, xyz1))
            colors = np.vstack((color0, color1))

            # Open3D 포인트클라우드 업데이트
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            if first_frame:
                vis.reset_view_point(True)  # 뷰포인트 초기화
                first_frame = False

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Streaming stopped by user.")
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    main()
