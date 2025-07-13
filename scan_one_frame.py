# Module Import
from ouster.sdk import client
from ouster.sdk.client import Scans
from ouster.sdk import open_source
from ouster.sdk.core import XYZLut
import open3d as o3d
import numpy as np

# Ouster LiDAR 불러오기
HOST = "192.168.0.49"
source = open_source(HOST) # 제너레이터로, 매 프레임마다 스캔 데이터 생성

# 첫번째 LiDAR의 metadata를 가져오고, 이것의 Range Data 추출
metadata = source.metadata[0]
xyz_lut = XYZLut(metadata, False)

for scans in source:

    # 첫번째 LiDAR의 스캔데이터
    scan = scans[0]
    xyz = xyz_lut(scan) 

    # Open3D로 시각화
    pcd_np = xyz.reshape(-1, 3) # Range Data 통해 Point Cloud 생성
    pcd = o3d.geometry.PointCloud() # 포인트클라우드 객체 생성
    pcd.points = o3d.utility.Vector3dVector(pcd_np) # open3D 벡터 변환
    o3d.visualization.draw_geometries([pcd]) # 시각화
    
    break
