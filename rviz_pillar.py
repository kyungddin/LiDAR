#!/usr/bin/env python
import rospy
import numpy as np
import struct
import torch
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2
from ouster.sdk import open_source
from ouster.sdk.core import XYZLut
import open3d as o3d
import tf.transformations as tft

# PointPillars 모델 전체 import (환경에 맞게 경로/클래스명 수정 필요)
from second.pytorch.models.pointpillars import PointPillars
from second.core import box_np_ops
from second.utils import box_utils  # 박스 후처리에 필요하면

# ---------------- 전처리 함수들 ----------------
def voxel_downsample(points, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down_pcd.points)

def remove_noise(points, nb_neighbors=25, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(cl.points)

def remove_vertical_planes(points, distance_threshold=0.05, ransac_n=3, num_iterations=2000, z_dot_threshold=0.3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    dot_with_z = abs(np.dot(normal, np.array([0, 0, 1])))
    if dot_with_z < z_dot_threshold:
        pcd = pcd.select_by_index(inliers, invert=True)
    return np.asarray(pcd.points)

def rgb_to_float(r, g, b):
    rgb = (r << 16) | (g << 8) | b
    return struct.unpack('f', struct.pack('I', rgb))[0]

def distance_to_color(dist, max_dist):
    ratio = np.clip(dist / max_dist, 0, 1)
    r = int(255 * (1 - ratio))
    g = 0
    b = int(255 * ratio)
    return r, g, b

def points_to_pointcloud2(points, max_dist=3.0, frame_id="map"):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.FLOAT32, 1),
    ]

    cloud_data = []
    for p in points:
        dist = np.linalg.norm(p)
        r, g, b = distance_to_color(dist, max_dist)
        rgb_float = rgb_to_float(r, g, b)
        cloud_data.append([p[0], p[1], p[2], rgb_float])

    cloud = pc2.create_cloud(header, fields, cloud_data)
    return cloud

# ---------------- 모델 로딩 ----------------
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # PointPillars 모델 클래스 인스턴스 생성 (환경에 맞게 인자 조정)
    model = PointPillars(num_input_features=4)  # 4 = x,y,z,intensity
    checkpoint_path = "/home/nsl/PointPillars/pretrained/epoch_160.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, device

# ---------------- PointPillars 입력 변환 및 추론 ----------------
def infer_pointpillars(model, device, points_with_intensity):
    """
    points_with_intensity: (N,4) numpy array - [x,y,z,intensity]
    """
    # 1. Point 범위 및 설정
    # (PointPillars 설정에 맞게 수정)
    x_min, x_max = 0, 70.4
    y_min, y_max = -40, 40
    z_min, z_max = -3, 1

    # 2. 범위 필터링
    mask = (
        (points_with_intensity[:, 0] >= x_min) & (points_with_intensity[:, 0] < x_max) &
        (points_with_intensity[:, 1] >= y_min) & (points_with_intensity[:, 1] < y_max) &
        (points_with_intensity[:, 2] >= z_min) & (points_with_intensity[:, 2] < z_max)
    )
    points = points_with_intensity[mask]

    if points.shape[0] == 0:
        return []

    # 3. voxelization (기본 voxel size 설정)
    voxel_size = [0.16, 0.16, 4]  # x,y,z
    grid_size = [
        int((x_max - x_min) / voxel_size[0]),
        int((y_max - y_min) / voxel_size[1]),
        int((z_max - z_min) / voxel_size[2])
    ]

    # voxelization: voxel에 포인트 할당 및 feature 계산 (간단화 버전)
    voxel_dict = {}
    for p in points:
        x_idx = int((p[0] - x_min) / voxel_size[0])
        y_idx = int((p[1] - y_min) / voxel_size[1])
        z_idx = int((p[2] - z_min) / voxel_size[2])
        key = (x_idx, y_idx, z_idx)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(p)

    # feature 만들기 (평균 등 간단히)
    voxels = []
    voxel_coords = []
    for key, pts in voxel_dict.items():
        pts_np = np.array(pts)
        feat = np.mean(pts_np, axis=0)  # (x,y,z,intensity)
        voxels.append(feat)
        voxel_coords.append(key)

    voxels = np.array(voxels)
    voxel_coords = np.array(voxel_coords)

    # 4. 입력 텐서 생성 (배치 1)
    # 여기서 PointPillars 모델 입력 형식에 맞게 수정 필요 (ex: batch_voxel_features, batch_voxel_coords, batch_voxel_num_points)
    # 간단 예시 (실제 모델 입력과 다를 수 있음)
    voxel_features = torch.tensor(voxels, dtype=torch.float32).unsqueeze(0).to(device)  # (1, voxel_num, feature_dim)
    voxel_coords = torch.tensor(voxel_coords, dtype=torch.int32).unsqueeze(0).to(device)  # (1, voxel_num, 3)

    # 실제로는 PointPillars 모델이 batch_voxel_features, batch_voxel_coords, batch_voxel_num_points 등 인자를 받음

    with torch.no_grad():
        # 모델 추론 (모델에 맞게 인자 조정 필요)
        preds = model(voxel_features, voxel_coords)  # output dict 예상

    # 5. 후처리 - 박스 및 점수 추출 (가상 코드)
    # 아래는 예시. 실제 후처리 코드/함수로 대체해야 함.
    boxes = preds.get('boxes', None)
    scores = preds.get('scores', None)
    labels = preds.get('labels', None)

    detections = []
    if boxes is not None:
        for i in range(boxes.shape[0]):
            bbox = boxes[i].cpu().numpy()  # [x,y,z,w,l,h,yaw]
            score = scores[i].item() if scores is not None else 1.0
            label = labels[i].item() if labels is not None else 0
            detections.append({
                "label": str(label),
                "score": score,
                "bbox": bbox.tolist()
            })

    return detections

# ---------------- RViz Marker 생성 ----------------
def create_bbox_marker(bbox, marker_id, frame_id="map"):
    x, y, z, w, l, h, yaw = bbox
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "detection"
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z + h / 2.0
    q = tft.quaternion_from_euler(0, 0, yaw)
    marker.pose.orientation.x = q[0]
    marker.pose.orientation.y = q[1]
    marker.pose.orientation.z = q[2]
    marker.pose.orientation.w = q[3]
    marker.scale.x = w
    marker.scale.y = l
    marker.scale.z = h
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 0.5
    marker.lifetime = rospy.Duration(0.1)
    return marker

# ---------------- 메인 노드 ----------------
def main():
    rospy.init_node("ouster_pointpillars_detector")
    pc_pub = rospy.Publisher("/ouster_points", PointCloud2, queue_size=1)
    marker_pub = rospy.Publisher("/detection_markers", MarkerArray, queue_size=1)

    HOST = "192.168.0.49"
    source = open_source(HOST)
    metadata = source.metadata[0]
    xyz_lut = XYZLut(metadata, False)

    model, device = load_model()

    max_distance = 10.0
    rate = rospy.Rate(10)

    try:
        for scans in source:
            scan = scans[0]
            xyz = xyz_lut(scan)
            points = xyz.reshape(-1, 3)
            points = points[~np.isnan(points).any(axis=1)]
            distances = np.linalg.norm(points, axis=1)
            points = points[distances <= max_distance]

            points = voxel_downsample(points)
            if points.shape[0] == 0:
                continue
            points = remove_noise(points)
            if points.shape[0] == 0:
                continue
            points = remove_vertical_planes(points)
            if points.shape[0] == 0:
                continue

            intensities = np.ones((points.shape[0], 1), dtype=np.float32)
            points_with_intensity = np.hstack([points, intensities])

            detections = infer_pointpillars(model, device, points_with_intensity)

            pc2_msg = points_to_pointcloud2(points, max_dist=max_distance)
            pc_pub.publish(pc2_msg)

            marker_array = MarkerArray()
            for i, det in enumerate(detections):
                bbox = det["bbox"]
                marker = create_bbox_marker(bbox, i)
                marker_array.markers.append(marker)
            marker_pub.publish(marker_array)

            rospy.sleep(0.05)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
