#!/usr/bin/env python

# ros와 numpy import
import rospy
import numpy as np

# Rviz 포인트 클라우드 시각화 관련 라이브러리 
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

# ouster SDK
from ouster.sdk import open_source
from ouster.sdk.core import XYZLut

# open3D 및 struct
import struct
import open3d as o3d

# voxel downsample 함수
def voxel_downsample(points, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down_pcd.points)

# 노이즈 제거 함수
def remove_noise(points, nb_neighbors=25, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(cl.points)

# 평면 제거 함수
def remove_vertical_planes(points, distance_threshold=0.05, ransac_n=3, num_iterations=2000, z_dot_threshold=0.3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)

    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    dot_with_z = abs(np.dot(normal, np.array([0, 0, 1])))

    # 수직 평면 (벽)일 경우 inliers 제거
    if dot_with_z < z_dot_threshold:
        pcd = pcd.select_by_index(inliers, invert=True)

    return np.asarray(pcd.points)

# rgb 값을 float으로 변환하는 함수
def rgb_to_float(r, g, b):
    rgb = (r << 16) | (g << 8) | b
    return struct.unpack('f', struct.pack('I', rgb))[0]

# 거리별 색상값 반영 함수
def distance_to_color(dist, max_dist):
    ratio = np.clip(dist / max_dist, 0, 1)
    r = int(255 * (1 - ratio))
    g = 0
    b = int(255 * ratio)
    return r, g, b

# numpy 포인트 배열을 ros의 포인트 클라우드로 변환
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

# 메인 함수
def main():
    rospy.init_node("ouster_publisher")
    pub = rospy.Publisher("/ouster_points", PointCloud2, queue_size=1)

    HOST = "192.168.0.49"
    source = open_source(HOST)
    metadata = source.metadata[0]
    xyz_lut = XYZLut(metadata, False)

    max_distance = 2.5
    rate = rospy.Rate(10)

    try:
        for scans in source:
            scan = scans[0]
            xyz = xyz_lut(scan)
            points = xyz.reshape(-1, 3)
            points = points[~np.isnan(points).any(axis=1)]
            distances = np.linalg.norm(points, axis=1)
            points = points[distances <= max_distance]

            # 다운샘플링
            points = voxel_downsample(points, voxel_size=0.05)
            if points.shape[0] == 0:
                continue

            # 노이즈 제거
            points = remove_noise(points, nb_neighbors=25, std_ratio=2.0)
            if points.shape[0] == 0:
                continue

            # 벽(수직 평면) 제거
            points = remove_vertical_planes(points, distance_threshold=0.05, ransac_n=3, num_iterations=2000, z_dot_threshold=0.3)
            if points.shape[0] == 0:
                continue

            pc2_msg = points_to_pointcloud2(points, max_dist=max_distance, frame_id="map")
            pub.publish(pc2_msg)
            rospy.loginfo("Published Ouster PointCloud2 with %d points" % points.shape[0])

            rate.sleep()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()