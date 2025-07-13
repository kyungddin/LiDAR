#!/usr/bin/env python

import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

from ouster.sdk import open_source
from ouster.sdk.core import XYZLut

import struct
import open3d as o3d#!/usr/bin/env python

import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

from ouster.sdk import open_source
from ouster.sdk.core import XYZLut

import struct
import open3d as o3d

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

def points_to_pointcloud2(points, frame_id="map"):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.FLOAT32, 1),
    ]

    cloud = pc2.create_cloud(header, fields, points)
    return cloud

def main():
    rospy.init_node("ouster_dual_return_publisher")
    pub = rospy.Publisher("/ouster_points", PointCloud2, queue_size=1)

    HOST = "192.168.0.49"
    source = open_source(HOST)
    metadata = source.metadata[0]
    xyz_lut = XYZLut(metadata, False)

    max_distance = 2.5
    rate = rospy.Rate(10)

    try:
        for scans in source:
            if len(scans) < 2:
                rospy.logwarn("Dual return not available. Only single return data received.")
                continue

            scan1, scan2 = scans
            xyz1 = xyz_lut(scan1).reshape(-1, 3)
            xyz2 = xyz_lut(scan2).reshape(-1, 3)

            xyz1 = xyz1[~np.isnan(xyz1).any(axis=1)]
            xyz2 = xyz2[~np.isnan(xyz2).any(axis=1)]

            dist1 = np.linalg.norm(xyz1, axis=1)
            dist2 = np.linalg.norm(xyz2, axis=1)
            xyz1 = xyz1[dist1 <= max_distance]
            xyz2 = xyz2[dist2 <= max_distance]

            xyz1 = voxel_downsample(xyz1)
            xyz2 = voxel_downsample(xyz2)

            if xyz1.shape[0] == 0 and xyz2.shape[0] == 0:
                continue

            xyz1 = remove_noise(xyz1)
            xyz2 = remove_noise(xyz2)

            xyz1 = remove_vertical_planes(xyz1)
            xyz2 = remove_vertical_planes(xyz2)

            points = []

            for p in xyz1:
                rgb = rgb_to_float(0, 0, 255)  # Blue for first return
                points.append([p[0], p[1], p[2], rgb])

            for p in xyz2:
                rgb = rgb_to_float(255, 0, 0)  # Red for second return
                points.append([p[0], p[1], p[2], rgb])

            if len(points) == 0:
                continue

            cloud_msg = points_to_pointcloud2(points)
            pub.publish(cloud_msg)

            rospy.loginfo("Published PointCloud2: %d first return + %d second return" % (len(xyz1), len(xyz2)))
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()


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

def points_to_pointcloud2(points, frame_id="map"):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.FLOAT32, 1),
    ]

    cloud = pc2.create_cloud(header, fields, points)
    return cloud

def main():
    rospy.init_node("ouster_dual_return_publisher")
    pub = rospy.Publisher("/ouster_points", PointCloud2, queue_size=1)

    HOST = "192.168.0.49"
    source = open_source(HOST)
    metadata = source.metadata[0]
    xyz_lut = XYZLut(metadata, False)

    max_distance = 2.5
    rate = rospy.Rate(10)

    try:
        for scans in source:
            if len(scans) < 2:
                rospy.logwarn("Dual return not available. Only single return data received.")
                continue

            scan1, scan2 = scans
            xyz1 = xyz_lut(scan1).reshape(-1, 3)
            xyz2 = xyz_lut(scan2).reshape(-1, 3)

            xyz1 = xyz1[~np.isnan(xyz1).any(axis=1)]
            xyz2 = xyz2[~np.isnan(xyz2).any(axis=1)]

            dist1 = np.linalg.norm(xyz1, axis=1)
            dist2 = np.linalg.norm(xyz2, axis=1)
            xyz1 = xyz1[dist1 <= max_distance]
            xyz2 = xyz2[dist2 <= max_distance]

            xyz1 = voxel_downsample(xyz1)
            xyz2 = voxel_downsample(xyz2)

            if xyz1.shape[0] == 0 and xyz2.shape[0] == 0:
                continue

            xyz1 = remove_noise(xyz1)
            xyz2 = remove_noise(xyz2)

            xyz1 = remove_vertical_planes(xyz1)
            xyz2 = remove_vertical_planes(xyz2)

            points = []

            for p in xyz1:
                rgb = rgb_to_float(0, 0, 255)  # Blue for first return
                points.append([p[0], p[1], p[2], rgb])

            for p in xyz2:
                rgb = rgb_to_float(255, 0, 0)  # Red for second return
                points.append([p[0], p[1], p[2], rgb])

            if len(points) == 0:
                continue

            cloud_msg = points_to_pointcloud2(points)
            pub.publish(cloud_msg)

            rospy.loginfo("Published PointCloud2: %d first return + %d second return" % (len(xyz1), len(xyz2)))
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
