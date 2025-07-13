
import open3d as o3d
import numpy as np
import time
from ouster.sdk import open_source
from ouster.sdk.core import XYZLut
from ouster.sdk.examples.client import configure_dual_returns
configure_dual_returns("192.168.0.49")

source = open_source("192.168.0.49")
metadata = source.metadata[0]
xyz_lut = XYZLut(metadata, False)

for scans in source:
    if len(scans) == 1:
        scan = scans[0]
        points = xyz_lut(scan).reshape(-1, 3)
    elif len(scans) == 2:
        points0 = xyz_lut(scans[0]).reshape(-1, 3)
        points1 = xyz_lut(scans[1]).reshape(-1, 3)
        points = np.vstack((points0, points1))
    else:
        continue
    # 이후 포인트 클라우드 처리
