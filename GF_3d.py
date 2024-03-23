import numpy as np
import open3d as o3d


def gf3d_mine(pcd, radius=0.01, epsilon=0.1):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points_copy = np.asarray(pcd.points)
    points = np.asarray(pcd.points)
    #print(points.shape)
    num_points = len(pcd.points)

    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if k < 3:
            continue

        neighbors = points[idx, :]
        mean = np.mean(neighbors, 0)
        mean2 = np.mean(neighbors**2, 0)
        a = (mean2 - mean*mean)/((mean2 - mean*mean)+epsilon)
        b = mean - a @ mean
        points_copy[i] = a @ points[i] + b
    print(points_copy.shape)
    return points_copy
