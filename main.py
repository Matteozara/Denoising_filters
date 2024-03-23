import numpy as np
import open3d as o3d
import copy
import os
from bilateral_filter import bilateral_filter
from GF_3d import gf3d_mine
from evaluation import eval_score


def apply_noise(pcd, mu=0.0, sigma=0.00000001):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

def merge(pcds_down):
    voxel_size = 0.05
    threshold = voxel_size*1.5
    res = pcds_down[0]

    for i in range(0, 5):
        trans_init = np.identity(4)
        #evaluation
        #print("Evaluation of transformation epoch ", i)
        #evaluation = o3d.pipelines.registration.evaluate_registration(pcds_down[i], res, voxel_size*1.5, trans_init)
        #print(evaluation)

        #transformation
        icp_transform = o3d.pipelines.registration.registration_icp(pcds_down[i], res, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        pcds_down[i] = pcds_down[i].transform(icp_transform.transformation)
        res += pcds_down[i]
    
    return res


lista = os.listdir("Dataset/NYU")


#OPEN DATA
for label in lista:
    pcds = []
    for i in range(1, 6):
        color_raw = o3d.io.read_image("Dataset/NYU/" + label + "/" + str(i) + ".jpg")
        depth_raw = o3d.io.read_image("Dataset/NYU/" + label + "/" + str(i) + ".png")

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=10000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)


        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcds.append(pcd)


        #o3d.visualization.draw_geometries([pcd])

    #MERGE ALL POINT CLOUD AND DOWNSAMPLE 
    output = merge(pcds)
    o3d.visualization.draw_geometries([output])

    downpcd = output.voxel_down_sample(voxel_size=0.000000005)
    o3d.visualization.draw_geometries([downpcd])

    # Save the merged point cloud
    o3d.io.write_point_cloud("Results/" + label + "_grandtrith.pcd", downpcd)

    #ADD NOISE
    noisy = apply_noise(downpcd)
    # Save the merged point cloud
    o3d.io.write_point_cloud("Results/" + label + "_noised.pcd", noisy)

    #APPLY BILATERAL FILTER
    points_BL = bilateral_filter(noisy, iterations=1, n_degree=10)

    # Save the merged point cloud
    o3d.io.write_point_cloud("Results/" + label + "_BL_denoised.pcd", points_BL)

    #APPLY METHOD2
    points_M2 = gf3d_mine(noisy)

    # Save the merged point cloud
    o3d.io.write_point_cloud("Results/" + label + "_method2_denoised.pcd", points_M2)

    #EVALUATION 
    print("EVALUATION ", label)
    print("Evaluation Bilateral filter")
    eval_score(downpcd, points_BL)

    print("")

    print("Evaluation 3D Guided filter")
    eval_score(downpcd, points_M2)
    

    print("-------------------------------------------")

