import numpy as np
import open3d as o3d
import os
import bilateral_filter
from GF_3d import gf3d_mine


def merge(pcds_down):
    voxel_size = 0.05
    threshold = voxel_size*1.5
    res = pcds_down[0]

    for i in range(0, 5):
        trans_init = np.identity(4)
        #evaluation
        #evaluation = o3d.pipelines.registration.evaluate_registration(pcds_down[i], res, voxel_size*1.5, trans_init)
        #print("Evaluation of transformation epoch ", i, ": ", evaluation)
        #transformation
        icp_transform = o3d.pipelines.registration.registration_icp(pcds_down[i], res, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        pcds_down[i] = pcds_down[i].transform(icp_transform.transformation)
        res += pcds_down[i]
    
    return res



lista = os.listdir("Dataset/My_data")


#OPEN DATA
for label in lista:
    pcds = []
    for i in range(1, 6):
        color_raw = o3d.io.read_image("Dataset/My_data/" + label + "/" + str(i) + ".png")
        depth_raw = o3d.io.read_image("Dataset/My_data/" + label + "/" + str(i) + "d.png")

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=10000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)


        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)


        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcds.append(pcd)
        #o3d.visualization.draw_geometries([pcd])

        #o3d.io.write_point_cloud("unified.pcd", pcd)

    #o3d.visualization.draw_geometries([pcd])

    #MERGE ALL POINT CLOUD AND DOWNSAMPLE 
    output = merge(pcds)
    #o3d.visualization.draw_geometries([output])


    downpcd = output.voxel_down_sample(voxel_size=0.000000005)
    o3d.visualization.draw_geometries([downpcd])
    # Save the merged point cloud
    o3d.io.write_point_cloud("Results/" + label + "_minenoised.pcd", downpcd)


    #APPLY BILATERAL FILTER
    points_BL = bilateral_filter(downpcd, iterations=1, n_degree=10)
    # Save the merged point cloud
    o3d.io.write_point_cloud("Results/" + label + "_mine__BL_denoised.pcd", points_BL)

    #APPLY METHOD2
    points_M2 = gf3d_mine(downpcd)
    # Save the merged point cloud
    o3d.io.write_point_cloud("Results/" + label + "_mine_method2_denoised.pcd", points_M2)