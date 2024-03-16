'''
Author: Wangjie & FeiKeXin
Date: 2024-03-16 15:44:19
LastEditors: Wangjie & FeiKeXin
LastEditTime: 2024-03-16 17:38:42
FilePath: \scripts\color-image-ntd.py
Description: 

Copyright (c) 2024 by JOJO, All Rights Reserved. 
'''

""" add depthMap to filter and interpolation """

import sys
import os
import numpy as np
import cv2
from PIL import Image

import math
from ctypes import *
# getxyz_so = cdll.LoadLibrary('./../lib_surface/bin/Getxyz.so')

sys.path.append('./../src')
import core
from core import transform_utils, depth_map_utils, demo_utils
from datasets.kitti.obj import obj_utils, calib_utils
from datasets.kitti.raw import raw_utils
from ip_basic import ip_basic
from init_surface import init_surface

import timeit
import pypcd
import open3d as o3d


def ypr(azimuth1, pitch1, roll1):
    R_x = np.array([1, 0, 0,
                    0, math.cos(roll1), -math.sin(roll1),
                    0, math.sin(roll1), math.cos(roll1)])  # 计算旋转矩阵的X分量 其中droll=route_roll-cur_roll
    R_x = np.reshape(R_x, (3, 3))
    R_y = np.array([math.cos(pitch1), 0, math.sin(pitch1),
                    0, 1, 0,
                    -math.sin(pitch1), 0, math.cos(pitch1)])  # 计算旋转矩阵的Y分量 dpitch
    R_y = np.reshape(R_y, (3, 3))
    R_z = np.array([math.cos(azimuth1), -math.sin(azimuth1), 0,
                    math.sin(azimuth1), math.cos(azimuth1), 0,
                    0, 0, 1])  # 计算旋转矩阵的Z分量 dazi
    R_z = np.reshape(R_z, (3, 3))
    R1 = np.dot(R_y , R_x)
    R = np.dot(R_z, R1)  # R=Rz*Ry*Rx
    # R1 = np.dot(R_x , R_y)
    # R = np.dot(R_z, R1)  # R=Rz*Rx*Ry
    return R


def ypr_fh(azimuth1, pitch1, roll1):
    R_x = np.array([1, 0, 0,
                    0, math.cos(pitch1), -math.sin(pitch1),
                    0, math.sin(pitch1), math.cos(pitch1)])  # 计算旋转矩阵的Y分量 dpitch
    R_x = np.reshape(R_x, (3, 3))
    R_y = np.array([math.cos(roll1), 0, math.sin(roll1),
                    0, 1, 0,
                    -math.sin(roll1), 0, math.cos(roll1)])  # 计算旋转矩阵的X分量 其中droll=route_roll-cur_roll
    R_y = np.reshape(R_y, (3, 3))
    R_z = np.array([math.cos(azimuth1), -math.sin(azimuth1), 0,
                    math.sin(azimuth1), math.cos(azimuth1), 0,
                    0, 0, 1])  # 计算旋转矩阵的Z分量 dazi
    R_z = np.reshape(R_z, (3, 3))

    # R1 = np.dot(R_y , R_x)
    # R = np.dot(R_z, R1)  # R=Rz*Ry*Rx
    R1 = np.dot(R_x , R_y)
    R = np.dot(R_z, R1)  # R=Rz*Rx*Ry
    return R


# def proj(im, K, kc, lidar):
def proj(im, K, kc):
    """" Lidar 2 Image Rejection """
    im = cv2.undistort(im, K, kc)
    # h, w = im.shape[:2]
    # uv, points, _ = maskfunc(lidar, h, w)
    # im[uv[:, 1], uv[:, 0]] = (0, 255, 0)
    cv2.namedWindow('proj', cv2.WINDOW_NORMAL)
    cv2.imshow('proj', im)
    cv2.waitKey()
    return im


def load_image(file, mode):
    """Load an image from file."""
    return Image.open(file).convert(mode)


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    """ lidar2car param """
    # R=np.array([0.999985, -0.020112, -0.005508,
    #             0.020176,  0.999933,  0.011602,
    #             0.005507, -0.011603,  0.999918])
    # R=R.reshape((3, 3))
    # T=np.array([2.8314, 100.243, 156])
    # T=T.reshape((3, 1))
    # cam_velo_R=np.linalg.inv(R)
    # cam_velo_T=np.dot(-cam_velo_R,T)
    # cam_velo_RT=np.concatenate((cam_velo_R, cam_velo_T), axis=1)
    """ lidar param 1 """
    lidar_to_image=np.array([861.946, -2075.43, 8.23112, -357781,
                             517.531, -15.6589, -2038.21, -388976,
                             0.999304, -0.0354251, 0.0116753, -410.024])
    lidar_to_image=lidar_to_image.reshape((3,4))
    """ lidar param 2 """
    lidar_to_infra=np.array([606.241, -1553.13, 8.46919, 472557,
                             506.356, -4.98041, -1563.88, -309226,
                             0.99985, 0.0155749, -0.00759014, -262.093])
    lidar_to_infra=lidar_to_infra.reshape((3,4))
    """ camera param """
    image_k=np.array([2043.59670623238,	0,	934.964802530681,
                      0,	2044.11429254569,	493.928432605384,
                      0,	0,	1])
    image_k=image_k.reshape((3,3))
    image_kc = np.array([[-0.540353461869206, 0.318540371333151, -0.00198010936620195, 0.000583312217270064, -0.121043541312205]])

    infra_k=np.array([1562.43710, 0, 581.89547,
                      0, 1560.04477, 518.07218,
                      0, 0, 1])
    infra_k=infra_k.reshape((3,3))
    infra_kc = np.array([[-0.32788, 0.00000, 0.00000, -0.00000, 0.00000]])

    infra_Tr = np.matmul(np.linalg.inv(infra_k), lidar_to_infra)  # 3X4 T mm
    ones_metrics_ = np.array([0, 0,	0, 1]).reshape(1,4)
    infra_Tr_ = infra_Tr.copy()  # deepCopy
    # infra_Tr_[:, 3] = infra_Tr_[:, 3] / 100.    # T m ==> error
    infra_Tr_[:, 3] = infra_Tr_[:, 3]   # T mm
    infra_Tr_ = np.concatenate((infra_Tr_, ones_metrics_), axis=0)  # 4X4 T mm

    #################################################
    color_path = "/media/fkx/WorkStation/done/lbk/2022-10-24"
    infra_path = "/media/fkx/WorkStation/done/lbk/2022-10-25"

    image_path = color_path + "/undistort_Image/"
    cimgfiles = os.listdir(image_path)
    cimgfiles.sort()
    image2_path = infra_path + "/undistort_Infra/"
    imgfiles = os.listdir(image2_path)
    imgfiles.sort()

    save_path = "./result/"

    day_pose_file = "/home/fkx/fiction/lbk/2022-10-24/meta_data/calculated_slam_all_pose.txt"
    daylist = []
    with open(day_pose_file, "r", encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n')  # 删除换行符
            daylist.append(line.split(" "))
    daypose = np.array(daylist)

    night_pose_file = "/home/fkx/fiction/lbk/2022-10-25/meta_data/calculated_slam_all_pose.txt"
    nightlist = []
    with open(night_pose_file, "r", encoding='utf-8') as file:
        ndiff = 0
        for nline in file:
            nline = nline.strip('\n')  # 删除换行符
            nightlist.append(nline.split(" "))
    nightpose = np.array(nightlist)

    carcoord_path = color_path + "/LidarData_dense_10hz_70m/"
    files = os.listdir(carcoord_path)
    files.sort()

    carcoord_night_path = infra_path + "/LidarData_dense_10hz_70m/"
    nfiles = os.listdir(carcoord_night_path)
    nfiles.sort()


    """ 目的 给夜晚的点云数据赋色，从而实现红外的彩色化 """
    # test only once is okay
    for pfile in files:
        """ read lidar points """
        """ load pcd """
        # 过pip下载的pypcd似乎与python3并不兼容
        # car_coord = pypcd.PointCloud.from_path(carcoord_night_path+file+'.pcd')
        
    with open('/home/fkx/fiction/lbk/result-ntd-single.txt', 'r') as f:
        for line in f:
            nums = line.strip().split()
            pfile = nums[2]+'.bin'
            file = nums[0]+'.bin'
            cimgfile = nums[3]+'.jpg'
            imgfile = nums[1]+'.jpg'
            print(pfile)

            """ load bin """
            car_coord = np.fromfile(carcoord_night_path+file, dtype=np.int32)
            car_coord = car_coord.reshape((-1, 4))
            car_coord = car_coord[:, :3]

            car_coord = car_coord[car_coord[:, 1]>0, :3]
            ones = np.ones((car_coord.shape[0], 1))
            car_coord = np.concatenate((car_coord, ones), axis=1)  # Nx4 cm

            """ lidar 晚上局部坐标转图像 右前上转前左上 """
            local_lidar = car_coord[:, :3] * 10  # cm => mm
            # local_lidar = local_lidar[car_coord[:, 1]>0, :]
            local_lidar[:, 0] = -local_lidar[:, 0]
            local_lidar[:, [0, 1]] = local_lidar[:, [1, 0]]

            ones_metrics = np.ones((local_lidar.shape[0], 1))
            local_lidar = np.concatenate((local_lidar, ones_metrics), axis=1)  # Nx4
            local_lidar = np.transpose(local_lidar)  # Nx4 => 4xN  at lidar coord  # mm

            # # undistorted before
            nimg = cv2.imread(image2_path + imgfile)
            # cv2.imshow("infra", nimg)
            # cv2.waitKey()
            h, w = nimg.shape[:2]

            """ 去除重复深度的点 深度图 T_cam 前左上"""
            infra_image_shape = [h, w]
            T_cam = np.matmul(infra_Tr, local_lidar).transpose(1, 0)  # Nx3 at infra coord   # mm
            """ 相机坐标系右下前 xyz 点云区域筛选
                对于相机坐标系，右下前，Z轴就是深度 """
            cam0_point_cloud, _ = obj_utils.filter_pc_to_area(
                T_cam.transpose(1, 0), area_extents=np.asarray([[-50*1000, 50*1000], [-20*1000, 20*1000], [0, 100*1000]]))  # 3xN mm

            """ Project point cloud to create depth map 相机坐标系点云三维to二维图像 """
            projected_depths = depth_map_utils.project_depths(cam0_point_cloud, infra_k, infra_image_shape, max_depth=130)  # mm => m

            projected_depths = depth_map_utils.sky(projected_depths)

            # cv2.imshow("image", projected_depths)
            # cv2.waitKey(0)
            
            """  对于路面的分割插值需要处理  """
            final_depth_map, _ = ip_basic.fill_in_multiscale(projected_depths)

            """ 已获得该帧视角下，最近表面的点云，恢复成点云 相机坐标系右下前 xyz
            # Return the points in the provided camera frame
            # point_cloud_map = np.asarray([x, y, z]) """
            cam0_curr_pc = depth_map_utils.get_depth_point_cloud(final_depth_map, infra_k)  # 3xN # 含有大量零点 # m
            # Mask to valid points  # 零点去除
            valid_mask = (cam0_curr_pc[2] != 0)  # 已筛选过必定大于等于0 # cam0_curr_pc 940x1824=1714560
            cam0_point = cam0_curr_pc[:, valid_mask]  # 3xN  # m # 1134097  # 相机系下的点云

            """ 点云 路面分割 => 路面插值 => 合并点云  """
            ones_metrics = np.ones((1, cam0_point.shape[1]))
            cam0_point_ = np.concatenate((cam0_point, ones_metrics), axis=0)  # 3xN => 4XN  # m
            """ at lidar coord
            右下前 转 前左上 """
            # transForm = np.array([ 0,  0,  1,  0,
            #                       -1,  0,  0,  0,
            #                        0, -1,  0,  0,
            #                        0,  0,  0,  1])
            # transForm = transForm.reshape((4, 4))
            # cam0_point_ = np.matmul(transForm, cam0_point_)
            cam0_point_[:3, :] = cam0_point_[:3, :] * 1000
            car_coord_re = np.matmul(np.linalg.inv(infra_Tr_), cam0_point_).transpose(1, 0)  # 4XN => Nx4  # mm

            # lidar_coord_re = car_coord_re[:, :3]*1000  # Nx4 => NX3   # m => mm
            lidar_coord_re = car_coord_re[:, :3]  # Nx4 => NX3   # m => mm
            ones_metrics = np.ones((lidar_coord_re.shape[0], 1))
            # == local_lidar  # replace car_coord # mm
            lidar_coord_re = np.concatenate((lidar_coord_re, ones_metrics), axis=1)  # Nx4
            # lidar_coord_re = np.transpose(lidar_coord_re)  # Nx4 => 4xN

            """ Featrue Segment BGK"""
            """ project """
            # undistorted
            pimg = nimg.copy()
            mask_img = np.zeros((h, w), dtype=np.uint8)

            img_coord = np.dot(lidar_to_infra, lidar_coord_re.transpose())
            img_coord = np.transpose(img_coord)
            img_coord = img_coord[:, :2] / img_coord[:, 2:]  # scale s

            nmask0 = np.logical_and(img_coord[:, 0] >= 0, img_coord[:, 0] < nimg.shape[1])
            nmask1 = np.logical_and(img_coord[:, 1] >= 0, img_coord[:, 1] < nimg.shape[0])
            # nmask2 = depth[:, 0]>0
            nmask = np.logical_and.reduce((nmask0, nmask1))
            img_coord = img_coord[nmask, :]
            img_coord = img_coord.astype(int)
            """ 在点云不变的情况下，img_coord 记录在 infra 图像的 uv 坐标 """
            # for m in range(img_coord.shape[0]):
            #     # if x[m] > 0 and x[m] < img.shape[1]and y[m] >= 0 and y[m] < img.shape[0]:
            #     cv2.circle(nimg, (img_coord[m, 0], img_coord[m, 1]), 1, (0, 0, 225), 1)
            # #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.imshow("image", nimg)
            # cv2.waitKey(0)

            """ 在矩阵不变的情况下，nmask 记录在 infra 图像的 对应的点云坐标 """
            """ 滤除 并 继承 点云矩阵，保留 红外图像 窗口 """
            points_color = np.ones((lidar_coord_re.shape[0], 3), dtype=np.int32)
            points_color = points_color[nmask]

            lidar_coord_re = lidar_coord_re[nmask, :]

            """ lidar 前左上转右前上 """
            lidar_coord_re[:, 1] = -lidar_coord_re[:, 1]
            lidar_coord_re[:, [0, 1]] = lidar_coord_re[:, [1, 0]]  # Nx4

            ####### 读取白天和晚上的位姿 #################
            # mask_wj1 = nightpose[:, 0] == nfilename.split(".")[0]
            # mask_wj2 = nightpose[nightpose[:, 0] == nfilename.split(".")[0], :]
            # mask_wj3 = nightpose[nightpose[:, 0] == nfilename.split(".")[0], :][0,1:7]
            dpose = daypose[daypose[:, 0]==pfile.split(".")[0], :][0, 1:7].astype(float)
            npose = nightpose[nightpose[:, 0]==file.split(".")[0], :][0, 1:7].astype(float)

            ####### lidar 白天局部坐标转全局坐标 #################
            # d = np.array([dpose[0], dpose[1], dpose[2]])*100  # 单位 m ==> cm
            d = np.array([dpose[0], dpose[1], dpose[2]])*1000  # 单位 m ==> mm
            # d = np.array([dpose[0], dpose[1], dpose[2]])  # 单位 m
            d = np.reshape(d, (3, 1))
            R = ypr_fh(dpose[5], dpose[4], dpose[3])
            RT = np.concatenate((R, d), axis=1)  # 局部转全局矩阵 3乘4
            day_RT = np.zeros((4, 4))
            day_RT[:3, :4] = RT
            day_RT[3, 3] = 1
            # global_lidar = np.dot(RT, car_coord.transpose((1,0))).transpose((1, 0))  # 3*4 4*n  3*n n*3

            ####### lidar 全局坐标白天转晚上局部 #################
            # n_d = np.array([npose[0], npose[1], npose[2]]) * 100  # 单位 m ==> cm
            n_d = np.array([npose[0], npose[1], npose[2]]) * 1000  # 单位 m ==> mm
            # n_d = np.array([npose[0], npose[1], npose[2]])  # 单位 m
            n_d = np.reshape(n_d, (3, 1))
            n_R = ypr_fh(npose[5], npose[4], npose[3])
            nRT = np.concatenate((n_R, n_d), axis=1)  # 局部转全局矩阵 3乘4
            night_RT = np.zeros((4, 4))
            night_RT[:3, :4] = nRT
            night_RT[3, 3] = 1

            AB_RT = np.dot(np.linalg.inv(day_RT), night_RT)  # 4x4 mm
            """ 和C有一点差距，米单位下，毫米级的差距 """
            local_lidar = np.dot(AB_RT, lidar_coord_re.transpose((1, 0))).transpose((1, 0))  # Nx4 mm
            # lidar_coord = local_lidar.transpose()  # Nx4 => 4xN mm
            lidar_coord = local_lidar  # Nx4

            # car_coord = local_lidar
            """ trans """
            # lidar_coord = car_coord[:, :3]*10  # before *100 now *10 m ==> mm
            lidar_coord[:, 0] = -lidar_coord[:, 0]
            lidar_coord[:, [0, 1]] = lidar_coord[:, [1, 0]]
            #
            # ones_metrics = np.ones((lidar_coord.shape[0], 1))
            # lidar_coord = np.concatenate((lidar_coord, ones_metrics), axis=1)
            # lidar_coord = np.transpose(lidar_coord)  # 3xN => 4XN
            lidar_coord = lidar_coord.transpose()  # Nx4 => 4xN  mm

            """ day lidar image """
            uv = np.dot(lidar_to_image, lidar_coord)
            uv = np.transpose(uv)
            # undistorted
            img = cv2.imread(image_path + cimgfile)

            # img = img.numpy()
            # depth1 = lidar_coord[0:1,:]
            uv[:, 0] = uv[:, 0] / uv[:, 2]
            uv[:, 1] = uv[:, 1] / uv[:, 2]

            mask0 = np.logical_and(uv[:, 0] >= 0 + img.shape[1] * 0.05, uv[:, 0] < img.shape[1] * 0.95)
            mask1 = np.logical_and(uv[:, 1] >= 0 + img.shape[1] * 0.05, uv[:, 1] < img.shape[0] * 0.95)
            # mask2=depth1[0,:]>0
            mask = np.logical_and.reduce((mask0, mask1))
            uv = uv[mask, :]
            uv = np.array(uv).astype('int')

            points_color = points_color[mask]
            points_color = img[uv[:, 1], uv[:, 0]]
            img_coord = img_coord[mask, :]

            for m in range(img_coord.shape[0]):
                # cv2.circle(pimg, (img_coord[m, 0], img_coord[m, 1]), 1, (int(points_color[m, 0]), int(points_color[m, 1]), int(points_color[m, 2])), 4)
                mask_img[img_coord[m, 1]][img_coord[m, 0]] = 1

            # cv2.imwrite('/home/fkx/fiction/lbk/result-mask/'+imgfile, pimg)

            # mask_img = np.stack([mask_img]*3, axis=-1)
            np.save('/home/fkx/fiction/lbk/mask/'+imgfile.replace('jpg', 'npy'), mask_img)

            # cv2.imshow("persudo", pimg*mask_img)
            # cv2.waitKey(0)

            # j = j + 1

