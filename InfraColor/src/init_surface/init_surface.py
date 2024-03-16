import os
import numpy as np
import cv2

from ctypes import *

from numba import njit, prange

getxyz_so = cdll.LoadLibrary('./../lib/bin/Getxyz.so')


def for_show(lidar_coord_re, lidar_to_infra, pimg):
    """ for show """
    img_coord = np.dot(lidar_to_infra, lidar_coord_re)
    img_coord = np.transpose(img_coord)
    # undistorted
    nimg = pimg.copy()
    # depth = local_lidar[1:2, :].transpose(1, 0)
    img_coord[:, 0] = img_coord[:, 0] / img_coord[:, 2]
    img_coord[:, 1] = img_coord[:, 1] / img_coord[:, 2]

    nmask0 = np.logical_and(img_coord[:, 0] >= 0, img_coord[:, 0] < nimg.shape[1])
    nmask1 = np.logical_and(img_coord[:, 1] >= 0, img_coord[:, 1] < nimg.shape[0])
    nmask = np.logical_and.reduce((nmask0, nmask1))
    img_coord = img_coord[nmask, :]
    img_coord = img_coord.astype(np.int)

    for m in range(img_coord.shape[0]):
        # if x[m] > 0 and x[m] < img.shape[1]and y[m] >= 0 and y[m] < img.shape[0]:
        cv2.circle(nimg, (img_coord[m, 0], img_coord[m, 1]), 1, (0, 255, 0), 1)
    cv2.imshow("image", nimg)
    cv2.waitKey(0)


def for_show_uv(lidar_coord_re, lidar_to_infra, pimg):
    """ for show
    input: points Nx4
           P 3x4
    """
    h, w = pimg.shape[:2]

    # uv = np.matmul(infra_k, T_cam.transpose(1, 0)).transpose(1, 0)
    uv = np.matmul(lidar_to_infra, lidar_coord_re.transpose()).transpose(1, 0)
    # depth = lidar[:, 2:3]  # height
    uv[:, :2] /= uv[:, 2:]
    uv = uv.astype(np.int)

    mask0 = np.logical_and(uv[:, 0] >= 0, uv[:, 0] < w)
    mask1 = np.logical_and(uv[:, 1] >= 0, uv[:, 1] < h)
    # mask2 = depth[:, 0] > 0
    # mask3 = depth[:, 0] < 30.
    # mask = np.logical_and.reduce((mask0, mask1, mask2, mask3))
    mask = np.logical_and.reduce((mask0, mask1))
    # print(sum(mask))
    uv = uv[mask]
    lidar = lidar_coord_re[mask]
    pimg[uv[:, 1], uv[:, 0]] = (0, 255, 0)
    cv2.namedWindow('proj', cv2.WINDOW_NORMAL)
    cv2.imshow('proj', pimg)
    cv2.waitKey(0)


def InitialFace(im, lidar, infra_Tr, infra_k, lidar_to_infra):
    """
    input: points 4xN
           P 3x4  K 3x3  RT 3x4
    output: T_cam lidarPoints at img coord
            xyz all points in image
            xyz_face the nearest points in image surface
    """
    h, w = im.shape[:2]
    # lidar = lidar.transpose() / 1000.  # mm => m
    lidar = lidar.transpose()  # mm  4xN => Nx4  at lidar coord
    lidar[:, -1] = 1.  # Nx4

    # infra_Tr_ = np.concatenate((infra_Tr[:, :3], infra_Tr[:, 3:4] / 100.), axis=1)  # m
    infra_Tr_ = infra_Tr  # mm
    T_cam = np.matmul(infra_Tr_, lidar.transpose()).transpose(1, 0)  # lidarPoints 2 img coord  Nx3 mm

    # for_show_uv(lidar, lidar_to_infra, im)

    xyz_ = Initialxyz(T_cam, im, infra_k, lidar_to_infra)  # lidar && camera merge
    # xyz_ = Initialxyz(lidar, im, infra_k, lidar_to_infra)  # lidar && camera merge # mm  4xN  at lidar coord

    xyz = xyz_.transpose(2, 0, 1)
    if not xyz.flags['C_CONTIGUOUS']:
        xyz = np.ascontiguousarray(xyz, dtype=xyz.dtype)  # 如果不是C连续的内存，必须强制转换
    xyz_ctypes_ptr = cast(xyz.ctypes.data, POINTER(c_float))

    xyz_face = np.zeros((4, h, w), dtype=np.float32)
    if not xyz_face.flags['C_CONTIGUOUS']:
        xyz_face = np.ascontiguousarray(xyz_face, dtype=xyz.dtype)  # 如果不是C连续的内存，必须强制转换
    xyz_face_ctypes_ptr = cast(xyz_face.ctypes.data, POINTER(c_float))

    getxyz_so.GetXYZ(xyz_ctypes_ptr, h, w, 30, xyz_face_ctypes_ptr)
    xyz = xyz.transpose(1, 2, 0)  # Nx3 mm
    xyz_face = xyz_face.transpose(1, 2, 0)  # Nx3 mm
    cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
    cv2.imshow('depth', xyz_face[:, :, 2].astype(np.uint8))
    # cv2.imshow('depth', xyz[:, :, 2].astype(np.uint8))
    # cv2.imwrite("point_filter.jpg", xyz_face[:, :, 2].astype(np.uint8))
    cv2.waitKey()

    # return T_cam, xyz, xyz_face  # at img coord
    return T_cam, xyz, xyz_face  # at lidar coord


def InitialFace_m(im, lidar, infra_Tr, infra_k, lidar_to_infra):
    """
    input: points 4xN
           P 3x4  K 3x3  RT 3x4
    output: T_cam lidarPoints at img coord
            xyz all points in image
            xyz_face the nearest points in image surface
    """
    h, w = im.shape[:2]
    # lidar = lidar.transpose() / 1000.  # mm => m
    # lidar[:3, :] = lidar[:3, :] / 1000.  # mm => m
    lidar = lidar.transpose()  # mm  4xN => Nx4  at lidar coord
    lidar[:, -1] = 1.  # Nx4

    # infra_Tr_ = np.concatenate((infra_Tr[:, :3], infra_Tr[:, 3:4] / 100.), axis=1)  # m
    infra_Tr_ = infra_Tr  # mm
    T_cam = np.matmul(infra_Tr_, lidar.transpose()).transpose(1, 0)  # lidarPoints 2 img coord  Nx3 mm

    T_cam[:, :3] = T_cam[:, :3] / 1000.  # mm => m

    # for_show_uv(lidar, lidar_to_infra, im)

    """ input m """
    xyz_ = Initialxyz_m(T_cam, im, infra_k)  # lidar && camera merge
    # xyz_ = Initialxyz(lidar, im, infra_k, lidar_to_infra)  # lidar && camera merge # mm  4xN  at lidar coord

    xyz = xyz_.transpose(2, 0, 1)
    if not xyz.flags['C_CONTIGUOUS']:
        xyz = np.ascontiguousarray(xyz, dtype=xyz.dtype)  # 如果不是C连续的内存，必须强制转换
    xyz_ctypes_ptr = cast(xyz.ctypes.data, POINTER(c_float))

    xyz_face = np.zeros((4, h, w), dtype=np.float32)
    if not xyz_face.flags['C_CONTIGUOUS']:
        xyz_face = np.ascontiguousarray(xyz_face, dtype=xyz.dtype)  # 如果不是C连续的内存，必须强制转换
    xyz_face_ctypes_ptr = cast(xyz_face.ctypes.data, POINTER(c_float))

    getxyz_so.GetXYZ(xyz_ctypes_ptr, h, w, 30, xyz_face_ctypes_ptr)
    xyz = xyz.transpose(1, 2, 0)  # Nx3 m
    xyz_face = xyz_face.transpose(1, 2, 0)  # Nx3 m
    xyz_face[:, :, 2] = xyz_face[:, :, 2] * 10
    cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
    cv2.imshow('depth', xyz_face[:, :, 2].astype(np.uint8))
    # cv2.imshow('depth', xyz[:, :, 2].astype(np.uint8))
    # cv2.imwrite("point_filter.jpg", xyz_face[:, :, 2].astype(np.uint8))
    cv2.waitKey()

    # return T_cam, xyz, xyz_face  # at img coord
    return T_cam, xyz, xyz_face  # at lidar coord


def Initialxyz(points, im, infra_k, lidar_to_infra):
    h, w = im.shape[:2]
    # depth = points[:, 2:3]  # height
    uv = np.matmul(infra_k, points.transpose(1, 0)).transpose(1, 0)  # lidar 2 img && img distortion
    # uv = np.matmul(lidar_to_infra, points.transpose()).transpose(1, 0)  # lidar 2 img && img distortion
    uv = uv[:, :2] / uv[:, 2:]  # scale s

    mask0 = np.logical_and(uv[:, 0] >= 0, uv[:, 0] < w)
    mask1 = np.logical_and(uv[:, 1] >= 0, uv[:, 1] < h)
    # mask2 = depth[:, 0] > 0
    # mask = np.logical_and.reduce((mask0, mask1, mask2))
    mask = np.logical_and.reduce((mask0, mask1))
    # uv = uv[mask].astype(np.int)
    uv = uv[mask].astype('int')
    points = points[mask]  # only lidar_points in img window needed
    # 同一个点有多个数值 会产生覆盖
    xyz = np.zeros((h, w, 4), dtype=np.float32)  # 生成 h w 4 的三维数组
    xyz[uv[:, 1], uv[:, 0], :3] = points[:, :3]  # x y z  at img coord
    xyz[uv[:, 1], uv[:, 0], 3] = 1.  # rec
    return xyz


def Initialxyz_m(points, im, infra_k):  # img coord points
    h, w = im.shape[:2]
    depth = points[:, 2:3]
    uv = np.matmul(infra_k, points.transpose(1, 0)).transpose(1, 0)  # lidar 2 img && img distortion
    uv = uv[:, :2] / uv[:, 2:]  # scale s

    mask0 = np.logical_and(uv[:, 0] >= 0, uv[:, 0] < w)
    mask1 = np.logical_and(uv[:, 1] >= 0, uv[:, 1] < h)
    mask2 = depth[:, 0] > 0
    mask = np.logical_and.reduce((mask0, mask1, mask2))
    uv = uv[mask].astype(np.int)
    points = points[mask]  # only lidar_points in img window needed
    xyz = np.zeros((h, w, 4), dtype=np.float32)  # 生成 h w 4 的三维数组
    xyz[uv[:, 1], uv[:, 0], :3] = points[:, :3]  # x y z  at img coord
    xyz[uv[:, 1], uv[:, 0], 3] = 1.  # rec
    return xyz


thresh = 1.0 * 100
@njit(parallel=True)
def FilterLidar(points, im2, xyz_face, points3d, K):
    """
    input: points 4xN in camera coord
           P 3x4  K 3x3  RT 3x4
    """
    h, w, _ = im2.shape
    for i in prange(len(points)):
        point = points[i, :]
        s = K[2, 0] * point[0] + K[2, 1] * point[1] + K[2, 2] * point[2]

        u = int((K[0, 0] * point[0] + K[0, 1] * point[1] + K[0, 2] * point[2]) / s)
        v = int((K[1, 0] * point[0] + K[1, 1] * point[1] + K[1, 2] * point[2]) / s)
        if 0 <= v < h and 0 <= u < w:
            expected_z = xyz_face[v, u, 2]
            if points3d[v, u, 3] == 0:
                if expected_z - thresh < point[2] < expected_z + thresh or expected_z == 0:
                    points3d[v, u, :3] = point
                    points3d[v, u, 3] = 1.
            else:
                dist = np.sqrt(np.sum(np.power(point[:3], 2)))
                tmp_dist = np.sqrt(np.sum(np.power(xyz_face[v, u, :3], 2)))
                if dist < tmp_dist and expected_z - thresh < point[2] < expected_z + thresh:
                    points3d[v, u, :3] = point
                    points3d[v, u, 3] = 1.

    return points3d