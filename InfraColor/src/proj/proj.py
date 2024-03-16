import numpy as np
import torch

'''This project is used to create point pairs automatically according to the calibration results'''
import os
from os.path import join
import glob
from PIL import Image
import cv2
import cv2 as cv

import random
from tqdm import tqdm
import chardet
from copy import copy

color = np.array([
        [0, 255, 0],
        [255, 255, 0],
        [0, 0, 255],
        [255, 0, 255],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])
# # Kitti
# targetsize=(376, 1241)
# srcsize=(376, 1241)
# Infra
targetsize=(1024,1280)
srcsize=(1024,1280)
# # Color
# targetsize=(940,1824)
# srcsize=(940,1824)

rh = targetsize[0]/srcsize[0]
rw = targetsize[1]/srcsize[1]
proj_h = targetsize[0]
proj_w = targetsize[1]
maxDis = 100
ObjDis = 50
# velodyne
fov_up = 20.0
fov_down = -20.0
# rs128
fov_up = 15.0
fov_down = -25.0
fov_up = fov_up / 180.0 * np.pi  # field of view up in rad
fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad


def run(points, show=False):
    '''
    INPUT:  lidarfilename: name of the lidar file  # Nx4
            resultfilename:name of the decetion result file
    OUTPUT: proj_xyz is the range image, resultID
    '''
    '''load lidar points&load results'''
    # 坐标系为 右下前
    ap = points.tolist()  # 将数组或者矩阵转换为列表
    points = points[:, 0:3]  # Nx4 => Nx3
  
    '''Calculating projection coordinates'''
    proj_xyz = np.full((proj_h, proj_w, 3), 0, dtype=np.float32)
    proj_x, proj_y = cal_position(points, flag=0)  # RangeImage 边界
    tmp = min(proj_y)
    proj_xyz[proj_y, proj_x] = (points)  # image store points
    proj_xyz = proj_xyz[:, :, :]
    # points=points.tolist()

    """ surface """
    for i in range(tmp, proj_h):
        for j in range(0, proj_w):
            if proj_xyz[i, j].any()==0:
                sum = 0
                cout = 0
                if (i - 1 )>0 and (i-1 )<proj_h and proj_xyz[i-1, j].any()!=0:
                    sum += proj_xyz[i-1, j]
                    cout = cout+1
                if (i + 1) > 0 and (i + 1) < proj_h and proj_xyz[i+1, j].any()!=0:
                    sum += proj_xyz[i + 1, j]
                    cout = cout + 1
                if (j - 1) > 0 and (j - 1) < proj_w and proj_xyz[i, j-1].any()!=0:
                    sum += proj_xyz[i, j-1]
                    cout = cout + 1
                if (j + 1) > 0 and (j + 1) < proj_w and proj_xyz[i, j+1].any()!=0:
                    sum += proj_xyz[i, j+1]
                    cout = cout + 1
                if(cout!=0):
                    proj_xyz[i, j] = sum/cout
                    a = [proj_xyz[i, j, 0], proj_xyz[i, j, 1], proj_xyz[i, j, 2], 1.0]
                    ap.append(a)  # Nx4
    points = np.array(ap, dtype=np.float32)[:, 0:3]  # Nx3

    """ for show twice RangeImage """
    proj_x, proj_y = cal_position(points, flag=0)
    tmp = min(proj_y)
    proj_xyz[proj_y, proj_x] = (points)
    proj_xyz = proj_xyz[:, :, :]

    if show:
        showim = (255 * (abs(proj_xyz) / maxDis)).astype(np.uint8)
        showim = cv2.applyColorMap(showim, 2)
    return np.array(ap, dtype=np.float32)  # Nx4 # 坐标系为 右下前


def hole_fill(depth_map, threshold_dist=10/1000, max_depth=100):
    """ depth_map in m """
    # # Convert to float32
    depths_in_ = np.float32(depth_map)
    depths_in = np.copy(depths_in_)
    # cv2.imshow("input_depth_map", depths_in)
    # cv2.waitKey()

    # Calculate a top mask 对 天空 空点 的 mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool)
    for pixel_col_idx in range(depths_in.shape[1]):
        pixel_col = depths_in[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)  # 返回第一个大于0.1的数
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    left_mask = np.ones(depths_in.shape, dtype=np.bool)  # [u, v]
    for pixel_col_idx in range(depths_in.shape[0]):  # traversal width
        pixel_col = depths_in[pixel_col_idx, :]  # traversal height
        left_pixel_col = np.argmax(pixel_col > 0.1)  # 返回第一个大于0.1的数
        left_mask[pixel_col_idx, 0:left_pixel_col] = False

    """ wait for right and bottom """
    # right_mask = np.ones(depths_in.shape, dtype=np.bool)  # [u, v]
    # for pixel_col_idx in range(depths_in.shape[0]-1, 0, -1):  # traversal width
    #     pixel_col = depths_in[pixel_col_idx, :]  # traversal height
    #     right_pixel_col = np.argmax(pixel_col > 0.1)  # 返回第一个大于0.1的数
    #     right_mask[pixel_col_idx, 0:left_pixel_col] = False

    """ only 2/3 up """
    brunch_mask = np.ones(depths_in.shape, dtype=np.bool)
    brunch_up = int(depths_in.shape[0] * 2/3)
    # for pixel_col_idx in range(depths_in.shape[1]):  # traversal width
    brunch_mask[0:brunch_up, :] = False
    # brunch_mask = ~brunch_mask

    """ 逐元素的
    np.logical_and()       与
    numpy.logical_or()     或
    numpy.logical_not()    非
     """
    # Get empty mask
    _mask_ = np.logical_and.reduce((top_mask, left_mask))
    _mask_ = np.logical_or.reduce((_mask_, brunch_mask))
    # valid_pixels = (depths_in > 0.1)
    # empty_pixels = ~valid_pixels & ~_mask_
    """ 
    ~_mask_  天空点 标记为 true
    ~valid_pixels  0深度点 标记为 true
     """
    # empty_pixels = ~empty_pixels
    # print(valid_pixels)
    # print(~valid_pixels)
    # exit(0)

    """ show emptyImage  
     raw empty pixel is equal to 0 """  # 0 -> black
    _h, _w = depths_in.shape[:2]
    # projected_depths = np.zeros(depths_in.shape)
    # projected_depths[~_mask_] = max_depth
    # depths_in[~_mask_] = max_depth
    # for i in range(0, _h):
    #     for j in range(0, _w):
    #         # if depths_in[i, j] == max_depth:
    #         if depths_in[i, j] > 20:
    #             depths_in[i, j] = 0
    #             # tmp = depths_in[i, j]
    # #             # cv2.waitKey()
    # cv2.imshow("select_depth_map", depths_in)
    # cv2.waitKey()

    """ surface """
    threshold_step = 10  # pixel  >> 5 10
    big_threshold_step = threshold_step * 2  # pixel
    min_threshold_dist = -threshold_dist
    big_threshold_dist = threshold_dist * 2
    depths_calcu_in = np.copy(depths_in)
    _h_part = int(_h * 3 / 4)  # from up to down
    # _w_part = int(_w * 3 / 4)

    for i in range(1, _h_part):  #
        for j in range(1, _w):
            """  """
            find_the_right_point = False
            if depths_in[i, j] != 0 and depths_in[i, j] != max_depth:
                """ fill 重复 
                应该设置已经赋值的点，不要在搜索 """
                # """  """
                find_the_right_need = False
                for j_step in range(0, threshold_step):
                    if (j + j_step) < _w:
                        for i_step in range(0, threshold_step):
                            if (i + i_step) < _h:
                                if depths_in[i + i_step, j + j_step] != 0:
                                    # tmp_judg = abs(depths_in[i, j] - depths_in[i + i_step, j + j_step])
                                    tmp_judg = depths_in[i + i_step, j + j_step] - depths_in[i, j]
                                    if tmp_judg > threshold_dist:
                                        depths_in[i + i_step, j + j_step] = 0
                # depths_in[i, j] = 0
    for i in range(_h_part, _h):  #
        for j in range(1, _w):
            """  """
            find_the_right_point = False
            if depths_in[i, j] != 0 and depths_in[i, j] != max_depth:
                for j_step in range(0, big_threshold_step):
                    if (j + j_step) < _w:
                        for i_step in range(0, big_threshold_step):
                            if (i + i_step) < _h:
                                if depths_in[i + i_step, j + j_step] != 0:
                                    # tmp_judg = abs(depths_in[i, j] - depths_in[i + i_step, j + j_step])
                                    tmp_judg = depths_in[i + i_step, j + j_step] - depths_in[i, j]
                                    if tmp_judg > threshold_dist:
                                        depths_in[i + i_step, j + j_step] = 0

    """ 超过 搜索像素范围 无法判断 共面 """
    
    depths_calcu_in = depths_in
   
    return depths_calcu_in


def cal_position(points, flag):  # 边界检查 RangeImage
    # 右下前转右前上
    x = points[:, 0]
    y = points[:, 2]
    z = -points[:, 1]

    depth = (np.linalg.norm(points, 2, axis=1))  # 求取范数
    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= 2 * proj_w  # in [0.0, W]
    proj_y *= proj_h  # in [0.0, H]

    proj_x = np.floor(proj_x)
    proj_y = np.floor(proj_y)
    ''' Limiting the axis range '''
    proj_x = np.minimum(proj_w - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    proj_y = np.minimum(proj_h - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    # if flag:
    #     self.points_lidar.append(int(proj_y))
    #     self.points_lidar.append(int(proj_x))

    return proj_x, proj_y


class MakePair:
    def Generator(self, im, lidar, pose, pseudo, mask_):  # lidar: 4 * n, already in cam0 coordinate
        h, w = im.shape[:2]

        # uv in im2
        # 此处确定坐标系为 右下前
        # 相机选择为 gray
        depth2 = lidar[2:3].transpose(1, 0)
        # gray 坐标系转到 color 坐标系
        uvs2 = torch.matmul(self.Matrix['P2'], lidar).transpose(1, 0)
        uvs2[:, :2] = uvs2[:, :2] / uvs2[:, 2:]
        uvs2[:, 2:] = 1.0
        uv2 = uvs2[:, :2]
        lidar_with_color = self.GetColor(lidar, uv2, depth2, im)

        # uv in im1
        rotated_lidar = torch.matmul(pose, lidar_with_color[:, :4].transpose(1, 0))
        depth1 = rotated_lidar[2:3].transpose(1, 0)
        uvs1 = torch.matmul(self.Matrix['P0'], rotated_lidar).transpose(1, 0)
        uvs1[:, :2] = uvs1[:, :2] / uvs1[:, 2:]
        uvs1[:, 2:] = 1.0
        uv1 = uvs1.int()

        # get corresponding pixel
        mask0 = (uv1[:, 0] >= 0) & (uv1[:, 0] < w)
        mask1 = (uv1[:, 1] >= 0) & (uv1[:, 1] < h)
        mask2 = depth1[:, 0] > 0
        mask = mask0 & mask1 & mask2

        uv1 = uv1[mask].long()
        lidar_with_color = lidar_with_color[mask]

        pseudo[uv1[:, 1], uv1[:, 0], :] = pseudo[uv1[:, 1], uv1[:, 0], :] + lidar_with_color[:, 4:]
        mask_[uv1[:, 1], uv1[:, 0], :] = mask_[uv1[:, 1], uv1[:, 0], :] + self.addOne
        return pseudo, mask_

    def GetColor(self, lidar, uv, depth, im):  # lidar: 4 * n, uv: n * 4
        h, w = im.shape[:2]

        mask0 = (uv[:, 0] >= 0) & (uv[:, 0] < w)
        mask1 = (uv[:, 1] >= 0) & (uv[:, 1] < h)
        mask2 = depth[:, 0] > 0
        mask = mask0 & mask1 & mask2

        uv = uv[mask]
        lidar = lidar.transpose(1, 0)[mask]

        x = uv[:, :1]
        y = uv[:, 1:]
        xbl = torch.floor(x)
        ybl = torch.floor(y)
        bl = torch.cat([xbl, ybl], dim=1).float()

        xbr = torch.clamp(xbl + 1, min=0, max=w-1)
        ybr = torch.clamp(ybl, min=0, max=h-1)
        br = torch.cat([xbr, ybr], dim=1).float()

        xtl = torch.clamp(xbl, min=0, max=w-1)
        ytl = torch.clamp(ybl + 1, min=0, max=h-1)
        tl = torch.cat([xtl, ytl], dim=1).float()

        xtr = torch.clamp(xbl + 1, min=0, max=w-1)
        ytr = torch.clamp(ybl + 1, min=0, max=h-1)
        tr = torch.cat([xtr, ytr], dim=1).float()

        bl_weight = (tr[:, 1:] - uv[:, 1:]) * (tr[:, :1] - uv[:, :1]).repeat([1, 3])
        br_weight = (uv[:, 1:] - tl[:, 1:]) * (tl[:, :1] - uv[:, :1]).repeat([1, 3])
        tl_weight = (br[:, :1] - uv[:, :1]) * (uv[:, 1:] - br[:, 1:]).repeat([1, 3])
        tr_weight = (uv[:, :1] - bl[:, :1]) * (uv[:, 1:] - bl[:, 1:]).repeat([1, 3])

        color = im[bl[:, 1].long(), bl[:, 0].long(), :] * bl_weight + im[br[:, 1].long(), br[:, 0].long(), :] * br_weight +\
                im[tl[:, 1].long(), tl[:, 0].long(), :] * tl_weight + im[tr[:, 1].long(), tr[:, 0].long(), :] * tr_weight
        lidar = torch.cat([lidar, color], dim=1)

        return lidar