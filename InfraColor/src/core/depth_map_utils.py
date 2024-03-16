import cv2
import numpy as np
import png

from datasets.kitti.obj import calib_utils


# used
def read_depth_map(depth_map_path):

    depth_image = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)

    depth_map = depth_image / 256.0

    # Discard depths less than 10cm from the camera
    depth_map[depth_map < 0.1] = 0.0

    return depth_map.astype(np.float32)


# used
def save_depth_map(save_path, depth_map,
                   version='cv2', png_compression=3):
    """Saves depth map to disk as uint16 png

    Args:
        save_path: path to save depth map
        depth_map: depth map numpy array [h w]
        version: 'cv2' or 'pypng'
        png_compression: Only when version is 'cv2', sets png compression level.
            A lower value is faster with larger output,
            a higher value is slower with smaller output.
    """

    # Convert depth map to a uint16 png
    depth_image = (depth_map * 256.0).astype(np.uint16)

    if version == 'cv2':
        ret = cv2.imwrite(save_path, depth_image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])

        if not ret:
            raise RuntimeError('Could not save depth map')

    elif version == 'pypng':
        with open(save_path, 'wb') as f:
            depth_image = (depth_map * 256.0).astype(np.uint16)
            writer = png.Writer(width=depth_image.shape[1],
                                height=depth_image.shape[0],
                                bitdepth=16,
                                greyscale=True)
            writer.write(f, depth_image)

    else:
        raise ValueError('Invalid version', version)


# used
def get_depth_point_cloud(depth_map, cam_p, min_v=0, flatten=True, in_cam0_frame=False):
    """Calculates the point cloud from a depth map given the camera parameters

    Args:
        depth_map: depth map
        cam_p: camera p matrix
        min_v: amount to crop off the top
        flatten: flatten point cloud to (3, N), otherwise return the point cloud
            in xyz_map (3, H, W) format. (H, W, 3) points can be retrieved using
            xyz_map.transpose(1, 2, 0)
        in_cam0_frame: (optional) If True, shifts the point cloud into cam_0 frame.
            If False, returns the point cloud in the provided camera frame

    Returns:
        point_cloud: (3, N) point cloud
    """

    depth_map_shape = depth_map.shape[0:2]

    # valid_mask = (depth_map[2] != 0)
    # depth_map = depth_map[:, valid_mask]

    if min_v > 0:
        # Crop top part
        depth_map[0:min_v] = 0.0

    # numpy.meshgrid() 成网格点坐标矩阵
    # np.linspace 创建等差数列
    """ .numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
    参数含义: 
    start:返回样本数据开始点
    stop:返回样本数据结束点
    num:生成的样本数据量，默认为50
    endpoint：True则包含stop；False则不包含stop
    retstep：If True, return (samples, step), where step is the spacing between samples.(即如果为True则结果会给出数据间隔)
    dtype：输出数组类型
    axis：0(默认)或-1
    """
    xx, yy = np.meshgrid(
        np.linspace(0, depth_map_shape[1] - 1, depth_map_shape[1]),
        np.linspace(0, depth_map_shape[0] - 1, depth_map_shape[0]))

    # Calibration centre x, centre y, focal length
    centre_u = cam_p[0, 2]
    centre_v = cam_p[1, 2]
    focal_length_x = cam_p[0, 0]
    focal_length_y = cam_p[1, 1]

    i = xx - centre_u
    j = yy - centre_v

    # Similar triangles ratio (x/i = d/f)
    ratio_x = depth_map / focal_length_x
    ratio_y = depth_map / focal_length_y

    x = i * ratio_x
    y = j * ratio_y
    z = depth_map

    # 针孔模型逆变换
    if in_cam0_frame:
        # Return the points in cam_0 frame
        # Get x offset (b_cam) from calibration: cam_p[0, 3] = (-f_x * b_cam)
        x_offset = -cam_p[0, 3] / focal_length_x

        valid_pixel_mask = depth_map > 0
        x[valid_pixel_mask] += x_offset

        """ add by wj """
        y_offset = -cam_p[1, 3] / focal_length_y

        valid_pixel_mask = depth_map > 0
        y[valid_pixel_mask] += y_offset

    # Return the points in the provided camera frame
    point_cloud_map = np.asarray([x, y, z])

    if flatten:
        point_cloud = np.reshape(point_cloud_map, (3, -1))  # 3xN 包含有大量零点
        # """" test 不在这里做，影响后续颜色去除 """
        # valid_mask = (point_cloud[2] != 0)  # 零点去除
        # point_cloud = point_cloud[:, valid_mask]
        return point_cloud.astype(np.float32)
    else:
        return point_cloud_map.astype(np.float32)


# used
def project_depths(point_cloud, cam_p, image_shape, nimg, max_depth=100.0):
# def project_depths(point_cloud, cam_p, image_shape, nimg, max_depth=100.0):
    """Projects a point cloud into image space and saves depths per pixel.

    Args:
        point_cloud: (3, N) Point cloud in cam0
        cam_p: camera projection matrix
        image_shape: image shape [h, w]
        max_depth: optional, max depth for inversion

    Returns:
        projected_depths: projected depth map
    """

    # Only keep points in front of the camera
    # # 切换到 米 为单位
    # Mat 0-255
    all_points = point_cloud.T/1000.  # Nx3
    # # 不切换到 米 为单位
    # # all_points = point_cloud.T  # Nx3 1048195


    # Save the depth corresponding to each point
    """ Projects a 3D point cloud to 2D points 带状图 """
    points_in_img = calib_utils.project_pc_to_image(all_points.T, cam_p)  # 2xN
    points_in_img_int = np.int32(np.round(points_in_img))  # 整数化 2xN 1048195
    # np.round(): 对给定的数组进行四舍五入
    # np.floor(): 对数组元素下取整
    # np.ceil(): 对数组元素上取整

    # Remove points outside image
    valid_indices = \
        (points_in_img_int[0] >= 0) & (points_in_img_int[0] < image_shape[1]) & \
        (points_in_img_int[1] >= 0) & (points_in_img_int[1] < image_shape[0])
    """ 图像窗口 """
    all_points = all_points[valid_indices]  # 439186 Nx3
    points_in_img_int = points_in_img_int[:, valid_indices]  # 439186 2xN
    # print(all_points)

    # Invert depths
    # 对于相机坐标系，右下前，Z轴就是深度
    all_points[:, 2] = max_depth - all_points[:, 2]  # 439186 Nx3

    # Only save valid pixels, keep closer points when overlapping
    # 整数化 可能导致点重复 过滤之
    projected_depths = np.zeros(image_shape)  # 0 -> black
    # cv2.imshow("zero_map", projected_depths)
    # cv2.waitKey()

    valid_indices = [points_in_img_int[1], points_in_img_int[0]]  # (u, v)
    
    """ traversal for nearest pixel """
    for idx in range(points_in_img_int.shape[1]):
        projected_depths[points_in_img_int[1, idx], points_in_img_int[0, idx]] = \
            max(projected_depths[points_in_img_int[1, idx], points_in_img_int[0, idx]],
                all_points[idx, 2])

    projected_depths[tuple(valid_indices)] = \
        max_depth - projected_depths[tuple(valid_indices)]

    """ Dynamic show proj in depth """

    return projected_depths.astype(np.float32)
