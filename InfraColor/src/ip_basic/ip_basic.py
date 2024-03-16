"""https://github.com/kujason/ip_basic"""

import collections

import cv2
import numpy as np

FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_9 = np.ones((19, 19), np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)
CROSS_KERNEL_3_half = np.asarray(
    [
        [0, 1/2, 0],
        [1/2, 1, 1/2],
        [0, 1/2, 0],
    ], dtype=np.uint8)
# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)
CROSS_KERNEL_5_half = np.asarray(
    [
        [0, 0, 1/2, 0, 0],
        [0, 0, 1/2, 0, 0],
        [1/2, 1/2, 1, 1/2, 1/2],
        [0, 0, 1/2, 0, 0],
        [0, 0, 1/2, 0, 0],
    ], dtype=np.uint8)
# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)
CROSS_KERNEL_7_half = np.asarray(
    [
        [0, 0, 0, 1/2, 0, 0, 0],
        [0, 0, 0, 1/2, 0, 0, 0],
        [0, 0, 0, 1/2, 0, 0, 0],
        [1/2, 1/2, 1/2, 1, 1/2, 1/2, 1/2],
        [0, 0, 0, 1/2, 0, 0, 0],
        [0, 0, 0, 1/2, 0, 0, 0],
        [0, 0, 0, 1/2, 0, 0, 0],
    ], dtype=np.uint8)


def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    start_time = time.time()
    depth_map = cv2.medianBlur(depth_map, 5)
    print('\n Median {:0.03f}'.format((time.time() - start_time) * 1000))

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':

        start_time = time.time()

        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 0.5, 2.0)

        print('Bilateral {:0.03f}'.format((time.time() - start_time) * 1000))

    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def fill_in_multiscale(depth_map, max_depth=100.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
					   # dilation_kernel_far=CROSS_KERNEL_3_half,
                       # dilation_kernel_med=CROSS_KERNEL_5_half,
                       # dilation_kernel_near=CROSS_KERNEL_7_half,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion 按深度值大小划分  mask
    valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
    valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
    valid_pixels_far = (depths_in > 30.0)

    # Invert (and offset) 深度值的反转
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.1)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]
    # print(s1_inverted_depths[valid_pixels])
    # exit(0)

    """ .cv2.dilate(src, kernel, iteration)
    参数说明: src表示输入的图片， kernel表示方框的大小， iteration表示迭代的次数
    膨胀操作原理：存在一个kernel，在图像上进行从左到右，从上到下的平移，如果方框中存在白色，那么这个方框内所有的颜色都是白色
    """
    # Multi-scale dilation 膨胀 参考论文 给 实点周围的 空点 赋予深度值
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)
    # # 矩阵大小不变
    # print(np.multiply(s1_inverted_depths, valid_pixels_far))
    # # 矩阵大小改变
    # print(s1_inverted_depths[valid_pixels_far])
    # exit(0)
    """         
    a = np.array([-5, 0, 10, 15, 1])
    b = (a[:] > 0)
    print(b)
    print(np.multiply(a, b))
    exit(0)
    """

    # Find valid pixels for each binned dilation  mask
    valid_pixels_near = (dilated_near > 0.1)
    valid_pixels_med = (dilated_med > 0.1)
    valid_pixels_far = (dilated_far > 0.1)
    #
    # Combine dilated versions, starting farthest to nearest 整合 拼成完整图像
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    """ .cv2.morphologyEx(src, op, kernel) 进行各类形态学的变化 作用: 滤波
    参数说明: src传入的图片, op进行变化的方式, kernel表示方框的大小
    op = cv2.MORPH_CLOSE 进行闭运算, 指的是先进行膨胀操作, 再进行腐蚀操作
    """
    # Small hole closure 排除小黑洞
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers 中值模糊去除异常值 滤波 #?
    s4_blurred_depths = np.copy(s3_closed_depths)  # 深拷贝
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.1)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]  # [1024, 1280]  [height, width]

    # Calculate a top mask 对 天空 空点 的 mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool)  # [u, v]
    top_mask_scale2 = np.ones(depths_in.shape, dtype=np.bool)  # [u, v]

    for pixel_col_idx in range(s4_blurred_depths.shape[1]):  # traversal width
        pixel_col = s4_blurred_depths[:, pixel_col_idx]  # traversal height
        top_pixel_row = np.argmax(pixel_col > 0.1)  # 返回第一个大于0.1的数
        top_mask[0:top_pixel_row, pixel_col_idx] = False
        top_mask_scale2[0:int(top_pixel_row * 0.8), pixel_col_idx] = False
        # print(pixel_col)
        # print(top_pixel_row)
        # print(top_mask)
        # tmp = pixel_col > 0.1
        # print(tmp)
        # exit(0)
    left_mask = np.ones(depths_in.shape, dtype=np.bool)  # [u, v]
    for pixel_col_idx in range(s4_blurred_depths.shape[0]):  # traversal width
        pixel_col = s4_blurred_depths[pixel_col_idx, :]  # traversal height
        left_pixel_col = np.argmax(pixel_col > 0.1)  # 返回第一个大于0.1的数
        left_mask[pixel_col_idx, 0:left_pixel_col] = False

    """ wait for right and bottom """
    # right_mask = np.ones(depths_in.shape, dtype=np.bool)  # [u, v]
    # for pixel_col_idx in range(s4_blurred_depths.shape[0]-1, 0, -1):  # traversal width
    #     pixel_col = s4_blurred_depths[pixel_col_idx, :]  # traversal height
    #     right_pixel_col = np.argmax(pixel_col > 0.1)  # 返回第一个大于0.1的数
    #     right_mask[pixel_col_idx, 0:left_pixel_col] = False

    """ only 2/3 up """
    brunch_mask = np.ones(depths_in.shape, dtype=np.bool)
    brunch_up = int(s4_blurred_depths.shape[0] * 2/3)
    # for pixel_col_idx in range(s4_blurred_depths.shape[1]):  # traversal width
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
    valid_pixels = (s4_blurred_depths > 0.1)
    empty_pixels = ~valid_pixels & _mask_
    # print(valid_pixels)
    # print(~valid_pixels)
    # exit(0)

    """ show emptyImage """
    # s4_blurred_depths[empty_pixels] = \
    #     (max_depth - s4_blurred_depths[empty_pixels]) / 255
    # cv2.imshow("depth_map", s4_blurred_depths)
    # cv2.waitKey()

    # Hole fill 膨胀 给 空洞 赋予深度值
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool)
    # 一样对顶部进行 mask
    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]
    # print(top_row_pixels)
    # print(top_pixel_values)
    # exit(0)
    # extrapolate = False 不对天空进行补全
    # 对 直到顶部第一个数值 取零
    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False
    #
    # Fill large holes with masked dilations  对 大的空洞 重复膨胀 赋予深度值
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur 中值滤波
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)  # 0?
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':  # 高斯双边滤波
        """ .cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
        参数说明: src：输入图像, d：过滤时周围每个像素领域的直径
        sigmaColor：Sigma_color较大，则在邻域中的像素值相差较大的像素点也会用来平均。
        sigmaSpace：Sigma_space较大，则虽然离得较远，但是，只要值相近，就会互相影响。
        """
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)  还原回真实的深度值
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    # s8_inverted_depths[~top_mask_scale2] = max_depth * 100

    # s8_inverted_depths[valid_pixels] = \
    #     (max_depth - s8_inverted_depths[valid_pixels]) / 255
    # cv2.imshow("depth_map", s8_inverted_depths)
    # cv2.waitKey()

    depths_out = s8_inverted_depths
    # depths_out[~_mask_] = max_depth * 2

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out, process_dict
