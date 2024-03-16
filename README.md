# 红外彩色化数据集

一种制备真实可信的红外彩色图像数据集的方法，为训练神经网络提供数据样本。

2022年数据效果：

![alt](https://github.com/wjjjjyourFA/NUDT_ALV_OPEN/blob/master/images/20221119.jpg)

2022年数据效果：

![alt](https://github.com/wjjjjyourFA/NUDT_ALV_OPEN/blob/master/images/20231016.jpg)

![alt](https://github.com/wjjjjyourFA/NUDT_ALV_OPEN/blob/master/images/20231017.jpg)



## 一、运行环境说明

本例程测试环境：硬件为联想Y9000P，操作系统为Ubuntu20.04，ROS版本为neotiic。

第三方库文件： octomap-1.9.0 、pcl-1.10 、 vtk-7.1 、 python3.8



## 二、编译和运行

1. 使用QtCreator编译 OctomapTakeOut

2. 使用PyCharm运行 InfraColor\scripts\color-image-ntd.py



## 三、工作流程

1. 基础设备准备：相机、激光雷达和惯导的联合标定，相机和激光雷达的时空同步，激光雷达基于惯导的动态补偿

2. 基础数据采集：在测试地点，白天场景采集惯导、激光雷达、可见光图像数据

​                                                     夜晚场景采集惯导、激光雷达、红外图像数据

（城市区域数据采集难度低于越野区域）

3. 高精地图构建：通过两次采集数据，构建该测试地点的高精地图，获取对应图像帧和雷达帧的精准全局位姿数据

4. 稠密点云选取：依据获得的全局位姿数据，从高精地图中求取该帧的激光雷达点云数据，并转换到激光雷达坐标系

5. 点云和图像数据匹配：通过时间戳、欧式距离与偏航角计算，分别获取成对的白天彩色图像与点云数据，红外图像数据与点云数据

6. 红外图像彩色化：将红外图像中的灰色像素，转换为可见光彩色像素



## Downloads

DataSets 下载链接： Syncing 



## TO DO：

1. 当前只实现了单帧的像素恢复，可通过多次匹配，优化恢复效果
2. 通过不同的采样优化方法，例如BGK，分段提升点云恢复效果



## 注意：提供示例数据，不提供高精地图构建方法

2022-10-24-map  白天彩色图像、点云和惯导数据

2022-10-25-map  夜晚红外图像、点云和惯导数据



## 部分参考文献

1. Research on Colorful Visualization and Object Detection in Low Illumination

2. In Defense of Classical Image Processing: Fast Depth Completion on the CPU. [cs.CV 31 Jan 2018 ]  [(IP-Basic)](https://github.com/kujason/ip_basic) 

3. Improving 3D Object Detection for Pedestrians with Virtual Multi-View Synthesis Orientation Estimation. [cs.CV 15 Jul 2019 ]  [(3D Scene Visualizer)](https://github.com/kujason/scene_vis) 



## Contact

If you have any questions, feel free to [open an issue]() or contact [Jie Wang]() for 1271706355@qq.com.