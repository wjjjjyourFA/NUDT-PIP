#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <random>
#include <iomanip>
#include <cmath>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/register_point_struct.h>
#include "pcl/visualization/pcl_visualizer.h"
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#include <pcl/common/io.h>
#include <pcl/impl/point_types.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "colors.hpp"
#include "point.h"

using namespace  Eigen;
using namespace std;
using namespace octomap;
using namespace pcl;

#define PI 3.14159265358979

#define draw_whole_map 0  //plot whole map or search around current_position
#define height_bias 0   //meters, only for visualization
#define downsample 1    //downsample rate
#define radiusSearch_thresh 100   //meters
#define extract_points_mode 1  //only to extract points from map, not to show points.

/* control */
#define Gpose_mode 2
/*
 * Gpose = 1;
 * Calculated_pose = 2
*/
#define point_num_choose 10  // 20

struct pointXYZ
{
    float x;
    float y;
    float z;
};

double tic() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return ((double)t.tv_sec + ((double)t.tv_usec)/1000000.);
}

void toc(double t) {
    double s = tic();
    std::cout<<std::max(0., (s-t)*1000)<<" ms"<<std::endl;
}

// 使用字符分割
void Stringsplit(const string& str, const char split, vector<string>& res)
{
    if (str == "")		return;
    //在字符串末尾也加入分隔符，方便截取最后一段
    string strs = str + split;
    size_t pos = strs.find(split);

    // 若找不到内容则字符串搜索函数返回 npos
    while (pos != strs.npos)
    {
        string temp = strs.substr(0, pos);
        res.push_back(temp);
        //去掉已分割的字符串,在剩下的字符串中进行分割
        strs = strs.substr(pos + 1, strs.size());
        pos = strs.find(split);
    }
}
// 使用字符串分割
void Stringsplit(const string& str, const string& splits, vector<string>& res)
{
    if (str == "")		return;
    //在字符串末尾也加入分隔符，方便截取最后一段
    string strs = str + splits;
    size_t pos = strs.find(splits);
    int step = splits.size();

    // 若找不到内容则字符串搜索函数返回 npos
    while (pos != strs.npos)
    {
        string temp = strs.substr(0, pos);
        res.push_back(temp);
        //去掉已分割的字符串,在剩下的字符串中进行分割
        strs = strs.substr(pos + step, strs.size());
        pos = strs.find(splits);
    }
}

static Eigen::Matrix3d ypr2Rzyx(double y, double p, double r)
{
    Eigen::Matrix3d Rz;
    Rz << cos(y), -sin(y), 0,
          sin(y),  cos(y), 0,
               0,       0, 1;

    Eigen::Matrix3d Ry;
    Ry << cos(p), 0., sin(p),
              0., 1.,     0.,
         -sin(p), 0., cos(p);

    Eigen::Matrix3d Rx;
    Rx << 1.,      0.,     0.,
          0., cos(r), -sin(r),
          0., sin(r),  cos(r);

    return Rz * Ry * Rx;
}

static Eigen::Matrix3d ypr2Rzxy(double y, double p, double r)
{
    Eigen::Matrix3d Rz;
    Rz << cos(y), -sin(y), 0,
          sin(y),  cos(y), 0,
               0,       0, 1;

    Eigen::Matrix3d Rx;
    Rx << 1.,      0.,     0.,
          0., cos(p), -sin(p),
          0., sin(p),  cos(p);

    Eigen::Matrix3d Ry;
    Ry << cos(r), 0., sin(r),
              0., 1.,     0.,
         -sin(r), 0., cos(r);

    return Rz * Rx * Ry;
}


// ############# from lbk #############
struct double3D{
    double x;
    double y;
    double z;
};

static Eigen::Matrix3d ypr2Rzxy_lbk(double y, double p, double r)
{
    double3D ypr;
    ypr.x =  y;
    ypr.y =  p;
    ypr.z =  r;

    Eigen::Matrix3d Rx,Ry,Rz,R_w_l;
    double crz = cos(ypr.x);
    double srz = sin(ypr.x);
    double crx = cos(ypr.y);
    double srx = sin(ypr.y);
    double cry = cos(ypr.z);
    double sry = sin(ypr.z);
    Ry<<cry, 0, sry,
        0 ,1, 0,
        -sry, 0, cry;
    Rx<<1, 0, 0,
        0, crx, -srx,
        0, srx, crx;
    Rz<<crz, -srz, 0,
        srz, crz, 0,
        0 , 0, 1;
    R_w_l = Rz*Rx*Ry; // R_azimuth * R_pitch * R_roll

    return R_w_l;
}
// ############# from lbk #############

int main()
{
    //
    // 20240301 input map.ot right front up

    std::cout<<"====> START. "<<std::endl;

    // 地图的建立是带有 gx gy h 的，然后建完整个地图后，人为的减去某个 center 。
    // target point highway ie. 1968172460 311365323 8281 ; city ie. 1968442816.718949 311273154.308825 4587.254528 ;
    Eigen::Vector3f current_position;    //from mapper.txt
//    current_position(0) = 1968188168.833001;    //gaussX
//    current_position(1) = 311332565.691782;     //gaussY
//    current_position(2) = 7588.035661;          //height
    // old
//    string map_file_path   = "./../../res_20201003/hexi_inside_map/all_map.ot";
//    char offset_path[100]  = "./../../res_20201003/hexi_inside_map/mapper_offset.txt";
//    string globalpose_path = "./../../res_20201003/input_gps/2020-07-17-13-23-51.txt";
    // 22222
//    string map_file_path   = "./../../res_20220524/school_map/point_cloud_map.ot";
//    char offset_path[100]  = "./../../res_20220524/school_map/map_center.txt";
//    string globalpose_path = "./../../res_20220524/input_gps/GPose.txt";
//    char m_SavePath[300] = "./../../res_20220524/out";
    // 33333
    string map_file_path   = "/home/fkx/fiction/lbk/2022-10-24/octomap/point_cloud_map.ot";
    char offset_path[100]  = "/home/fkx/fiction/lbk/2022-10-24/octomap/map_center.txt";
#if Gpose_mode == 1
    string globalpose_path = "/media/jojo/AQiDePan/done/city/2022-05-17/2022-05-17-15-39-32/GPose_manual.txt";
#elif Gpose_mode == 2
    string globalpose_path = "/home/fkx/fiction/lbk/2022-10-24/meta_data/mapper.txt";
#elif Gpose_mode == 3
//    string globalpose_path = "/media/jojo/AQiDePan/done/city/2022-05-17/2022-05-17-15-39-32/ImagePose.txt";
    string globalpose_path = "/media/jojo/AQiDePan/done/city/2022-05-17/2022-05-17-15-39-32/ImageKeyFramePose.txt";

#endif
//    char m_SavePath[300] = "/media/jojo/AQiDeDATA/done/2022-10-25-map/out_data";
    char m_SavePath[300] = "/media/fkx/WorkStation/done/lbk/2022-10-24/LidarData_dense_10hz_70m";
//    char m_SavePath[300] = "/media/jojo/AQiDeDATA/done/2022-10-24-map/LidarData_dense_10hz_70m";

    double t = tic();

    // 读取.ot map center ==> Offset1[]
    // old
    /*
    int32_t Offset1[3]; // x y z
    FILE* fp = fopen(offset_path, "r");
    for(int i=0; i<3; i++)
        fscanf(fp, "%d", &Offset1[i]);
    fclose(fp);
    */
    // new
    double Offset1[3]; // x y z
    FILE* fp = fopen(offset_path, "r");
    for(int i=0; i<3; i++)
        fscanf(fp, "%lf", &Offset1[i]);
    fclose(fp);

    // 读取 gps 位姿
    string s;  int point_num = 0;
#if 0
    vector<string> timestamp;
#elif 1
    vector<int64_t> timestamp;
#endif
    vector<vector<double>> coord;

    // 字符串中转，导致double类型数据有偏差
    /*
    ifstream infile;
    infile.open(globalpose_path.data());

    while(getline(infile,s))
    {
        istringstream iss(s);
        vector<string> v;
        string token;
        while(getline(iss, token, ' ')) {
            v.push_back(token);
        }
        vector<double> coord_once;
#if Gpose_mode == 1
        for(int i=4;i<=9;i++)  // GNSS_gaussx GNSS_gaussy GNSS_height GNSS_pitch GNSS_roll GNSS_azimuth 单位-厘米 角度 0.01度 -9000
#elif Gpose_mode == 2 || Gpose_mode == 3
        for(int i=1;i<=6;i++)  // GNSS_gaussx GNSS_gaussy GNSS_height GNSS_pitch GNSS_roll GNSS_azimuth 单位-米 弧度 不用 -9000
#else
        // 11111
        for(int i=1;i<=3;i++)  // located_gaussx located_gaussy located_height 单位-米
//        for(int i=1;i<=6;i++)  // located_gaussx located_gaussy located_height located_pitch located_roll located_azimuth
//        for(int i=7;i<=12;i++)  // GNSS_gaussx GNSS_gaussy GNSS_height GNSS_pitch GNSS_roll GNSS_azimuth
#endif
        {
            double a;
            // 字符串转长浮点数
            sscanf(v[i].data(),"%lf",&a);
            coord_once.push_back(a);
            // located_gaussx located_gaussy located_height
//             cout<<"v["<<i<<"].data(): "<<a<<endl;
        }
        coord.push_back(coord_once);
//        std::cout<<point_num<< std::setprecision(20) <<": "<<int(coord[point_num][0])<<"\t"
//                                                           <<int(coord[point_num][1])<<"\t"
//                                                           <<int(coord[point_num][2])<<"\t"
//                                                           <<timestamp[point_num]<<std::endl;
        string str(v[0]);
        vector<string> strList;
        Stringsplit(str, '.', strList);
        timestamp.push_back(strList[0]);

        point_num ++ ;
    }

    */

// #define MAX_NUMBER_OF_FRAMES 30000
#define MAX_NUMBER_OF_FRAMES 6000

    std::cout<<"====> reading coord. "<<std::endl;

    // 20230401
    /*
    int64_t *image_timestamp_DB = new int64_t [MAX_NUMBER_OF_FRAMES];
    double *image_pose_DB = new double [MAX_NUMBER_OF_FRAMES*6];
    FILE *fp2 = fopen(globalpose_path.c_str(),"r");
    if (fp2 != NULL) {
        while (!feof(fp2)) {
            fscanf(fp, "%ld %lf %lf %lf %lf %lf %lf\n", image_timestamp_DB+point_num,
                   image_pose_DB+point_num*6, image_pose_DB+point_num*6+1, image_pose_DB+point_num*6+2,
                   image_pose_DB+point_num*6+3, image_pose_DB+point_num*6+4, image_pose_DB+point_num*6+5);
            point_num++;
            assert(num_of_frames<MAX_NUMBER_OF_FRAMES);
        }
        fclose(fp2);
    }
    */
    // ############# from lbk #############
    // for 读取 gps 位姿
    int global_iter = 0;  //  ===>  point_num
    int64_t *lidar_timestamp_DB = new int64_t [MAX_NUMBER_OF_FRAMES];  //  ===>  image_pose_DB
    double *slam_pose_DB = new double [MAX_NUMBER_OF_FRAMES*6];
    double *GPS_pose_DB = new double [MAX_NUMBER_OF_FRAMES*6];
    FILE *fp2 = fopen(globalpose_path.c_str(),"r");
    if (fp2 != NULL) {
        while (!feof(fp2)) {
            fscanf(fp2, "%ld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", lidar_timestamp_DB+global_iter,
                   slam_pose_DB+global_iter*6, slam_pose_DB+global_iter*6+1, slam_pose_DB+global_iter*6+2,
                   slam_pose_DB+global_iter*6+3, slam_pose_DB+global_iter*6+4, slam_pose_DB+global_iter*6+5,
                   GPS_pose_DB+global_iter*6, GPS_pose_DB+global_iter*6+1, GPS_pose_DB+global_iter*6+2,
                   GPS_pose_DB+global_iter*6+3, GPS_pose_DB+global_iter*6+4, GPS_pose_DB+global_iter*6+5);

            global_iter++;
            assert(global_iter<MAX_NUMBER_OF_FRAMES);
        }
        fclose(fp2);
    }
    /*
    slam_pose_DB



    GPS_pose_DB


    */
    // ############# from lbk #############

    std::cout<<"====> checking coord. "<<std::endl;
    // int point_sum = point_num;  // gpose 数目 ==> frame num
    int point_sum = global_iter;  // gpose 数目 ==> frame num

    for(int frame_idx=0; frame_idx<point_sum; frame_idx++) {
        vector<double> coord_once;
        for(int i=0; i<6; i++) {
            // coord_once.push_back(image_pose_DB[6*frame_idx+i]);
            coord_once.push_back(slam_pose_DB[6*frame_idx+i]);
        }
        coord.push_back(coord_once);
//        cout<< std::setprecision(20) << coord[frame_idx][0]<<endl;
//        cout<<"================================================"<<endl;
        // timestamp.push_back(image_timestamp_DB[frame_idx]);
        timestamp.push_back(lidar_timestamp_DB[frame_idx]);

    }

    std::cout<<"====> read coord end. "<<std::endl;

    std::cout<<"====> reading map. "<<std::endl;

    // 读取.octomap
    AbstractOcTree* tree_pnt1 = AbstractOcTree::read(map_file_path);
    const ColorOcTree& tree1 = ((const ColorOcTree&) *tree_pnt1);
    // 基本的OcTree类和带有颜色信息的ColorOcTree类

    // 加载 octomap 到 pcd
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputXYZ(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ o;

    point3d temp_point;
    int index1 = 0;

    for(ColorOcTree::tree_iterator it = tree1.begin_tree(16),end=tree1.end_tree(); it!= end; ++it)
    {
        if (it.isLeaf()) {  // voxels for leaf nodes  // whether the current node is a leaf, i.e. has no children or is at max level
            if (tree1.isNodeOccupied(*it))  // occupied voxels
            {
                temp_point = it.getCoordinate();
                // 	return the center coordinate of the current node
                o.x=temp_point(0);
                o.y=temp_point(1);
                o.z=temp_point(2);

                index1++;
                if(index1%downsample == 0) {
                    inputXYZ->points.push_back(o);
                }
            }
        }
    }
    std::cout<<"====> read map end. "<<std::endl;

    // pcl
    inputXYZ->width = 1;
    inputXYZ->height = inputXYZ->size();

    // 全地图
//        pcl::io::savePCDFileASCII("./map3.pcd", *inputXYZ);  exit(0);
//        pcl::io::savePcdBinary();

    int select_point_fram_idx = 0;
    for(auto coord_once :coord)
    {
#if point_num_choose
    // if(select_point_fram_idx%10==0)  // 10-1
    //if(select_point_fram_idx%1==0)  // 1-1
    //{
#endif
        // 转到该地图的差值
        // old
//        current_position(0) = (coord_once[0] - Offset1[0])/100.f;
//        current_position(1) = (coord_once[1] - Offset1[1])/100.f;
//        current_position(2) = (coord_once[2] - Offset1[2])/100.f;
#if Gpose_mode == 1
        // globalpose 以厘米为单位， map 以米为单位
//        current_position(0) = (coord_once[0]/100 - Offset1[0]);
//        current_position(1) = (coord_once[1]/100 - Offset1[1]);
//        current_position(2) = (coord_once[2]/100 - Offset1[2]);
#elif Gpose_mode == 2 || Gpose_mode == 3
        // globalpose 以米为单位
        current_position(0) = (coord_once[0] - Offset1[0]);
        current_position(1) = (coord_once[1] - Offset1[1]);
        current_position(2) = (coord_once[2] - Offset1[2]);
#endif
        // 在地图上，距离(0,0,0)点的位置
        std::cout<<select_point_fram_idx<< std::setprecision(20) <<": "<<int(current_position(0))<<"\t"
                                                                       <<int(current_position(1))<<"\t"
                                                                       <<int(current_position(2))<<std::endl;
//        printf("%f, %f, %f \n", current_position(0), current_position(1), current_position(2));

        // string name split
        /*
        name.append(timestamp[point_num]);
        name.append(back);
        char txtName[300];
        sprintf(txtName, name.data());
        */

        // pcl
//        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>());

        std::vector<Pointxyzi> _p;

        for(const auto& p : inputXYZ->points) {
            Eigen::Vector3f p1(p.x, p.y, p.z);  // 差值坐标
//            cout<<p<<endl;

#if 1
//            float temp = (p1 - current_position).sum();
            // 按圆的半径逐帧取点
            float dist =(p1 - current_position).norm();  // norm返回的是向量的二范数
            // 半径 30 m
            if(dist < 70.0) {
                // 已测试，pcl 不提供 double 类型，直接存补上 offset，会导致点的位置错位。
                /*
                std::cout<< "before " <<p.x<<' '<<p.y<<' '<<p.z<<std::endl;
                double x = p.x*100 + double(Offset1[0]);
                double y = p.y*100 + double(Offset1[1]);
                double z = p.z*100 + double(Offset1[2]);

                std::cout<< std::setprecision(20) << "after " <<op.x<<' '<<op.y <<' '<< op.z<< std::endl;
                std::cout<< std::setprecision(20) << "after " <<x<<' '<<y<<' '<<z<<std::endl;
                */
//                cloud_out->push_back(p);

                // 还原到 GNSS 全局坐标系
                // double 不要用 float 去括号
                Pointxyzi tmp;
                tmp.x= p.x + Offset1[0];
//                std::cout<<"tmp.x "<< tmp.x <<std::endl; // ==> double
                tmp.y= p.y + Offset1[1];
                tmp.z= p.z + Offset1[2];
                tmp.i= 1;
                _p.push_back(tmp);
            }
#elif 0
            // 按正方形，前1000米，左右100米， 20帧取点
            if( (p1(0)-current_position(0)) > 0 && (p1(0)-current_position(0)) < 1000 && \
                abs(p1(1)-current_position(1)) < 100 && \
                abs(p1(2)-current_position(2)) < 10 )
            {
                // 还原到 GNSS 全局坐标系
                Pointxyzi tmp;
                tmp.x= p.x + Offset1[0];
                tmp.y= p.y + Offset1[1];
                tmp.z= p.z + Offset1[2];
                tmp.i= 1;
                _p.push_back(tmp);
            }
#else
            // 按正方形，前后1000米，左右1000米， 20帧取点
            if( abs(p1(0)-current_position(0)) < 1000 && \
                abs(p1(1)-current_position(1)) < 1000 && \
                abs(p1(2)-current_position(2)) < 10 )
            {
                // 还原到 GNSS 全局坐标系
                Pointxyzi tmp;
                tmp.x= p.x + Offset1[0];
                tmp.y= p.y + Offset1[1];
                tmp.z= p.z + Offset1[2];
                tmp.i= 1;
                _p.push_back(tmp);
            }
#endif
        }

#define pcl_show 1
#if pcl_show
        pcl::PointCloud<pcl::PointXYZI>::Ptr point_clouds (new pcl::PointCloud<PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI> point_cloud_tmp;
#endif

//        /*
        Eigen::Matrix<double,Dynamic,4,RowMajor> _ppp;
        int _p_num = _p.size();  // 必须获得所创建数组的维度
        _ppp.resize(_p_num,4);
        for(int i=0;i<_p_num;i++) {  // 单位 厘米
            // 20230401
            // _ppp(i,0) = _p[i].x*100.;
            // _ppp(i,1) = _p[i].y*100.;
            // _ppp(i,2) = _p[i].z*100.;
            // 20240301
            _ppp(i,0) = _p[i].x;
            _ppp(i,1) = _p[i].y;
            _ppp(i,2) = _p[i].z;
            _ppp(i,3) = 1.0;
//            std::cout<<"_ppp(i,0) "<< _ppp(i,0) <<std::endl; // ==> double

#if pcl_show
            // pointXYZI float 无法存下全局坐标，太大了
            // 减去一个基准值
            pcl::PointXYZI point;
            // 20230401
            // point.x = (_ppp(i,0)/100. - Offset1[0]);
            // point.y = (_ppp(i,1)/100. - Offset1[1]);
            // point.z = (_ppp(i,2)/100. - Offset1[2]);
            // 20240301
            point.x = (_ppp(i,0) - Offset1[0]);
            point.y = (_ppp(i,1) - Offset1[1]);
            point.z = (_ppp(i,2) - Offset1[2]);
//            std::cout<<"point.x "<<point.x<<std::endl;
//            std::cout<<"point.y "<<point.y<<std::endl;
            point_clouds->push_back(point);
#endif
        }
        _p.clear();  // 清零
//        */

#if pcl_show
        std::vector<int> idx;
        pcl::removeNaNFromPointCloud(*point_clouds, *point_clouds, idx);

        point_cloud_tmp = *point_clouds;

        int pointsize = point_cloud_tmp.points.size();

        if (false) {
            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_test(new pcl::PointCloud<pcl::PointXYZRGB>);
            copyPointCloud(point_cloud_tmp, *point_cloud_test);
            for (int i = 0; i < pointsize; i++) {
                point_cloud_test->points[i].r = 80;
                point_cloud_test->points[i].g = 80;
                point_cloud_test->points[i].b = 80;
            }

            // writer
            viewer->removeAllPointClouds();
            viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_test);
            while (!viewer->wasStopped()) {
                viewer->spinOnce(10);
            }
        }

//        delete &point_clouds;
//        delete &viewer;
//        delete &point_cloud_test;
#endif

        // 计算旋转矩阵
        double y, p, r;
#if Gpose_mode == 1
        y = (coord_once[5]-9000) / 180/100.0 * M_PI;
        p = coord_once[3] / 180/100.0 * M_PI;
        r = coord_once[4] / 180/100.0 * M_PI;
#elif Gpose_mode == 2 || Gpose_mode == 3
        // y = coord_once[3];
        // p = coord_once[4];
        // r = coord_once[5];
        // 弧度制
        // for lbk
        // y = coord_once[5]*M_PI/180.0-M_PI/2;
        // p = coord_once[4]*M_PI/180.0;
        // r = coord_once[3]*M_PI/180.0;
        y = coord_once[5]-M_PI/2;
        p = coord_once[4];
        r = coord_once[3];
#endif
//        Eigen::Matrix3d R_tmp = ypr2Rzyx(y, p, r);
        // Eigen::Matrix3d R_tmp = ypr2Rzxy(y, p, r);  // for fh
        Eigen::Matrix3d R_tmp = ypr2Rzxy_lbk(y, p, r);  // for lbk
        cout<<timestamp[point_num]<<endl;
        cout<<y<<endl;
        cout<<p<<endl;
        cout<<r<<endl;
        cout<<"**********"<<endl;
        cout<<R_tmp<<endl;
        cout<<"========>"<<endl;

        Eigen::Matrix4d RT_tmp = Eigen::Matrix4d::Zero();
        RT_tmp.block<3, 3>(0, 0) = R_tmp;
#if Gpose_mode == 1
        RT_tmp(0,3) = coord_once[0];
        RT_tmp(1,3) = coord_once[1];
        RT_tmp(2,3) = coord_once[2];
#elif Gpose_mode == 2 || Gpose_mode == 3
        // Cm
        // 20230401
        // RT_tmp(0,3) = coord_once[0]*100;
        // RT_tmp(1,3) = coord_once[1]*100;
        // RT_tmp(2,3) = coord_once[2]*100;
        // Mi
        // for lbk
        // 20240301
        RT_tmp(0,3) = coord_once[0];
        RT_tmp(1,3) = coord_once[1];
        RT_tmp(2,3) = coord_once[2];
#endif
        RT_tmp(3,3) = 1;

        Eigen::Matrix4d RT = RT_tmp.inverse();  // 局部到全局的旋转矩阵的逆
        cout<<RT <<endl;
        cout<<"========>"<<endl;
        // 全局坐标转局部坐标 ==> 车体/惯导坐标
        // 取决于建图的点云坐标是 惯导 or 雷达坐标
        // 这里是 车体坐标系 x 雷达坐标系 y
        Eigen::Matrix<double,4,Dynamic> local_gps_points_all = RT * _ppp.transpose();  // 4xN = 4x4 * 4xN
        // 存 全局坐标 太大了
        // 减去一个基准值
//        Eigen::Matrix<double,4,Dynamic> local_gps_points_all = _ppp.transpose();  // 4xN = 4x4 * 4xN
        _ppp = _ppp.Zero(_p_num,4);
//        cout<<local_gps_points_all<<endl;
        // GNSS 转 Lidar

        // save output
        if(access(m_SavePath,0)==-1)
            if(mkdir(m_SavePath,0744)==-1)
                std::cout<<"The data folder create error!"<<std::endl<<m_SavePath<<std::endl;

        char filename[300];

        // .bin
//        /*
//        char globalpose_pc_time[300];
//        strcpy(globalpose_pc_time, timestamp[point_num].c_str());
#if 0
        sprintf(filename,"%s/%s.bin", m_SavePath, timestamp[point_num].c_str());
#elif 1
        sprintf(filename,"%s/%.13ld.bin", m_SavePath, timestamp[select_point_fram_idx]);
#endif
        FILE *fp_lidar;
        fp_lidar = fopen(filename,"wb");
        if(fp_lidar==NULL) {
            perror("Lidar file create error!");
        }
        int tmp_x, tmp_y, tmp_z, tmp_intensity = 1;

        pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud (new pcl::PointCloud<pcl::PointXYZI>);


        for(int i=0; i<_p_num;i++)
        {
            // 20230401 Cm
            // tmp_x = int (local_gps_points_all(0,i));
            // tmp_y = int (local_gps_points_all(1,i));
            // tmp_z = int (local_gps_points_all(2,i));
            // 20240301 Cm
            tmp_x = int (local_gps_points_all(0,i) *100);
            tmp_y = int (local_gps_points_all(1,i) *100);
            tmp_z = int (local_gps_points_all(2,i) *100);
            // 存 全局坐标 太大了
            // 减去一个基准值
//            tmp_x = int ((local_gps_points_all(0,i)/100. - Offset1[0])*100);
//            tmp_y = int ((local_gps_points_all(1,i)/100. - Offset1[1])*100);
//            tmp_z = int ((local_gps_points_all(2,i)/100. - Offset1[2])*100);

//            if( tmp_x > 0 && tmp_x < 1000*100 && abs(tmp_y) < 100*100 && abs(tmp_z) < 10*100 )
//            if( tmp_x > 0 && tmp_x < 100000 && abs(tmp_y) < 10000 && abs(tmp_z) < 1000 )
//            {
                fwrite(&(tmp_x),sizeof(int),1,fp_lidar);
                fwrite(&(tmp_y),sizeof(int),1,fp_lidar);
                fwrite(&(tmp_z),sizeof(int),1,fp_lidar);
                fwrite(&tmp_intensity,sizeof(int),1,fp_lidar);
//            }
//            else {
//                continue;
//            }


            // for pcd
            PointXYZI point;
            point.x = float(tmp_x/100.);
            point.y = float(tmp_y/100.);
            point.z = float(tmp_z/100.);
            //            point.x = float(tmpdata[0]/1.);
            //            point.y = float(tmpdata[1]/1.);
            //            point.z = float(tmpdata[2]/1.);
            point.intensity = tmp_intensity;
            point_cloud->push_back(point);
        }
        fclose(fp_lidar);


        // char filename_pcd[300];
        // sprintf(filename_pcd,"%s/%.13ld.pcd", m_SavePath, timestamp[select_point_fram_idx]);
        // pcl::io::savePCDFileBinary(filename_pcd, *point_cloud);

#if 0
        std::vector<int> idx;
        pcl::removeNaNFromPointCloud(*point_clouds, *point_clouds, idx);

        point_cloud_tmp = *point_clouds;

        int pointsize = point_cloud_tmp.points.size();

        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_test(new pcl::PointCloud<pcl::PointXYZRGB>);
        copyPointCloud(point_cloud_tmp, *point_cloud_test);
        for (int i = 0; i < pointsize; i++) {
            point_cloud_test->points[i].r = 80;
            point_cloud_test->points[i].g = 80;
            point_cloud_test->points[i].b = 80;
        }

        if (true) {
            // writer
            viewer->removeAllPointClouds();
            viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_test);
            while (!viewer->wasStopped()) {
                viewer->spinOnce(10);
            }
        }

        delete &point_clouds;
        delete &viewer;
        delete &point_cloud_test;
#endif
//        */

        // .pcd
        /*
        //show showClouds;
        pcl::visualization::PCLVisualizer *pp = NULL;
        if (pp == NULL) {
            pp = new pcl::visualization::PCLVisualizer ("map1!");
            pp->addCoordinateSystem(1.0);
            pp->removeAllPointClouds();
            pp->removeAllShapes();
            pp->setBackgroundColor(0.0,0.0,0.0);
        }
        pp->addPointCloud(showClouds);
        pp->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3);
        pp->spin();
        delete pp;
        */
        /*
        pcl::PointCloud<pcl::PointXYZ>::Ptr transform_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        Eigen::Matrix4f RT_ = (RT.block<4, 4>(0, 0)).cast<float>();

        pcl::transformPointCloud(*cloud_out, *transform_cloud, RT_);

        sprintf(filename,"%s/%s.pcd", m_SavePath, timestamp[point_num].c_str());
        pcl::io::savePCDFileASCII(filename, *transform_cloud);   // exit(0);
        */

        // KD tree  # error now
        /*
        pcl::PointXYZ pt;

        FILE *fp2;
        fp2 = fopen(txtName,"wb");
        if(!draw_whole_map)
        {
            pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr map_kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
            map_kd_tree->setInputCloud(inputXYZ);

            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            pointSearchInd.clear();
            pointSearchSqDis.clear();

            int flag = 1;
            // 寻找最近邻
            map_kd_tree->radiusSearch(pcl::PointXYZ(current_position(0), current_position(1), current_position(2)), radiusSearch_thresh, pointSearchInd, pointSearchSqDis);
            for(int i=0; i<pointSearchInd.size(); i++)
            {
                pt = inputXYZ->points[pointSearchInd[i]];
                float x = pt.x-current_position(0);
                float y = pt.y-current_position(1);
                float z = pt.z-current_position(2);
                float q = 1;
                fwrite(&x, sizeof(float), 1, fp2);
                fwrite(&y, sizeof(float), 1, fp2);
                fwrite(&z, sizeof(float), 1, fp2);
                fwrite(&q, sizeof(float), 1, fp2);
                if(flag) {
                    cout << x << " " << y << " " << z << " " << q << endl;
                    flag = 0;
                }
//                fprintf(fp2,"%.2f %.2f %.2f\n",pt.x-current_position(0),pt.y-current_position(1),pt.z-current_position(2));
            }
        }
        */

#if point_num_choose
    //} // point_num%200==0
#endif
        select_point_fram_idx++;
        std::cout<<"point:"<<select_point_fram_idx<<"/"<<point_sum<<std::endl;
    }

    // delete &image_pose_DB;
    // delete &image_timestamp_DB;
    delete &slam_pose_DB;
    delete &GPS_pose_DB;
    delete &lidar_timestamp_DB;
    delete &inputXYZ;

}

bool plot_xy(float input_x, float input_y, bool sort_by_x, int piece_num_all, int num_target)
{
    if(sort_by_x)
    {
        float x_start = x_min + (num_target)*(x_max - x_min)/piece_num_all;
        float x_end = x_min + (num_target+1)*(x_max - x_min)/piece_num_all;
        if(input_x>=x_start && input_x<=x_end)
            return true;
        else
            return false;
    }
    else
    {
        float y_start = y_min + (num_target)*(y_max - y_min)/piece_num_all;
        float y_end = y_min + (num_target+1)*(y_max - y_min)/piece_num_all;
        if(input_y>=y_start && input_y<=y_end)
            return true;
        else
            return false;
    }
}
