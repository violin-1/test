#pragma once

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>         
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vtkPLYReader.h>
#include <vtkOBJReader.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>


#include <Eigen/Dense>
#include <Eigen/Core>

#include <vector>
#include <algorithm>
#include <math.h>
#include<vector>
#include<list>
#include<string>
#include<array>
#include<map>
#include<set>

#include <mutex>
#include <thread>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp> 


#define DEBUG 0
using namespace Eigen;
void abstract_cube(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr res_cloud, MatrixXd args);
void abstract_cube(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr res_cloud, MatrixXd args);
void transfrom_Rotate(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed, MatrixXd args);
void readPCLXYZfile(char* filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
void readPCLXYZRGBfile(char* filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
void readPCLXYZCfile(char* filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
void pcl_render_ball(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, MatrixXd& result, MatrixXd args);
void readSTLfile(char* filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
void readSTLfile(char* filename, vtkSmartPointer<vtkPolyData>& polydata);
struct PointInfo {
    int x, y, z;
    float r, g, b;
};
inline double uniform_deviate(int seed);
inline void randomPointTriangle(float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3, Eigen::Vector4f& p);
inline void randPSurface(vtkPolyData* polydata, std::vector<double>* cumulativeAreas, double totalArea, Eigen::Vector4f& p);
void uniform_sampling(vtkSmartPointer<vtkPolyData> polydata, size_t n_samples, pcl::PointCloud<pcl::PointXYZ>& cloud_out);
//void transfrom_sphere(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, MatrixXd & result, MatrixXd args);
//void transfrom_sphere(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, MatrixXd & result, MatrixXd args);

