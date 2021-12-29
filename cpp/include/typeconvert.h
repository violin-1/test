#pragma once
#include <pcl/io/pcd_io.h>                      
#include <pcl/point_types.h>                   
#include <pcl/visualization/cloud_viewer.h>   

#include <Eigen/Dense>
#include <Eigen/Core>

#include <vector>
#include<algorithm>

using namespace Eigen;

void matrixxd2pointcloudXYZ(MatrixXd input, pcl::PointCloud<pcl::PointXYZ>::Ptr output);
void matrixxd2pointcloudXYZRGB(MatrixXd input, pcl::PointCloud<pcl::PointXYZRGB>::Ptr output);
void pointcloudXYZ2matrixxd(pcl::PointCloud<pcl::PointXYZ>::Ptr input, MatrixXd & output);
void pointcloudXYZRGB2matrixxd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputs, MatrixXd& output, int outputc);
void pointcloudXYZRGB2pointcloudXYZ(pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputs, pcl::PointCloud<pcl::PointXYZ>::Ptr outputs);
float convertColor(float colorIn);
float convertColor(double colorIn);
void convert2rgb(float colorIn, float* rgb);
double rgbconvert2color(float* rgb);
float rgbconvert2color_(float* rgb);
