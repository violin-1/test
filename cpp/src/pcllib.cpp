#include "pcllib.h"
#include <typeconvert.h>
void abstract_cube(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr res_cloud, MatrixXd args){
    //pcl::PointCloud<pcl::PointXYZ>::Ptr res_cloud(new pcl::PointCloud<pcl::PointXYZ>);
 // result= Matrix<double,Dynamic,Dynamic>();
    if(DEBUG){
        std::cout << "width:" << cloud->points.size() << " heigth :" << cloud->height << std::endl;
    }
    //int ans_num = 0;
    //cout << args<<endl;
    double xrange_l = args(0,0),xrange_r = args(1,0);
    double yrange_l = args(2,0),yrange_r = args(3,0);
    double zrange_l = args(4,0),zrange_r = args(5,0);
    for(int i = 0 ; i < cloud->points.size() ; i ++){
        if(
            (cloud->points[i].x > xrange_l) && (cloud->points[i].x < xrange_r) &&
            (cloud->points[i].y > yrange_l) && (cloud->points[i].y < yrange_r) &&
            (cloud->points[i].z > zrange_l) && (cloud->points[i].z < zrange_r)){
            res_cloud->points.push_back(cloud->points[i]);
            //ans_num++;
        }
    }
    //pointcloudXYZ2matrixxd(res_cloud, result);
  //result.resize(ans_num,3);
  //int j = 0;
  //for(int i = 0 ; i < cloud->points.size() ; i ++){
  //  if(
  //    (cloud->points[i].x > xrange_l) && (cloud->points[i].x < xrange_r) &&
  //    (cloud->points[i].y > yrange_l) && (cloud->points[i].y < yrange_r) &&
  //    (cloud->points[i].z > zrange_l) && (cloud->points[i].z < zrange_r)){
  //      // mat.row(i) = VectorXd::Map(&data[i][0],data[i].size());
  //      result(j,0) = cloud->points[i].x;
  //      result(j,1) = cloud->points[i].y;
  //      result(j,2) = cloud->points[i].z;
  //      
  //      if((DEBUG) && (j < 10)){
  //         std::cout << "i :---------" << i <<"---------" <<std::endl;
  //        std::cout << "ori"<< i << ": ( " << cloud->points[i].x << "  ,  " << cloud->points[i].y << "  ,  " << cloud->points[i].z << ")" << std::endl;
  //        std::cout << "target"<<j<<":  " << result.row(j) << std::endl;
  //      }
  //      
  //      j++;
  //  }
  //}
  //if(DEBUG){
  //  std::cout << "result:  rows:" << result.rows() << " cols :" << result.cols() << std::endl;
  //  std::cout << "ans_num: " << ans_num<<std::endl;
  //}
}
void abstract_cube(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr res_cloud, MatrixXd args) {
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr res_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    if (DEBUG) {
        std::cout << "width:" << cloud->points.size() << " heigth :" << cloud->height << std::endl;
    }
    //int ans_num = 0;
    // cout << args << endl;
    double xrange_l = args(0, 0), xrange_r = args(1, 0);
    double yrange_l = args(2, 0), yrange_r = args(3, 0);
    double zrange_l = args(4, 0), zrange_r = args(5, 0);
    for (int i = 0; i < cloud->points.size(); i++) {
        if (
            (cloud->points[i].x > xrange_l) && (cloud->points[i].x < xrange_r) &&
            (cloud->points[i].y > yrange_l) && (cloud->points[i].y < yrange_r) &&
            (cloud->points[i].z > zrange_l) && (cloud->points[i].z < zrange_r)) {
            //ans_num++;
            res_cloud->points.push_back(cloud->points[i]);
        }
    }
    //pointcloudXYZRGB2matrixxd(res_cloud, result);
    //result.resize(ans_num, 6);
    //int j = 0;
    //for (int i = 0; i < cloud->points.size(); i++) {
    //    if (
    //        (cloud->points[i].x > xrange_l) && (cloud->points[i].x < xrange_r) &&
    //        (cloud->points[i].y > yrange_l) && (cloud->points[i].y < yrange_r) &&
    //        (cloud->points[i].z > zrange_l) && (cloud->points[i].z < zrange_r)) {
    //        // mat.row(i) = VectorXd::Map(&data[i][0],data[i].size());
    //        result(j, 0) = cloud->points[i].x;
    //        result(j, 1) = cloud->points[i].y;
    //        result(j, 2) = cloud->points[i].z;
    //        result(j, 3) = cloud->points[i].r;
    //        result(j, 4) = cloud->points[i].g;
    //        result(j, 5) = cloud->points[i].b;

    //        j++;
    //    }
    //}
}
void transfrom_Rotate(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed, MatrixXd args){

  Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();//定义平移矩阵，并初始化为单位阵
  //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
  if(args.rows() >=5 && args(5,0)){
    Eigen::Vector4f cloudCentroid;
    pcl::compute3DCentroid(*cloud, cloudCentroid);//计算点云质心
    translation(0, 3) = -cloudCentroid[0];
    translation(1, 3) = -cloudCentroid[1];
    translation(2, 3) = -cloudCentroid[2];
  }
  double x,y,z,theta;
//   std::cout << args.cols() << std::endl;
  if (args.cols() == 1) {
      x = args(0, 0);
      y = args(1, 0);
      z = args(2, 0);
      theta = args(3, 0) * M_PI / 180;
      translation(0, 0) = cos(theta) + (1 - cos(theta)) * x * x;
      translation(0, 1) = (1 - cos(theta)) * x * y - sin(theta) * z;
      translation(0, 2) = (1 - cos(theta)) * x * z + sin(theta) * y;
      translation(1, 0) = (1 - cos(theta)) * y * x + sin(theta) * z;
      translation(1, 1) = cos(theta) + (1 - cos(theta)) * y * y;
      translation(1, 2) = (1 - cos(theta)) * y * z - sin(theta) * x;
      translation(2, 0) = (1 - cos(theta)) * z * x - sin(theta) * y;
      translation(2, 1) = (1 - cos(theta)) * z * y + sin(theta) * x;
      translation(2, 2) = cos(theta) + (1 - cos(theta)) * z * z;
  }
  else if (args.rows() == 4 && args.cols() == 3) {
      for (int i = 0; i < args.rows(); i++) {
          for (int j = 0; j < args.cols(); j++) {
              translation(i, j) = args(i + 1, j);
          }
      }
  }else{
    std::cout << "RotateMatrix wrong!" << std::endl;
    return;
  }
  if(DEBUG){
    std::cout << std::endl;
    std::cout << "------------------------------------------------"<<std::endl;
    std::cout << translation << std::endl;
    std::cout << "------------------------------------------------"<<std::endl;
    std::cout << std::endl;
  }
  
  pcl::transformPointCloud(*cloud, *cloud_transformed, translation);
  //pointcloudXYZ2matrixxd(cloud_transformed, result);
  // std::cout << result << std::endl;
  return;
}
void readPCLXYZfile(char* filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    FILE* ifp;
    const int BUFSIZE = 512;
    char buf[BUFSIZE];
    int i;
    if ((ifp = fopen(filename, "r+")) == NULL)   // input file handle
    {
        printf("Cannot open this file!\n");
        exit(1);
    }
    // memory allocation for three arrays
    pcl::PointXYZ tmp;

    i = 0;
    while (fgets(buf, BUFSIZE, ifp) != NULL)  // read info line by line
    {
        sscanf(buf, "%lf%lf%lf", &tmp.x, &tmp.y, &tmp.z);
        cloud->points.push_back(tmp);
        i++;
    }
    //pointcloudXYZ2matrixxd(cloud, outputs);
    fclose(ifp);
}
void readPCLXYZRGBfile(char* filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    FILE* ifp;
    const int BUFSIZE = 512;
    char buf[BUFSIZE];
    
    int i;
    if ((ifp = fopen(filename, "r+")) == NULL)   // input file handle
    {
        printf("Cannot open this file!\n");
        exit(1);
    }
    //fgets(buf, BUFSIZE, ifp);   // read record entry number
    //number = atoi(buf);

    // memory allocation for three arrays
    pcl::PointXYZRGB tmp;

    i = 0;
    while (fgets(buf, BUFSIZE, ifp) != NULL)  // read info line by line
    {
        sscanf(buf, "%f%f%f%f%f%f", &tmp.x, &tmp.y, &tmp.z, &tmp.r, &tmp.g, &tmp.b);
        if ((tmp.x == tmp.y) && (tmp.y == tmp.z) && (tmp.z == 0)) {
            continue;
        }
        cloud->points.push_back(tmp);
        i++;
    }
    
    fclose(ifp);
}
void readPCLXYZCfile(char* filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    FILE* ifp;
    const int BUFSIZE = 512;
    char buf[BUFSIZE];
    
    int i;
    if ((ifp = fopen(filename, "r+")) == NULL)   // input file handle
    {
        printf("Cannot open this file!\n");
        exit(1);
    }
    //fgets(buf, BUFSIZE, ifp);   // read record entry number
    //number = atoi(buf);

    // memory allocation for three arrays
    pcl::PointXYZRGB tmp;

    i = 0;
    while (fgets(buf, BUFSIZE, ifp) != NULL)  // read info line by line
    {
        sscanf(buf, "%f%f%f%f", &tmp.x, &tmp.y, &tmp.z, &tmp.rgb);
        cloud->points.push_back(tmp);
        i++;
    }
    
    fclose(ifp);
}
void pcl_render_ball(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, MatrixXd& result, MatrixXd args) {
    int h, w, n, r, mo;
    h = args(0, 0);
    w = args(1, 0);
    n = args(2, 0);
    r = args(3, 0);
    r = std::max(r, 1);

    MatrixXd depth(h, w);
    for (int i = 0; i < depth.rows(); i++) {
        for (int j = 0; j < depth.cols(); j++) {
            depth(i, j) = -2100000000;
        }
    }
    //vector<int> depth(h * w, -2100000000);
    std::vector<PointInfo> pattern;
    for (int dx = -r; dx <= r; dx++)
        for (int dy = -r; dy <= r; dy++)
            if (dx * dx + dy * dy < r * r) {
                double dz = sqrt(double(r * r - dx * dx - dy * dy));
                PointInfo pinfo;
                pinfo.x = dx;
                pinfo.y = dy;
                pinfo.z = dz;
                pinfo.r = dz / r;
                pinfo.g = dz / r;
                pinfo.b = dz / r;

                pattern.push_back(pinfo);
            }
    double zmin = 0, zmax = 0;
    //for (int i = 0; i < n; i++) {
    //    if (i == 0) {
    //        zmin = xyzs[i * 3 + 2] - r;
    //        zmax = xyzs[i * 3 + 2] + r;
    //    }
    //    else {
    //        zmin = min(zmin, double(xyzs[i * 3 + 2] - r));
    //        zmax = max(zmax, double(xyzs[i * 3 + 2] + r));
    //    }
    //}
    for (int i = 0; i < cloud->points.size(); i++) {
        if (i == 0) {
            zmin = cloud->points[i].z - r;
            zmax = cloud->points[i].z + r;
        }
        else {
            zmin = std::min(zmin, double(cloud->points[i].z - r));
            zmax = std::max(zmax, double(cloud->points[i].z + r));
        }
    }
    result.resize(3*h, w);
    for (int i = 0; i < n; i++) {
        int x = cloud->points[i].x, y = cloud->points[i].y, z = cloud->points[i].z;
        for (int j = 0; j<int(pattern.size()); j++) {
            int x2 = x + pattern[j].x;
            int y2 = y + pattern[j].y;
            int z2 = z + pattern[j].z;
            if (!(x2 < 0 || x2 >= w || y2 < 0 || y2 >= h) && depth(y2, x2) < z2) {
                depth(y2, x2) = z2;
                double intensity = std::min(1.0, (z2 - zmin) / (zmax - zmin) * 0.7 + 0.3);
                if (args(4, 0) == 0) {
                    intensity = 1;
                }
                result(y2,x2) = pattern[j].r * cloud->points[i].r * intensity; 
                y2 += h;
                result(y2, x2) = pattern[j].g * cloud->points[i].g * intensity;
                y2 += h;
                result(y2, x2) = pattern[j].b * cloud->points[i].b * intensity;
                //show[(x2 * w + y2) * 3 + 0] = pattern[j].b * cloud->points[i].r * intensity;
                //show[(x2 * w + y2) * 3 + 1] = pattern[j].g * cloud->points[i].g * intensity;
                //show[(x2 * w + y2) * 3 + 2] = pattern[j].r * cloud->points[i].b * intensity;
            }
        }
    }
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (depth(i, j) == -2100000000) {
                result(i, j) = 0;
                result(i + h, j) = 0;
                result(i + h + h, j) = 0;
            }
        }
    }

}
void readSTLfile(char* filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
    reader->SetFileName(filename);
    reader->Update();
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata = reader->GetOutput();
    polydata->GetNumberOfPoints();
    pcl::io::vtkPolyDataToPointCloud(polydata, *cloud);
}
void readSTLfile(char* filename, vtkSmartPointer<vtkPolyData> & polydata) {
    vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
    reader->SetFileName(filename);
    reader->Update();
    polydata = reader->GetOutput();
    polydata->GetNumberOfPoints();
}
double uniform_deviate(int seed){
    double ran = seed * (1.0 / (RAND_MAX + 1.0));
    return ran;
}
void randomPointTriangle(float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3, Eigen::Vector4f& p){
    float r1 = static_cast<float> (uniform_deviate(rand()));
    float r2 = static_cast<float> (uniform_deviate(rand()));
    float r1sqr = sqrtf(r1);
    float OneMinR1Sqr = (1 - r1sqr);
    float OneMinR2 = (1 - r2);
    a1 *= OneMinR1Sqr;
    a2 *= OneMinR1Sqr;
    a3 *= OneMinR1Sqr;
    b1 *= OneMinR2;
    b2 *= OneMinR2;
    b3 *= OneMinR2;
    c1 = r1sqr * (r2 * c1 + b1) + a1;
    c2 = r1sqr * (r2 * c2 + b2) + a2;
    c3 = r1sqr * (r2 * c3 + b3) + a3;
    p[0] = c1;
    p[1] = c2;
    p[2] = c3;
    p[3] = 0;
}
void randPSurface(vtkPolyData* polydata, std::vector<double>* cumulativeAreas, double totalArea, Eigen::Vector4f& p){
    float r = static_cast<float> (uniform_deviate(rand()) * totalArea);

    std::vector<double>::iterator low = std::lower_bound(cumulativeAreas->begin(), cumulativeAreas->end(), r);
    vtkIdType el = vtkIdType(low - cumulativeAreas->begin());

    double A[3], B[3], C[3];
    vtkIdType npts = 0, *ptIds = NULL;
    polydata->GetCellPoints(el, npts, ptIds);
    polydata->GetPoint(ptIds[0], A);
    polydata->GetPoint(ptIds[1], B);
    polydata->GetPoint(ptIds[2], C);
    randomPointTriangle(float(A[0]), float(A[1]), float(A[2]),
        float(B[0]), float(B[1]), float(B[2]),
        float(C[0]), float(C[1]), float(C[2]), p);
}
void uniform_sampling(vtkSmartPointer<vtkPolyData> polydata, size_t n_samples, pcl::PointCloud<pcl::PointXYZ>& cloud_out){
    polydata->BuildCells();
    vtkSmartPointer<vtkCellArray> cells = polydata->GetPolys();

    double p1[3], p2[3], p3[3], totalArea = 0;
    std::vector<double> cumulativeAreas(cells->GetNumberOfCells(), 0);
    size_t i = 0;
    vtkIdType npts = 0, *ptIds = NULL;
    for (cells->InitTraversal(); cells->GetNextCell(npts, ptIds); i++)
    {
        polydata->GetPoint(ptIds[0], p1);
        polydata->GetPoint(ptIds[1], p2);
        polydata->GetPoint(ptIds[2], p3);
        totalArea += vtkTriangle::TriangleArea(p1, p2, p3);
        cumulativeAreas[i] = totalArea;
    }

    cloud_out.points.resize(n_samples);
    cloud_out.width = static_cast<uint32_t> (n_samples);
    cloud_out.height = 1;

    for (i = 0; i < n_samples; i++)
    {
        Eigen::Vector4f p;
        randPSurface(polydata, &cumulativeAreas, totalArea, p);
        cloud_out.points[i].x = p[0];
        cloud_out.points[i].y = p[1];
        cloud_out.points[i].z = p[2];
    }
}
//void render_pcl(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, MatrixXd& args) {
//    int w = args(0, 0);
//    int h = args(1, 0);
//    cv::Mat panel(h, w, CV_8UC3);
//    for (int k = 0; k < cloud->points.size(); k++) {
//
//    }
//    for (int i = 0; i < panel.rows; i++)
//    {
//        for (int j = 0; j < panel.cols; j++)
//        {
//            panel.at<cv::Vec3b>(i, j)[0] = panel.at<cv::Vec3b>(i, j)[0];
//            panel.at<cv::Vec3b>(i, j)[1] = panel.at<cv::Vec3b>(i, j)[1];
//            panel.at<cv::Vec3b>(i, j)[2] = panel.at<cv::Vec3b>(i, j)[2];
//        }
//    }
//}
//void transfrom_Rotate(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed, MatrixXd args) {
//
//    Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();//定义平移矩阵，并初始化为单位阵
//    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
//    if (args.rows() >= 4) {
//        Eigen::Vector4f cloudCentroid;
//        pcl::compute3DCentroid(*cloud, cloudCentroid);//计算点云质心
//        translation(0, 3) = -cloudCentroid[0];
//        translation(1, 3) = -cloudCentroid[1];
//        translation(2, 3) = -cloudCentroid[2];
//    }
//    double x, y, z, theta;
//    std::cout << args.cols() << std::endl;
//    if (args.cols() == 1) {
//        x = args(0, 0);
//        y = args(1, 0);
//        z = args(2, 0);
//        theta = args(3, 0) * M_PI / 180;
//        translation(0, 0) = cos(theta) + (1 - cos(theta)) * x * x;
//        translation(0, 1) = (1 - cos(theta)) * x * y - sin(theta) * z;
//        translation(0, 2) = (1 - cos(theta)) * x * z + sin(theta) * y;
//        translation(1, 0) = (1 - cos(theta)) * y * x + sin(theta) * z;
//        translation(1, 1) = cos(theta) + (1 - cos(theta)) * y * y;
//        translation(1, 2) = (1 - cos(theta)) * y * z - sin(theta) * x;
//        translation(2, 0) = (1 - cos(theta)) * z * x - sin(theta) * y;
//        translation(2, 1) = (1 - cos(theta)) * z * y + sin(theta) * x;
//        translation(2, 2) = cos(theta) + (1 - cos(theta)) * z * z;
//    }
//    else if (args.rows() == 4 && args.cols() == 3) {
//        for (int i = 0; i < args.rows(); i++) {
//            for (int j = 0; j < args.cols(); j++) {
//                translation(i, j) = args(i + 1, j);
//            }
//        }
//    }
//    else {
//        std::cout << "RotateMatrix wrong!" << std::endl;
//        return;
//    }
//    if (DEBUG) {
//        std::cout << std::endl;
//        std::cout << "------------------------------------------------" << std::endl;
//        std::cout << translation << std::endl;
//        std::cout << "------------------------------------------------" << std::endl;
//        std::cout << std::endl;
//    }
//
//    //pcl::transformPointCloud(*cloud, *cloud_transformed, translation);
//    //pointcloudXYZRGB2matrixxd(cloud_transformed, result);
//    // std::cout << result << std::endl;
//    return;
//}
//void transfrom_sphere(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, MatrixXd & result, MatrixXd args = MatrixXd::Zero(4, 1)){
//  
//  result.resize(cloud->points.size(),3);
//
//  double theta,fi,radius;
//  double theta_min= 3,theta_max=-3,fi_min =3,fi_max = -3;
//  double theta_range_min = args(0,0), theta_range_max = args(1,0);
//  double fi_range_min = args(2,0), fi_range_max = args(3,0);
//  for(int i = 0 ; i < cloud->points.size() ; i ++){
//    cloud->points[i].y += 10;
//    radius = cloud->points[i].x * cloud->points[i].x + cloud->points[i].y * cloud->points[i].y;
//    fi = asin(cloud->points[i].y / sqrt(radius));
//
//    radius += cloud->points[i].z * cloud->points[i].z;
//    theta = asin(cloud->points[i].z / sqrt(radius));
//
//    if(theta_min > theta){
//      theta_min = theta;
//    }
//    if(theta_max < theta){
//      theta_max = theta;
//    }
//
//    if(fi_min > fi){
//      fi_min = fi;
//    }
//    if(fi_max < fi){
//      fi_max = fi;
//    }
//
//    result(i,0) = theta;
//    result(i,1) = fi;
//    result(i,2) = sqrt(radius);
//  }
//  double k1 = (theta_range_max - theta_range_min) / (theta_max - theta_min);
//  double k2 = (fi_range_max - fi_range_min) / (fi_max - fi_min);
//  for(int i = 0 ; i < cloud->points.size() ; i ++){
//    result(i,0) = k1 * (result(i,0) - theta_min)  + theta_range_min;
//    result(i,1) = k2 * (result(i,1) - fi_min) + fi_range_min;
//  }
//
//  return;
//}
//void transfrom_sphere(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, MatrixXd & result, MatrixXd args = MatrixXd::Zero(4, 1)){
//  
//  result.resize(cloud->points.size(),4);
//
//  double theta,fi,radius;
//  double theta_min= 3,theta_max=-3,fi_min =3,fi_max = -3;
//  double theta_range_min = args(0,0), theta_range_max = args(1,0);
//  double fi_range_min = args(2,0), fi_range_max = args(3,0);
//  double scale = 0.1;
//  int theta_idx = 0;
//  int fi_idx = 1;
//  int radius_idx = 2;
//  if(args.rows()>=4){
//    scale = args(4,0);
//  }
//
//  for(int i = 0 ; i < cloud->points.size(); i ++){
//    cloud->points[i].z -= scale;
//    radius = cloud->points[i].x * cloud->points[i].x + cloud->points[i].y * cloud->points[i].y;
//    fi = asin(cloud->points[i].y / sqrt(radius));
//
//    radius += cloud->points[i].z * cloud->points[i].z;
//    theta = asin(cloud->points[i].z / sqrt(radius));
//
//    if(theta_min > theta){
//      theta_min = theta;
//    }
//    if(theta_max < theta){
//      theta_max = theta;
//    }
//
//    if(fi_min > fi){
//      fi_min = fi;
//    }
//    if(fi_max < fi){
//      fi_max = fi;
//    }
//
//    result(i, radius_idx) = -1;
//    result(i, theta_idx) = sin(cloud->points[i].y)*cos(cloud->points[i].x);
//    result(i, fi_idx) = sin(cloud->points[i].y) * sin(cloud->points[i].x);//sqrt(radius);
//    result(i, 3) = cloud->points[i].rgb;
//  }
//  //double k1 = (theta_range_max - theta_range_min) / (theta_max - theta_min);
//  //double k2 = (fi_range_max - fi_range_min) / (fi_max - fi_min);
//  //for(int i = 0 ; i < cloud->points.size() ; i ++){
//  //  result(i, theta_idx) = k1 * (result(i, theta_idx) - theta_min)  + theta_range_min;
//  //  result(i, fi_idx) = k2 * (result(i, fi_idx) - fi_min) + fi_range_min;
//  //}
//
//  return;
//}

