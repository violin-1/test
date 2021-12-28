#include "pcllib.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <typeconvert.h>

MatrixXd abstract_cube_api(MatrixXd points, MatrixXd args){
  if(points.cols() == 1){
      points.transposeInPlace();
  }
  if(points.cols() == 3){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr res_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    matrixxd2pointcloudXYZ(points,cloud);
    MatrixXd result;
    abstract_cube(cloud, res_cloud, args);
    pointcloudXYZ2matrixxd(res_cloud, result);
    return result;
  }
  else if(points.cols() == 6){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr res_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    matrixxd2pointcloudXYZRGB(points,cloud);
    MatrixXd result;
    abstract_cube(cloud, res_cloud, args);
    pointcloudXYZRGB2matrixxd(res_cloud, result, args(6,0));
    return result;
  }
}
// MatrixXd transfrom_sphere_api(MatrixXd points, MatrixXd args){
//   if(points.cols() == 1){
//       points.transposeInPlace();
//   }
//   if(points.cols() == 3){
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     matrixxd2pointcloudXYZ(points,cloud);
//     MatrixXd result;
//     transfrom_sphere(cloud, result, args);
//     return result;
//   }
//   else if(points.cols() == 4){
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
//     matrixxd2pointcloudXYZC(points,cloud);
//     MatrixXd result;
//     transfrom_sphere(cloud, result, args);
//     return result;
//   }
// }
MatrixXd transfrom_Rotation_api(MatrixXd points, MatrixXd args){

  if(points.cols() == 1){
      points.transposeInPlace();
  }
  if(points.cols() == 3){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    matrixxd2pointcloudXYZ(points,cloud);
    if(DEBUG){
      std::cout << "------------------------------------------------"<<std::endl;
      std::cout << "rows: " << args.rows() << "width: "<< args.cols() << std::endl;
      std::cout << args << std::endl;
      std::cout << "------------------------------------------------"<<std::endl;
    }
    MatrixXd result;
    transfrom_Rotate(cloud, cloud_transformed, args);
    pointcloudXYZ2matrixxd(cloud_transformed,result);
    return result;
  }
  else if(points.cols() == 6){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    matrixxd2pointcloudXYZ(points,cloud);
    if(DEBUG){
      std::cout << "------------------------------------------------"<<std::endl;
      std::cout << "rows: " << args.rows() << "width: "<< args.cols() << std::endl;
      std::cout << args << std::endl;
      std::cout << "------------------------------------------------"<<std::endl;
    }
    transfrom_Rotate(cloud, cloud_transformed, args);
    MatrixXd result;
    pointcloudXYZ2matrixxd(cloud_transformed,result);
    return result;
  }
  
}

std::map<std::string, MatrixXd> readPCLfile(std::vector<std::string>& files, MatrixXd args) {
    std::map<std::string, MatrixXd> out;
    MatrixXd result;
    for (int i = 0; i < files.size(); i++)
    {
      if(args(0,0) == 3){
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        readPCLXYZfile(&files[i][0],cloud);
        pointcloudXYZ2matrixxd(cloud, result);
      }
      else if(args(0,0) == 4){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        readPCLXYZCfile(&files[i][0],cloud);
        pointcloudXYZRGB2matrixxd(cloud, result, args(1,0));
      }
      else if(args(0,0) == 6){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        readPCLXYZRGBfile(&files[i][0],cloud);
        pointcloudXYZRGB2matrixxd(cloud, result, args(1,0));
      }
      out.insert({ files[i], result });
    }
    return out;
}

std::map<std::string, MatrixXd> readOBJfile(std::vector<std::string>& files, MatrixXd args) {
    std::map<std::string, MatrixXd> out;
    MatrixXd result;
    for (int i = 0; i < files.size(); i++)
    {
      vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
      readSTLfile(&files[i][0],polydata);
      uniform_sampling(polydata, args(0,0), *cloud);
      pointcloudXYZ2matrixxd(cloud, result);
      out.insert({ files[i], result });
    }
    return out;
}

MatrixXd en_de_code_color(MatrixXd points,MatrixXd args){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  MatrixXd result;
  if(points.cols() == 1){
      points.transposeInPlace();
  }
  if(points.cols() == 4){
    matrixxd2pointcloudXYZRGB(points,cloud);
    pointcloudXYZRGB2matrixxd(cloud,result,args(0,0));
  }
  else if(points.cols() == 6){
    matrixxd2pointcloudXYZRGB(points,cloud);
    pointcloudXYZRGB2matrixxd(cloud,result,args(0,0));
  }
  else{
    result.resize(1,1);
    result(0,0) = 0;
  }

  return result;
}

namespace py = pybind11;

std::vector<float> vec_add(std::vector<float>& a, std::vector<float>& b) {

    std::vector<float> out;
    assert(a.size() == b.size());
    for (int i = 0; i < a.size(); i++)
    {
        out.push_back(a[i] + b[i]);
    }

    return out;

}

std::array<float, 20> vec_sin(std::array<float, 20>& x) {
    std::array<float, 20> out;
    for (int i = 0; i < x.size(); i++)
    {
        out[i] = sinf(i);
    }
    return out;
}

std::map<std::string, int> get_map(std::vector<std::string>& keys, std::vector<int> values) {
     
    std::map<std::string, int> out;
    
    for (int i = 0; i < keys.size(); i++)
    {
        out.insert({ keys[i], values[i] });
    }

    return out;
}

std::set<std::string> get_set(std::vector<std::string>& values) {



    std::set<std::string> out;
    for (auto& i:values)
    {
        std::cout << i << std::endl;
        out.insert(i);
    }

    return out;
}

MatrixXd render_ball(MatrixXd points, MatrixXd args){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  MatrixXd result;
  if(points.cols() == 1){
      points.transposeInPlace();
  }
  if(points.cols() == 4){
    matrixxd2pointcloudXYZRGB(points,cloud);
    pcl_render_ball(cloud,result,args);
  }
  else if(points.cols() == 6){
    matrixxd2pointcloudXYZRGB(points,cloud);
    pcl_render_ball(cloud,result,args);
  }
  else{
    result.resize(1,1);
    result(0,0) = 0;
  }

  return result;
}

PYBIND11_MODULE(pclpyd,m)
{
  m.doc() = "pybind11 example plugin";
  m.def("abstract_cube", &abstract_cube_api, py::return_value_policy::reference_internal );
  // m.def("transfrom_sphere", &transfrom_sphere_api, py::return_value_policy::reference_internal );
  m.def("transfrom_Rotation", &transfrom_Rotation_api, py::return_value_policy::reference_internal );
  m.def("readPCLfile", &readPCLfile, py::return_value_policy::reference_internal );
  m.def("readOBJfile", &readOBJfile, py::return_value_policy::reference_internal );
  m.def("vec_add", &vec_add, py::return_value_policy::reference_internal);
  m.def("vec_sin", &vec_sin, py::return_value_policy::reference_internal);
  m.def("get_map", &get_map, py::return_value_policy::reference_internal);
  m.def("get_set", &get_set, py::return_value_policy::reference_internal);
  m.def("endecode_color", &en_de_code_color, py::return_value_policy::reference_internal);
  m.def("render_ball", &render_ball, py::return_value_policy::reference_internal);
}