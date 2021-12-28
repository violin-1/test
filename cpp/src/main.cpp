///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/************************************************************************************
 ** This sample demonstrates how to use PCL (Point Cloud Library) with the ZED SDK **
 ************************************************************************************/

 // ZED includes
#include <sl/Camera.hpp>

// PCL includes
// Undef on Win32 min/max for PCL
#ifdef _WIN32
#undef max
#undef min
#endif


// Sample includes
//#include <thread>
//#include <mutex>
#include "pcllib.h"
#include "typeconvert.h"

// Namespace
using namespace sl;
using namespace std;

// Global instance (ZED, Mat, callback)
Camera zed;
Mat data_cloud;
std::thread zed_callback;
std::mutex mutex_input;
bool stop_signal;
bool has_data;

// Sample functions
void startZED();
void run();
void closeZED();
shared_ptr<pcl::visualization::PCLVisualizer> createRGBVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);
inline float convertColor(float colorIn);

sl::Resolution cloud_res;
sl::RuntimeParameters rt_parameters;

// Main process

int ZEDtest(int argc, char** argv) {

    if (argc > 2) {
        cout << "Only the path of a SVO can be passed in arg" << endl;
        return -1;
    }

    // Set configuration parameters
    InitParameters init_params;
    if (argc == 2)
        init_params.input.setFromSVOFile(argv[1]);
    else {
        init_params.camera_resolution = RESOLUTION::VGA;
        init_params.camera_fps = 100;
    }
    init_params.coordinate_units = UNIT::METER;
    init_params.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_params.depth_mode = DEPTH_MODE::QUALITY;
    init_params.depth_maximum_distance = 30;
    rt_parameters.confidence_threshold = 100;
    rt_parameters.texture_confidence_threshold = 100;
    rt_parameters.sensing_mode = sl::SENSING_MODE::STANDARD;

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        cout << toString(err) << endl;
        zed.close();
        return 1;
    }

    cloud_res = sl::Resolution(640, 360);
    Eigen::MatrixXd args(5, 1);
    args(0, 0) = -180;
    args(1, 0) = 180;
    args(2, 0) = -360;
    args(3, 0) = 360;
    args(4, 0) = 0;
//Eigen:MatrixXd tmp;


    // Allocate PCL point cloud at the resolution
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr p_pcl_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    p_pcl_point_cloud->points.resize(cloud_res.area());

    // Create the PCL point cloud visualizer
    shared_ptr<pcl::visualization::PCLVisualizer> viewer = createRGBVisualizer(p_pcl_point_cloud);

    // Start ZED callback
    startZED();

    // Set Viewer initial position
    viewer->setCameraPosition(0, 0, 2, 0, 0, 1, 0, 1, 0);
    viewer->setCameraClipDistances(0.1, 1000);

    // Loop until viewer catches the stop signal
    while (!viewer->wasStopped()) {

        //Lock to use the point cloud
        mutex_input.lock();
        float* p_data_cloud = data_cloud.getPtr<float>();
        int index = 0;

        // Check and adjust points for PCL format
        for (auto& it : p_pcl_point_cloud->points) {
            float X = p_data_cloud[index];
            if (!isValidMeasure(X)) // Checking if it's a valid point
                it.x = it.y = it.z = it.rgb = 0;
            else {
                it.x = X;
                it.y = p_data_cloud[index + 1];
                it.z = p_data_cloud[index + 2];
                it.rgb = convertColor(p_data_cloud[index + 3]); // Convert a 32bits float into a pcl .rgb format
                //printf("ori:r:%d,g:%d,b:%d,rgba:%d\n", it.r, it.g, it.b,it.rgba);
                //float tmp[3];
                //printf("%.2f\n", p_data_cloud[index + 3]);
                //convert2rgb(p_data_cloud[index + 3],tmp);
                //it.r = tmp[0];
                //it.g = tmp[1];
                //it.b = tmp[2];
                //printf("%.2f %.2f %.2f\n", tmp[0], tmp[1], tmp[2]);
                //float x = rgbconvert2color_(tmp);
                //printf("%lf\n", x);
                //it.rgb = convertColor(x);
                //printf("ori:r:%d,g:%d,b:%d,rgba:%d\n", it.r, it.g, it.b, it.rgba);
                //printf("END\n");
            }
            index += 4;
        }

        //MatrixXd result1;
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr test1(new pcl::PointCloud<pcl::PointXYZRGB>);
        //pointcloudXYZRGB2matrixxd(p_pcl_point_cloud, result1);
        //matrixxd2pointcloudXYZRGB(result1, test1);

        //MatrixXd result2;
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr test2(new pcl::PointCloud<pcl::PointXYZRGB>);
        //pointcloudXYZC2matrixxd(p_pcl_point_cloud, result2);
        //matrixxd2pointcloudXYZRGB(result2, test2);
        
        //MatrixXd result, args(4, 1);
        //args(0, 0) = 800;
        //args(1, 0) = 800;
        //args(2, 0) = p_pcl_point_cloud->points.size();
        //args(3, 0) = 1;
        //double radius = 0, meanx = 0, meany = 0, meanz = 0, tmp = 0;
        //for (int i = 0; i < args(2, 0); i++) {
        //    meanx += p_pcl_point_cloud->points[i].x;
        //    meany += p_pcl_point_cloud->points[i].y;
        //    meanz += p_pcl_point_cloud->points[i].z;

        //    tmp = 0;
        //    tmp += p_pcl_point_cloud->points[i].x * p_pcl_point_cloud->points[i].x;
        //    tmp += p_pcl_point_cloud->points[i].y * p_pcl_point_cloud->points[i].y;
        //    tmp += p_pcl_point_cloud->points[i].z * p_pcl_point_cloud->points[i].z;
        //    if (tmp > radius) {
        //        radius = tmp;
        //    }
        //}
        //meanx = meanx / args(2, 0);
        //meany = meany / args(2, 0);
        //meanz = meanz / args(2, 0);
        //radius = std::sqrt(radius);
        //for (int i = 0; i < args(2, 0); i++) {
        //    p_pcl_point_cloud->points[i].x = p_pcl_point_cloud->points[i].x - meanx;
        //    p_pcl_point_cloud->points[i].x = p_pcl_point_cloud->points[i].x * args(0, 0) / (2.2 * radius);
        //    p_pcl_point_cloud->points[i].x = p_pcl_point_cloud->points[i].x + args(0,0) / 2;

        //    p_pcl_point_cloud->points[i].y = p_pcl_point_cloud->points[i].y - meany;
        //    p_pcl_point_cloud->points[i].y = p_pcl_point_cloud->points[i].y * args(0, 0) / (2.2 * radius);
        //    p_pcl_point_cloud->points[i].y = p_pcl_point_cloud->points[i].y + args(0, 0) / 2;

        //    p_pcl_point_cloud->points[i].z = p_pcl_point_cloud->points[i].z - meanz;
        //    p_pcl_point_cloud->points[i].z = p_pcl_point_cloud->points[i].z * args(0, 0) / (2.2 * radius);
        //}


        //pcl_render_ball(p_pcl_point_cloud, result, args);
        // Unlock data and update Point cloud
        mutex_input.unlock();
        //transfrom_sphere(p_pcl_point_cloud, tmp, args);
        //matrixxd2pointcloudRGB(tmp, p_pcl_point_cloud);
        viewer->updatePointCloud(p_pcl_point_cloud);
        viewer->spinOnce(10);
    }

    // Close the viewer
    viewer->close();

    // Close the zed
    closeZED();

}
/**
 *  This functions start the ZED's thread that grab images and data.
 **/
void startZED() {
    // Start the thread for grabbing ZED data
    stop_signal = false;
    has_data = false;
    zed_callback = std::thread(run);

    //Wait for data to be grabbed
    while (!has_data)
        sleep_ms(1);
}

/**
 *  This function loops to get the point cloud from the ZED. It can be considered as a callback.
 **/
void run() {
    while (!stop_signal) {
        if (zed.grab(rt_parameters) == ERROR_CODE::SUCCESS) {
            mutex_input.lock(); // To prevent from data corruption
            zed.retrieveMeasure(data_cloud, MEASURE::XYZRGBA, MEM::CPU, cloud_res);
            mutex_input.unlock();
            has_data = true;
        }
        else
            sleep_ms(1);
    }
}

/**
 *  This function frees and close the ZED, its callback(thread) and the viewer
 **/
void closeZED() {
    // Stop the thread
    stop_signal = true;
    zed_callback.join();
    zed.close();
}

/**
 *  This function creates a PCL visualizer
 **/
shared_ptr<pcl::visualization::PCLVisualizer> createRGBVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) {
    // Open 3D viewer and add point cloud
    shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("PCL ZED 3D Viewer"));
    viewer->setBackgroundColor(0.12, 0.12, 0.12);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return (viewer);
}

int main(int argc, char** argv) {
    //ZEDtest(argc, argv);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outputs(new pcl::PointCloud<pcl::PointXYZRGB>);
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    //std::string str = "C:\\Users\\VioLin\\Documents\\GitHub\\kunbo_team\\datasets\\3dvisionpcl\\.XYZ\\1-1.xyz";
    std::string str = "C:\\Users\\VioLin\\Documents\\GitHub\\kunbo_team\\datasets\\solidwork\\test1.STL";
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    readSTLfile(&str[0], polydata);
    size_t n_samples = 100;
    uniform_sampling(polydata, n_samples, *cloud);
    //pcl::io::vtkPolyDataToPointCloud(polydata, *cloud_xyz);
    //pcl::visualization::PCLVisualizer visu3("stl2pcd");
    //// visu3.setBackgroundColor(255, 255, 255);
    //visu3.addPointCloud(cloud_xyz, "input_cloud");
    //visu3.spin();
    
    //PCLVisualizer 显示原STL文件
    pcl::visualization::PCLVisualizer vis;
    //保存pcd文件
    pcl::io::savePCDFileASCII("C:\\Users\\VioLin\\Documents\\GitHub\\kunbo_team\\datasets\\solidwork\\test1.pcd", *cloud);
    while (!vis.wasStopped())
    {
        vis.spinOnce();
    }
    //保存pcd文件
    //readPCLXYZRGBfile(&str[0],cloud);
    //MatrixXd args(6, 1);
    //args << -500, 500, -500, 500, 0, 1000;
    ////pointcloudXYZRGB2pointcloudXYZ(cloud, cloud_xyz);
    //abstract_cube(cloud, outputs, args);

    return 0;
}

