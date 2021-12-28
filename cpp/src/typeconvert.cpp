#include <typeconvert.h>
void matrixxd2pointcloudXYZ(MatrixXd input, pcl::PointCloud<pcl::PointXYZ>::Ptr output){
    //if (output->points.size() == 0) {
    //    pcl::PointXYZ temp;
    //    output->height = 1;
    //    output->width = input.rows();
    //    for (int i = 0; i < input.rows(); i++) {
    //        temp.x = input(i, 0);
    //        temp.y = input(i, 1);
    //        temp.z = input(i, 2);
    //        output->points.push_back(temp);

    //    }
    //}
    //else{
    output->points.resize(input.rows());
    for (int i = 0; i < input.rows(); i++) {
        output->points[i].x = input(i, 0);
        output->points[i].y = input(i, 1);
        output->points[i].z = input(i, 2);
    }
    //}
}
void matrixxd2pointcloudXYZRGB(MatrixXd input, pcl::PointCloud<pcl::PointXYZRGB>::Ptr output) {
    //if (output->points.size() == 0) {
    //    pcl::PointXYZRGB temp;
    //    output->height = 1;
    //    output->width = input.rows();
    //    for (int i = 0; i < input.rows(); i++) {
    //        temp.x = input(i, 0);
    //        temp.y = input(i, 1);
    //        temp.z = input(i, 2);
    //        temp.r = input(i, 3);
    //        temp.g = input(i, 4);
    //        temp.b = input(i, 5);
    //        output->points.push_back(temp);

    //    }
    //}
    //else{
    output->points.resize(input.rows());
    for (int i = 0; i < input.rows(); i++) {
        if (input.cols() == 6) {
            output->points[i].x = input(i, 0);
            output->points[i].y = input(i, 1);
            output->points[i].z = input(i, 2);
            output->points[i].r = input(i, 3);
            output->points[i].g = input(i, 4);
            output->points[i].b = input(i, 5);
        }
        else if (input.cols() == 4) {
            output->points[i].x = input(i, 0);
            output->points[i].y = input(i, 1);
            output->points[i].z = input(i, 2);
            output->points[i].rgb = convertColor(input(i, 3));
        }

    }
    //}
}
void pointcloudXYZ2matrixxd(pcl::PointCloud<pcl::PointXYZ>::Ptr input, MatrixXd & output){
    output.resize(input->points.size(),3);
    for(int i = 0 ; i < input->points.size(); i ++){
        output(i,0) = input->points[i].x;
        output(i,1) = input->points[i].y;
        output(i,2) = input->points[i].z;
    }
}
void pointcloudXYZRGB2matrixxd(pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputs, MatrixXd& output, int outputc) {
    output.resize(inputs->points.size(), outputc);
    if (outputc == 6) {
        for (int i = 0; i < inputs->points.size(); i++) {
            output(i, 0) = inputs->points[i].x;
            output(i, 1) = inputs->points[i].y;
            output(i, 2) = inputs->points[i].z;
            output(i, 3) = inputs->points[i].r;
            output(i, 4) = inputs->points[i].g;
            output(i, 5) = inputs->points[i].b;
        }
    }
    else if (outputc == 4) {
        for (int i = 0; i < inputs->points.size(); i++) {
            output(i, 0) = inputs->points[i].x;
            output(i, 1) = inputs->points[i].y;
            output(i, 2) = inputs->points[i].z;
            output(i, 3) = inputs->points[i].rgb;
        }
    }
    else if (outputc == 3) {
        for (int i = 0; i < inputs->points.size(); i++) {
            output(i, 0) = inputs->points[i].x;
            output(i, 1) = inputs->points[i].y;
            output(i, 2) = inputs->points[i].z;
        }
    }
}
void pointcloudXYZRGB2pointcloudXYZ(pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputs, pcl::PointCloud<pcl::PointXYZ>::Ptr outputs){
    //pcl::PointXYZ tmp;
    //if (outputs->points.size() == 0) {
    //    for (int i = 0; i < inputs->points.size(); i++) {
    //        tmp.x = inputs->points[i].x;
    //        tmp.y = inputs->points[i].y;
    //        tmp.z = inputs->points[i].z;
    //        outputs->points.push_back(tmp);
    //    }
    //}
    //else {
    outputs->points.resize(inputs->points.size());
    for (int i = 0; i < inputs->points.size(); i++) {
        outputs->points[i].x = inputs->points[i].x;
        outputs->points[i].y = inputs->points[i].y;
        outputs->points[i].z = inputs->points[i].z;

    }

    //}
}
float convertColor(float colorIn) {
    uint32_t color_uint = *(uint32_t*)&colorIn;
    unsigned char* color_uchar = (unsigned char*)&color_uint;
    //printf("%u\n", color_uint);
    color_uint = ((uint32_t)color_uchar[0] << 16 | (uint32_t)color_uchar[1] << 8 | (uint32_t)color_uchar[2]);
    return *reinterpret_cast<float*> (&color_uint);
}
void convert2rgb(float colorIn, float*  rgb) {
    std::vector<float> color;
    uint32_t color_uint = *(uint32_t*)&colorIn;
    unsigned char* color_uchar = (unsigned char*)&color_uint;
    //color_uint = ((uint32_t)color_uchar[0] << 16 | (uint32_t)color_uchar[1] << 8 | (uint32_t)color_uchar[2]);
    rgb[0] = (float)color_uchar[0];
    rgb[1] = (float)color_uchar[1];
    rgb[2] = (float)color_uchar[2];
    return ;
}
float rgbconvert2color_(float* rgb) {
    uint32_t color_uint = ((uint32_t)rgb[0] << 16 | (uint32_t)rgb[1] << 8 | (uint32_t)rgb[2]);
    return *reinterpret_cast<float*> (&color_uint);
}
double rgbconvert2color(float* rgb) {
    uint32_t color_uint = ((uint32_t)rgb[0] << 16 | (uint32_t)rgb[1] << 8 | (uint32_t)rgb[2]);
    double x = (double)(int)((-1 << 23) - color_uint) * (1 << 26) * (1 << 26) * (1 << 26) * (1 << 26);
    return x;
}
float convertColor(double colorIn) {
    float color = (float)(colorIn / (1 << 26));
    uint32_t color_uint = *(uint32_t*)&color;
    unsigned char* color_uchar = (unsigned char*)&color_uint;
    //printf("%u\n", color_uint);
    color_uint = ((uint32_t)color_uchar[0] << 16 | (uint32_t)color_uchar[1] << 8 | (uint32_t)color_uchar[2]);
    return *reinterpret_cast<float*> (&color_uint);
}