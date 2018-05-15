#include "PtCloudReconstructor.h"
#include "ar_track_alvar/DetectFrame.h"

#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <OpenNI.h>
#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <chrono>
#include <thread>
#include <sstream>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

using namespace cv;
using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
  openni::Status rc = openni::STATUS_OK;

  openni::Device device;
  openni::VideoStream depth, color, *streamsD[1], *streamsC[1];
  streamsD[0] = &depth;
  streamsC[0] = &color;

  rc = openni::OpenNI::initialize();
  rc = device.open(openni::ANY_DEVICE);

  if (rc != openni::STATUS_OK)
  {
    printf("Device open failed:\n%s\n", openni::OpenNI::getExtendedError());
    openni::OpenNI::shutdown();
    return 1;
  }

  device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
  rc = depth.create(device, openni::SENSOR_DEPTH);
  rc = color.create(device, openni::SENSOR_COLOR);

  const openni::SensorInfo* sinfo = device.getSensorInfo(openni::SENSOR_DEPTH);
  const openni::Array< openni::VideoMode>& modesDepth = sinfo->getSupportedVideoModes();
  int sensorID = 0;
  for (int i = 0; i < modesDepth.getSize(); i++) {
      if (modesDepth[i].getResolutionX() == 640
              && modesDepth[i].getResolutionY() == 480
              && modesDepth[i].getFps() == 30
              && modesDepth[i].getPixelFormat() == openni::PIXEL_FORMAT_DEPTH_1_MM) {
          sensorID = i;
          break;
      }
  }
  rc = depth.setVideoMode(modesDepth[sensorID]);
  printf("Depth Mode: %ix%i, %i fps, %i format\n", modesDepth[sensorID].getResolutionX(), modesDepth[sensorID].getResolutionY(), modesDepth[sensorID].getFps(), modesDepth[sensorID].getPixelFormat());

  sinfo = device.getSensorInfo(openni::SENSOR_COLOR);
  const openni::Array< openni::VideoMode>& modesColor = sinfo->getSupportedVideoModes();
  sensorID = 0;
  for (int i = 0; i < modesColor.getSize(); i++) {
      if (modesColor[i].getResolutionX() == 640
              && modesColor[i].getResolutionY() == 480
              && modesColor[i].getFps() == 30
              && modesColor[i].getPixelFormat() == openni::PIXEL_FORMAT_RGB888) {
          sensorID = i;
          break;
      }
  }
  rc = color.setVideoMode(modesColor[sensorID]);
  printf("Color Mode: %ix%i, %i fps, %i format\n", modesColor[sensorID].getResolutionX(), modesColor[sensorID].getResolutionY(), modesColor[sensorID].getFps(), modesColor[sensorID].getPixelFormat());

  rc = color.start();
  rc = depth.start();

  openni::VideoMode depthVideoMode = depth.getVideoMode();
  openni::VideoMode colorVideoMode = color.getVideoMode();
  int depthW = depthVideoMode.getResolutionX();
  int depthH = depthVideoMode.getResolutionY();
  int colorW = colorVideoMode.getResolutionX();
  int colorH = colorVideoMode.getResolutionY();
  std::cout << depthW << " " << depthH << std::endl;

  openni::VideoFrameRef depthFR, colorFR;

  Mat colorI(colorH, colorW, CV_8UC3);
  Mat depthI(depthH, depthW, CV_16UC1);

  Matrix3f ir_intrinsics;
  ir_intrinsics << 569.8283037644665, 0, 322.7751444492567, 0, 570.4683437626238, 240.5925536668499, 0, 0, 1;
  VectorXf distortion(5);
  distortion << -0.05599350794855829, 0.2247353063682397, -0.001782436366004092, 0.001198656971150721, -0.4493849715735848;
  Matrix<float, 3, 4> rgb_intrinsics;
  rgb_intrinsics << 535.2900990271, 0.0000000000, 320.0000000000, 0, 0, 535.2900990271, 240.0000000000, 0, 0, 0, 1, 0;
  Affine3f extrinsics = Quaternionf(0.9999819697526113, 0.002884272301322874, 0.003902225023827343, 0.003537482557250309) * Translation3f(-0.02785245055260123, -0.0009402795212491741, 0.01483928478523931);

  PointCloudReconstructor recon(extrinsics, ir_intrinsics, rgb_intrinsics, distortion);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  cloud->height = depthI.rows;
  cloud->width = depthI.cols;
  cloud->is_dense = false;
  cloud->points.resize(cloud->height * cloud->width);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudOld(new pcl::PointCloud<pcl::PointXYZRGB>());
  cloud->height = depthI.rows;
  cloud->width = depthI.cols;
  cloud->is_dense = false;
  cloud->points.resize(cloud->height * cloud->width);

  pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
  visualizer->addPointCloud(cloud, "cloud");
  visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
  visualizer->initCameraParameters();
  visualizer->setPosition(0, 0);
  visualizer->setSize(depthI.cols, depthI.rows);
  visualizer->setShowFPS(true);
  visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);
  //visualizer->setCameraParameters(rgb_intrinsics.leftCols(3), Eigen::Affine3f::Identity().matrix());

  uint8_t ar_tag_data[5][5] = {{255, 255,   0, 255, 255},
                              {255, 255,   0, 255, 255},
                              {255,   0, 255,   0, 255},
                              {0,   255, 255,   0,   0},
                              {255,   0, 255, 255, 255}};
  Mat ar_tag(5, 5, CV_8UC1, &ar_tag_data);

  //DetectFrame ar_detector(20.3);
  DetectFrame ar_detector(5.4);

  // This is our reference start
  Eigen::Affine3f ar_wrt_cam_initial;
  int count = 0;

  while (true) {
    // Throttle
    // std::this_thread::sleep_for(std::chrono::milliseconds(500));

    int streamIndex;
    rc = openni::OpenNI::waitForAnyStream(streamsD, 1, &streamIndex);
    rc = openni::OpenNI::waitForAnyStream(streamsC, 1, &streamIndex);
    depth.readFrame(&depthFR);
    color.readFrame(&colorFR);

    memcpy(colorI.data, colorFR.getData(), colorFR.getDataSize());
    memcpy(depthI.data, depthFR.getData(), depthFR.getDataSize());

    cvtColor(colorI, colorI, CV_RGB2BGR);

    bool success;
    Eigen::Affine3f ar_wrt_cam = ar_detector.detectARTag(colorI, ar_tag, &success);

    if (!success) {
        continue;
    }

    if (count == 0) {
        ar_wrt_cam_initial = ar_wrt_cam;
        visualizer->addCoordinateSystem(0.1, ar_wrt_cam_initial, "ar_tag");
    }

    visualizer->updateCoordinateSystemPose("ar_tag", ar_wrt_cam);

    Matrix<float, 4, Dynamic> cloudPoints = recon.reconstruct(depthI);
    Eigen::Affine3f transform = ar_wrt_cam_initial * ar_wrt_cam.inverse();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudNew = recon.register_depth(colorI, cloudPoints, depthW, depthH, transform);
    //*cloud += *cloudNew;
    count++;

    visualizer->updatePointCloud(cloud, "cloud");
    visualizer->spinOnce(1);

    Eigen::Vector4f ar_centroid_in_cam_frame = ar_wrt_cam * Eigen::Vector4f(0., 0., 0., 1.);
    Eigen::Vector4f ar_xaxis_in_cam_frame = ar_wrt_cam * Eigen::Vector4f(0.1, 0., 0., 1.);
    Eigen::Vector3f image_ar = rgb_intrinsics * ar_centroid_in_cam_frame;
    Eigen::Vector3f image_x_ar = rgb_intrinsics * ar_xaxis_in_cam_frame;

    int ux = (int) (image_ar(0) / image_ar(2));
    int uy = (int) (image_ar(1) / image_ar(2));
    int uxp = (int) (image_x_ar(0) / image_x_ar(2));
    int uyp = (int) (image_x_ar(1) / image_x_ar(2));

    //circle(colorI, cv::Point(ux, uy), 20, Scalar(255,255,255), CV_FILLED);
    arrowedLine(colorI, cv::Point(ux, uy), cv::Point(uxp, uyp), Scalar(60, 255, 0), 5);

    ar_centroid_in_cam_frame = (ar_wrt_cam_initial * ar_wrt_cam.inverse()) * ar_centroid_in_cam_frame;
    ar_xaxis_in_cam_frame = (ar_wrt_cam_initial * ar_wrt_cam.inverse()) * ar_xaxis_in_cam_frame;
    image_ar = rgb_intrinsics * ar_centroid_in_cam_frame;
    image_x_ar = rgb_intrinsics * ar_xaxis_in_cam_frame;
    ux = (int) (image_ar(0) / image_ar(2));
    uy = (int) (image_ar(1) / image_ar(2));
    uxp = (int) (image_x_ar(0) / image_x_ar(2));
    uyp = (int) (image_x_ar(1) / image_x_ar(2));
    arrowedLine(colorI, cv::Point(ux, uy), cv::Point(uxp, uyp), Scalar(255, 255, 128), 5);
    //circle(colorI, cv::Point(ux, uy), 20, Scalar(255,255,128), CV_FILLED);


    imshow("image", colorI);
    int c = waitKey(1);
    if (c != -1) {

        if (c == 117) { // u for undo
            *cloud = *cloudOld;
            cout << "Undone!" << endl;
        } else {
            *cloudOld = *cloud;
            *cloud += *cloudNew;
            cout << "Saved!" << endl;
        }

        pcl::io::savePLYFileBinary("/home/rishi/Desktop/soylentimgs/cloud.ply", *cloud);
    }
  }

  return 0;
}
