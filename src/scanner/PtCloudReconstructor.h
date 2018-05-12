#ifndef PT_CLOUD_RECONSTRUCTOR_H
#define PT_CLOUD_RECONSTRUCTOR_H

#include <opencv/cv.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

class PointCloudReconstructor {
private:
    Eigen::Affine3f extrinsics_; // rigid body motion from rgb to ir camera
    const Eigen::Matrix3f &ir_intrinsics_;
    const Eigen::Matrix<float, 3, 4> &rgb_intrinsics_;
    const Eigen::VectorXf &distortion_coeffs_;

public:
  PointCloudReconstructor(
    const Eigen::Affine3f &extrinsics, const Eigen::Matrix3f &ir_intrinsics
    , const Eigen::Matrix<float, 3, 4> &rgb_intrinsics
    , const Eigen::VectorXf &distortion_coeffs);

  //vector<PointXYZ> reconstruct(const Mat &depthImage) {
  //    pcl::PointCloud<pcl::PointXYZ>::Ptr reconstruct(
  //const Mat &depthImageDistorted) {
  Eigen::Matrix<float, 4, Eigen::Dynamic> reconstruct(const cv::Mat &depthImageDistorted);
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
    register_depth(
      const cv::Mat &rgbImage, const Eigen::Matrix<float, 4, Eigen::Dynamic> &reconstructedCloud
      , int width, int height);
};

#endif
