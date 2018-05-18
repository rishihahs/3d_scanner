#include "PtCloudReconstructor.h"
#include <pcl/filters/statistical_outlier_removal.h>

using namespace cv;
using namespace Eigen;

PointCloudReconstructor::PointCloudReconstructor(const Affine3f &extrinsics, const Matrix3f &ir_intrinsics, const Matrix<float, 3, 4> &rgb_intrinsics, const VectorXf &distortion_coeffs)
  : extrinsics_(extrinsics),
    ir_intrinsics_(ir_intrinsics),
    rgb_intrinsics_(rgb_intrinsics),
    distortion_coeffs_(distortion_coeffs) {}

Matrix<float, 4, Dynamic> PointCloudReconstructor::reconstruct(
  const Mat &depthImageDistorted) {
  Matrix<float, 4, Dynamic>
    points(4, depthImageDistorted.rows * depthImageDistorted.cols);

  //Mat depthImage;
  //Mat intrinsics(3, 3, CV_32FC1);
  //Mat distortion(5, 1, CV_32FC1);
  //eigen2cv(ir_intrinsics_, intrinsics);
  //eigen2cv(distortion_coeffs_, distortion);
  //cv::undistort(depthImageDistorted, depthImage, intrinsics, distortion);
  Mat depthImage = depthImageDistorted;

  float sx = ir_intrinsics_(0, 0);
  float sy = ir_intrinsics_(1, 1);
  float x_c = ir_intrinsics_(0, 2);
  float y_c = ir_intrinsics_(1, 2);

  float *pointsData = points.data(); // Raw data pointer for greater efficiency

  for (int y = 0; y < depthImage.rows; ++y) {
    const uint16_t *row = depthImage.ptr<uint16_t>(y);

    for (int x = 0; x < depthImage.cols; ++x) {
      // If depth is 0, skip
      if (row[x] == 0.) {
          continue;
      }

      // Analytic solution to intrinsics^-1(point) * depth
      // Eigen is column major order, so *4 is column size
      pointsData[(depthImage.rows*y + x)*4 + 0] =
        (x - x_c) * (1.0 / sx) * (row[x] / 1000.0f);
      pointsData[(depthImage.rows*y + x)*4 + 1] =
        (y - y_c) * (1.0 / sy) * (row[x] / 1000.0f);

      pointsData[(depthImage.rows*y + x)*4 + 2] = row[x] / 1000.0f;
      pointsData[(depthImage.rows*y + x)*4 + 3] = 1.0f;
    }
  }

  return points;
  //return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudReconstructor::register_depth(const Mat &rgbImage, const Matrix<float, 4, Dynamic> &reconstructedCloud, int width, int height, const Eigen::Affine3f &transformation) {
    Matrix<float, 4, Dynamic> cloudInRGB = extrinsics_.inverse() * reconstructedCloud;
    //Matrix<float, 4, Dynamic> cloudInRGB = reconstructedCloud;
    Matrix<float, 3, Dynamic> pixelCorresp = rgb_intrinsics_ * cloudInRGB;
    Matrix<float, 4, Dynamic> cloudTransformed = transformation * cloudInRGB;

    // Raw data pointer for greater efficiency
    const float *reconCloudData = reconstructedCloud.data();
    const float *cloudInRGBData = cloudInRGB.data();
    const float *pixelCorrespData = pixelCorresp.data();
    const float *cloudTransformedData = cloudTransformed.data();


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->height = height;
    cloud->width = width;
    cloud->is_dense = false;
    cloud->points.resize(cloud->height * cloud->width);

    pcl::PointXYZRGB *pclCloud = &cloud->points[0];
    for (int i = 0; i < cloudInRGB.cols(); ++i) {
        float tfx = cloudTransformedData[i*4 + 0] / cloudTransformedData[i*4 + 3];
        float tfy = cloudTransformedData[i*4 + 1] / cloudTransformedData[i*4 + 3];
        float tfz = cloudTransformedData[i*4 + 2] / cloudTransformedData[i*4 + 3];

        float dist = tfx*tfx + tfy*tfy + tfz*tfz;
        if (reconCloudData[i*4 + 2] != 0. && dist <= 0.8) {
            int x_pixel = (int) (pixelCorrespData[i*3 + 0] / pixelCorrespData[i*3 + 2]);
            int y_pixel = (int) (pixelCorrespData[i*3 + 1] / pixelCorrespData[i*3 + 2]);
            if (
                (x_pixel >= 0) && (x_pixel <= width) &&
                (y_pixel >= 0) && (y_pixel <= height)
              ) {

                pclCloud[i].x = tfx;
                pclCloud[i].y = tfy;
                pclCloud[i].z = tfz;

                Vec3b color = rgbImage.at<Vec3b>(y_pixel, x_pixel);
                pclCloud[i].b = color[0];
                pclCloud[i].g = color[1];
                pclCloud[i].r = color[2];
            }
        } else {
            pclCloud[i].x = pclCloud[i].y = pclCloud[i].z = std::numeric_limits<float>::quiet_NaN();
            pclCloud[i].rgb = 0;
        }
    }

    std::cout << "Filtering Point Cloud" << std::endl;
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_filtered);
    std::cout << "=== Finished Filtering Point Cloud" << std::endl;

    return cloud_filtered;
}

