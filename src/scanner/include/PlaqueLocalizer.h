#ifndef PLAQUE_LOCALIZER_H
#define PLAQUE_LOCALIZER_H

#include <Eigen/Dense>
#include <cv.hpp>

#include <vector>

class CameraIntrinsics;
class RigidTrans;

RigidTrans localizePlaque(CameraIntrinsics k,
                          const std::vector<cv::Point> &corners, bool optimize);

#endif
