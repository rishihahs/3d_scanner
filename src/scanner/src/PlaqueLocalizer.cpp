#include "PlaqueLocalizer.h"
#include "ModelPlaque.h"
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include "mapping_calibration/CameraIntrinsics.h"
#include "mapping_calibration/HomographyShortcuts.h"
#include "mapping_calibration/RigidTrans.h"

namespace {
    const ModelPlaque kPlaqueModel(0.054, 0.054);
}

/*
 * Localizes plaque (i.e. plaque w.r.t. camera)
 * corners must be in clockwise order from top left
 */
RigidTrans localizePlaque(CameraIntrinsics k,
                          const std::vector<cv::Point> &corners,
                          bool optimize) {
  assert(corners.size() == 4);

  // Set up corners
  Eigen::MatrixXd corns(2, 4);
  corns(0, 0) = corners[0].x;
  corns(1, 0) = corners[0].y;
  corns(0, 1) = corners[1].x;
  corns(1, 1) = corners[1].y;
  corns(0, 2) = corners[2].x;
  corns(1, 2) = corners[2].y;
  corns(0, 3) = corners[3].x;
  corns(1, 3) = corners[3].y;

  std::vector<Eigen::MatrixXd> plaquePoints;
  plaquePoints.push_back(corns);

  // Get our homography.
  std::vector<Eigen::MatrixXd> homographies =
      computeHomographies(kPlaqueModel.getModelPH2D(), plaquePoints);

  std::vector<RigidTrans> plaqueRTs = extractRTs(k, homographies);
  RigidTrans rt = plaqueRTs[0];

  Eigen::Vector3d a(0., 0., 0.);
  Eigen::Vector3d b(0.054, 0., 0.);
  Eigen::Vector3d c(0.054, 0.054, 0.);
  Eigen::Vector3d d(0., 0.054, 0.);

  std::vector<Eigen::Vector3d> world_corners;
  world_corners.push_back(a);
  world_corners.push_back(b);
  world_corners.push_back(c);
  world_corners.push_back(d);

  // Optimize that baby
  if (optimize) {
    rt = optimizeRT(k, corns, world_corners, rt);
  }

  return rt;
}
