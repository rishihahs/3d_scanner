#ifndef DETECT_FRAME_H
#define DETECT_FRAME_H

#include "Alvar.h"
#include "Pose.h"
#include "Util.h"
#include "CvTestbed.h"
#include "MarkerDetector.h"
#include "Shared.h"
#include "FileFormat.h"
#include <vector>
#include <Eigen/Dense>

namespace alvar {

class ALVAR_EXPORT DetectFrame {

public:
    DetectFrame(double marker_size_) :
        marker_size(marker_size_),
        // Xtion intrinsics 535.2900990271, 0.0000000000, 320.0000000000, 0, 0, 535.2900990271, 240.0000000000, 0, 0, 0, 1, 0
        cam(535.2900990271, 535.2900990271, 320.0000000000, 240.0000000000, 640., 480.) {
            marker_detector.SetMarkerSize(marker_size, marker_resolution, marker_margin);
        }	

    Eigen::Affine3f detectARTag(const cv::Mat &image, const cv::Mat &ar_tag, bool *success, std::vector<Eigen::Vector2f> &image_corners, std::vector<Eigen::Vector3f> &model_corners);	

private:
    Camera cam;
    MarkerDetector<MarkerData> marker_detector;

    double marker_size;
    double max_new_marker_error = 0.08;
    double max_track_error = 0.2;
    int marker_resolution = 5; // default marker resolution
    int marker_margin = 2; // default marker margin

};

} // namespace alvar

#endif
