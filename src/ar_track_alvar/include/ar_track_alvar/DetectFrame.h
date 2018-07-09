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
#include <unordered_map>
#include <Eigen/Dense>

namespace alvar {

struct ARTag {
    int id;
    Eigen::Affine3f pose;
    std::vector<Eigen::Vector2f> image_corners;
    std::vector<Eigen::Vector3f> model_corners;
    std::vector<Eigen::Vector3f> camera_corners; // Corners in the camera frame
};

class ALVAR_EXPORT DetectFrame {

public:
    std::unordered_map<int, Eigen::Affine3f> tfToLeader;

    DetectFrame(double marker_size_) :
        marker_size(marker_size_),
        // Xtion intrinsics 535.2900990271, 0.0000000000, 320.0000000000, 0, 0, 535.2900990271, 240.0000000000, 0, 0, 0, 1, 0
        cam(535.2900990271, 535.2900990271, 320.0000000000, 240.0000000000, 640., 480.) {
            marker_detector.SetMarkerSize(marker_size, marker_resolution, marker_margin);

            for (int mid : marker_ids) {
                tfToLeader.insert({mid, Eigen::Affine3f::Identity()});
            }
        }	

    std::vector<ARTag> detectARTags(const cv::Mat &image);	

private:
    Camera cam;
    MarkerDetector<MarkerData> marker_detector;
    std::vector<int> marker_ids = {10, 7, 13, 220};
    int leader_id = -1;

    double marker_size;
    double max_new_marker_error = 0.08;
    double max_track_error = 0.2;
    int marker_resolution = 5; // default marker resolution
    int marker_margin = 2; // default marker margin

    Eigen::Affine3f optimizeRT(std::vector<ARTag> &ar_tags);
};

} // namespace alvar

#endif
