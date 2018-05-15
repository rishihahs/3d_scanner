#include "ar_track_alvar/DetectFrame.h"
#include <opencv2/highgui/highgui.hpp>

using namespace alvar;
using namespace std;

Eigen::Affine3f DetectFrame::detectARTag(const cv::Mat &image, const cv::Mat &ar_tag, bool *success) {
    // GetMultiMarkersPoses expects an IplImage*, but as of ros groovy, cv_bridge gives
    // us a cv::Mat. I'm too lazy to change to cv::Mat throughout right now, so I
    // do this conversion here -jbinney
    IplImage ipl_image = image;

    // This returns pose of camera w.r.t. AR Tag
    marker_detector.Detect(&ipl_image, &cam, true, false, max_new_marker_error, max_track_error, CVSEQ, true);

    /*for (size_t i=0; i<marker_detector.markers->size(); i++) {
        int id = (*(marker_detector.markers))[i].GetId();
        std::cout << id << std::endl;
        std::cout << cv::cvarrToMat((*(marker_detector.markers))[i].GetContent()) << std::endl;
        std::cout << "----------" << std::endl;
    }
    std::cout << "\n\n" << std::endl;*/

    for (size_t i=0; i<marker_detector.markers->size(); i++)
    {
        //Get the pose relative to the camera
        int id = (*(marker_detector.markers))[i].GetId();

        cv::Mat detectedTag = cv::cvarrToMat((*(marker_detector.markers))[i].GetContent());
        bool correctData = std::equal(detectedTag.begin<uchar>(), detectedTag.end<uchar>(), ar_tag.begin<uchar>());
        /*if (!correctData) {
            std::cout << detectedTag << std::endl;
            std::cout << "---------" << std::endl;
            continue;
        }*/

        Pose p = (*(marker_detector.markers))[i].pose;
        double px = p.translation[0]/100.0;
        double py = p.translation[1]/100.0;
        double pz = p.translation[2]/100.0;
        double qw = p.quaternion[0];
        double qx = p.quaternion[1];
        double qy = p.quaternion[2];
        double qz = p.quaternion[3];

        Eigen::Quaternionf rotation(qw, qx, qy, qz);
        Eigen::Translation3f trans(px, py, pz);
        // Alvar returns pose of AR Tag w.r.t. camera,
        Eigen::Affine3f pose = trans * rotation;

        //tf::Vector3 z_axis_cam = tf::Transform(rotation, tf::Vector3(0,0,0)) * tf::Vector3(0, 0, 1);
        //  ROS_INFO("%02i Z in cam frame: %f %f %f",id, z_axis_cam.x(), z_axis_cam.y(), z_axis_cam.z());
        Eigen::Vector3f z_axis_cam = rotation * Eigen::Vector3f(0., 0., 1.);
        /// as we can't see through markers, this one is false positive detection
        if (z_axis_cam(2) > 0)
        {
            continue;
        }

        // TODO: This assumes only one AR Tag in the frame
        *success = true;
        return pose;
    }

    *success = false;
    Eigen::Affine3f undef;
    return undef;
}
