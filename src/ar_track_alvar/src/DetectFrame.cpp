#include "ar_track_alvar/DetectFrame.h"

using namespace alvar;
using namespace std;

Eigen::Affine3f DetectFrame::detectARTag(const cv::Mat &image, bool *success) {
    // GetMultiMarkersPoses expects an IplImage*, but as of ros groovy, cv_bridge gives
    // us a cv::Mat. I'm too lazy to change to cv::Mat throughout right now, so I
    // do this conversion here -jbinney
    IplImage ipl_image = image;

    // This returns pose of camera w.r.t. AR Tag
    marker_detector.Detect(&ipl_image, &cam, true, false, max_new_marker_error, max_track_error, CVSEQ, true);

    if (marker_detector.markers->size() <= 0) {
        *success = false;
        Eigen::Affine3f id;
        return id;
    }

    for (size_t i=0; i<marker_detector.markers->size(); i++)
    {
        //Get the pose relative to the camera
        int id = (*(marker_detector.markers))[i].GetId();
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
        // Alvar returns pose of camera w.r.t. AR Tag,
        // returning the inverse makes more sense as that
        // would be the obvious return
        Eigen::Affine3f pose = (trans * rotation).inverse();

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
}
