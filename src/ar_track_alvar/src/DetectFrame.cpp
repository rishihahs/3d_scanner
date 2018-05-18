#include "ar_track_alvar/DetectFrame.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace alvar;
using namespace std;

Eigen::Affine3f DetectFrame::detectARTag(const cv::Mat &image, const cv::Mat &ar_tag, bool *success, vector<Eigen::Vector2f> &image_corners, vector<Eigen::Vector3f> &model_corners) {
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

        if (id != 10) {
            std::cout << "WOOOOOOOOOOOOW" << std::endl;
            std::cout << id << std::endl;
            exit(1);
        }

        cv::Mat detectedTag = cv::cvarrToMat((*(marker_detector.markers))[i].GetContent());
        bool correctData = std::equal(detectedTag.begin<uchar>(), detectedTag.end<uchar>(), ar_tag.begin<uchar>());
        /*if (!correctData) {
            std::cout << detectedTag << std::endl;
            std::cout << "---------" << std::endl;
            continue;
        }*/

        const vector<PointDouble> &corns = (*(marker_detector.markers))[i].marker_corners_img;
        const vector<PointDouble> &corns_model = (*(marker_detector.markers))[i].marker_corners;

        image_corners.clear();
        model_corners.clear();
        for (int i = 0; i < corns.size(); i++) {
            PointDouble img_corn = corns[i];
            PointDouble model_corn = corns_model[i];
            image_corners.push_back(Eigen::Vector2f(img_corn.x, img_corn.y));
            // We want in meters
            model_corners.push_back(Eigen::Vector3f(model_corn.x / 100., model_corn.y / 100., 0.));
        }

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


        /*CameraIntrinsics intrinsics(535.2900990271,
                                535.2900990271,
                                0.,
                                6.2082029864741003e+02,
                                4.6925264246286338e+02);*/

        // Alvar returns pose of AR Tag w.r.t. camera,
        Eigen::Affine3f pose = trans * rotation;
        /*Eigen::Matrix<float, 3, 3> rgb_intrinsics;
        rgb_intrinsics << 535.2900990271, 0.0000000000, 320.0000000000, 0, 535.2900990271, 240.0000000000,  0, 0, 1;
        Eigen::Vector3f imgpoint = (rgb_intrinsics * pose * a);
        Eigen::Vector3f imgpoint2 = (rgb_intrinsics * pose * b);
        Eigen::Vector3f imgpoint3 = (rgb_intrinsics * pose * c);
        imgpoint(0) /= imgpoint(2);
        imgpoint(1) /= imgpoint(2);
        imgpoint(2) /= imgpoint(2);
        imgpoint2(0) /= imgpoint2(2);
        imgpoint2(1) /= imgpoint2(2);
        imgpoint2(2) /= imgpoint2(2);
        imgpoint3(0) /= imgpoint3(2);
        imgpoint3(1) /= imgpoint3(2);
        imgpoint3(2) /= imgpoint3(2);

        std::cout << "CORNER: " << corners[0].x << " " << corners[0].y << std::endl;
        std::cout << "CORNER2: " << corners[1].x << " " << corners[1].y << std::endl;
        std::cout << "CORNER3: " << corners[2].x << " " << corners[2].y << std::endl;
        std::cout << "IMGPOINT: " << imgpoint.transpose() << std::endl;
        std::cout << "IMGPOINT2: " << imgpoint2.transpose() << std::endl;
        std::cout << "IMGPOINT3: " << imgpoint3.transpose() << std::endl;
        std::cout << "===========" << std::endl;*/

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
