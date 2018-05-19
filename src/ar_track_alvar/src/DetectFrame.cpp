#include "ceres/ceres.h"
#include "ar_track_alvar/DetectFrame.h"
#include "ar_track_alvar/MultiProductParameterization.h"
#include "ar_track_alvar/ARTagCostFunctor.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>

using namespace ceres;
using namespace alvar;
using namespace std;

vector<ARTag> DetectFrame::detectARTags(const cv::Mat &image) {
    // GetMultiMarkersPoses expects an IplImage*, but as of ros groovy, cv_bridge gives
    // us a cv::Mat. I'm too lazy to change to cv::Mat throughout right now, so I
    // do this conversion here -jbinney
    IplImage ipl_image = image;

    // This returns pose of camera w.r.t. AR Tag
    marker_detector.Detect(&ipl_image, &cam, true, false, max_new_marker_error, max_track_error, CVSEQ, true);

    vector<ARTag> ar_tags;

    for (size_t i=0; i<marker_detector.markers->size(); i++)
    {
        //Get the pose relative to the camera
        int id = (*(marker_detector.markers))[i].GetId();

        // If ID is not recognized, then skip
        if (find(marker_ids.begin(), marker_ids.end(), id) == marker_ids.end()) {
            std::cout << "==============" << std::endl;
            std::cout << "Wrong ID: " << id << std::endl;
            std::cout << "==============" << std::endl;
            continue;
        }

        const vector<PointDouble> &corns = (*(marker_detector.markers))[i].marker_corners_img;
        const vector<PointDouble> &corns_model = (*(marker_detector.markers))[i].marker_corners;

        vector<Eigen::Vector2f> image_corners;
        vector<Eigen::Vector3f> model_corners;
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

        ARTag tag;
        tag.id = id;
        tag.pose = pose;
        tag.image_corners = image_corners;
        tag.model_corners = model_corners;
        ar_tags.push_back(tag);
    }

    // TODO: CHange baCK!!!
    //if (ar_tags.size() > 1) {
    if (ar_tags.size() > 0) {
        if (leader_id < 0) {
            leader_id = ar_tags[0].id;
        }

        optimizeRT(ar_tags);
    }

    return ar_tags;
}

Eigen::Affine3f DetectFrame::optimizeRT(vector<ARTag> &ar_tags) {
    Eigen::Matrix3f intrinsic_matrix = (Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(&cam.calib_K_data[0][0])).cast<float>();

    int cur_leader = leader_id;
    // Does the leader exist? If not, change
    if (find_if(ar_tags.begin(), ar_tags.end(), [&](const ARTag &tag) { return tag.id == leader_id; }) == ar_tags.end()) {
        cur_leader = ar_tags.at(0).id;
    }

    Problem problem;

    Eigen::Affine3f leader_pose = find_if(ar_tags.begin(), ar_tags.end(), [cur_leader](const ARTag &tag) { return tag.id == cur_leader; })->pose;
    vector<double> se3datavec;
    se3datavec.resize(7 * ar_tags.size());
    double *se3data = se3datavec.data();
    vector<LocalParameterization*> local_parameterizations;
    int count = 0;
    for (auto it=ar_tags.begin(); it != ar_tags.end(); ++it, ++count) {
        ARTag &tag = *it;
        Eigen::Quaternionf rotation;
        Eigen::Vector3f translation;

        // Transform to left compose to the one being optimized
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();

        if (tag.id == cur_leader) {
            rotation = Eigen::Quaternionf(tag.pose.linear());
            translation = tag.pose.translation();
        } else if (leader_id == cur_leader) {
            // We want leader_pose * tfToLeader[tag.id];
            transform = leader_pose;
            rotation = Eigen::Quaternionf(tfToLeader[tag.id].linear());
            translation = tfToLeader[tag.id].translation();
        } else if (leader_id != cur_leader) {
            // We want leader_pose * tfToLeader[cur_leader].inverse() * tfToLeader[tag.id];
            transform = leader_pose * tfToLeader[cur_leader].inverse();
            rotation = Eigen::Quaternionf(tfToLeader[tag.id].linear());
            translation = tfToLeader[tag.id].translation();
        }

        double cur_se3[7] = {rotation.x(), rotation.y(), rotation.z(), rotation.w(), translation(0), translation(1), translation(2)};
        memcpy(&se3data[count*7], &cur_se3[0], 7*sizeof(double));

        for (int i = 0; i < 4; i++) {
          // Ownership is taken of the following, so no memory leak is happening
          Eigen::Vector3f world_corner = tag.model_corners[i];
          std::cout << "Real: " << tag.image_corners[i].transpose() << std::endl;
          Eigen::Vector3f res = intrinsic_matrix * (transform * (rotation * world_corner + translation));
          res /= res(2);
          std::cout << "Initial: " << res.transpose() << std::endl;
          DynamicAutoDiffCostFunction<ARTagCostFunctor> *cost =
              new DynamicAutoDiffCostFunction<ARTagCostFunctor>(new ARTagCostFunctor(
                  intrinsic_matrix, transform.matrix(), count*7, tag.image_corners[i],
                  tag.model_corners[i]));

          cost->AddParameterBlock(7 * ar_tags.size());
          cost->SetNumResiduals(2);
          problem.AddResidualBlock(cost, NULL, &se3data[0]);
        }

        local_parameterizations.push_back(new ProductParameterization(new EigenQuaternionParameterization(), new IdentityParameterization(3)));
    }

    // We are argmining w.r.t an SE(3) element
    MultiProductParameterization *se3_param = new MultiProductParameterization(local_parameterizations);
    problem.SetParameterization(&se3data[0], se3_param);

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.use_explicit_schur_complement = true;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.BriefReport() << endl;

    // Update rigid body motions based on optimized
    count = 0;
    for (auto it=ar_tags.begin(); it != ar_tags.end(); ++it, ++count) {
        ARTag &tag = *it;

        Eigen::Quaterniond rotation_opt(se3data[count*7 + 3], se3data[count*7 + 0], se3data[count*7 + 1], se3data[count*7 + 2]);
        Eigen::Translation3d translation_opt(se3data[count*7 + 4], se3data[count*7 + 5], se3data[count*7 + 6]);
        Eigen::Affine3f opt = (translation_opt * rotation_opt).cast<float>();

        if (tag.id == cur_leader) {
            tag.pose = opt;
        } else {
            tfToLeader[tag.id] = opt;
        }
    }

    // Set appropriate poses
    leader_pose = find_if(ar_tags.begin(), ar_tags.end(), [cur_leader](const ARTag &tag) { return tag.id == cur_leader; })->pose;
    count = 0;
    for (auto it=ar_tags.begin(); it != ar_tags.end(); ++it, ++count) {
        ARTag &tag = *it;

        if (tag.id == cur_leader) {
            // Do nothing
        } else if (leader_id == cur_leader) {
            // We want leader_pose * tfToLeader[tag.id];
            tag.pose = leader_pose * tfToLeader[tag.id];
        } else if (leader_id != cur_leader) {
            // We want leader_pose * tfToLeader[cur_leader].inverse() * tfToLeader[tag.id];
            tag.pose = leader_pose * tfToLeader[cur_leader].inverse() * tfToLeader[tag.id];
        }

        for (int i = 0; i < 4; i++) {
          Eigen::Vector3f world_corner = tag.model_corners[i];
          Eigen::Vector3f res = intrinsic_matrix * tag.pose * world_corner;
          res /= res(2);
          std::cout << "Final: " << res.transpose() << std::endl;
        }
    }
}
