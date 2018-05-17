#include <Eigen/Geometry>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

using namespace Eigen;
using namespace ceres;

class RigidCostFunctor {

public:
  RigidCostFunctor(const Matrix4f &arToCamInitial, const Vector3f &cam_image_corner_initial, const Vector3f &cam_image_corner): arToCamInitial_(arToCamInitial) {
      X_initial = cam_image_corner_initial[0];
      Y_initial = cam_image_corner_initial[1];
      Z_initial = cam_image_corner_initial[2];

      X_cur = cam_image_corner[0];
      Y_cur = cam_image_corner[1];
      Z_cur = cam_image_corner[2];
  }

  template <typename T>
  bool operator()(const T* const rt, T* residuals) const {
  //bool test(const T* const rt, T* residuals) const {

    T quaternion[4];
    quaternion[0] = rt[0];
    quaternion[1] = rt[1];
    quaternion[2] = rt[2];
    quaternion[3] = rt[3];

    T translation[3];
    translation[0] = rt[4];
    translation[1] = rt[5];
    translation[2] = rt[6];

    Eigen::Matrix<T, 3, 1> cam_image_corner;
    cam_image_corner << T(X_cur), T(Y_cur), T(Z_cur);

    Eigen::Matrix<T, 4, 4> arToCamInitial;
    arToCamInitial << T(arToCamInitial_(0, 0)), T(arToCamInitial_(0, 1)), T(arToCamInitial_(0, 2)), T(arToCamInitial_(0, 3)),
                  T(arToCamInitial_(1, 0)), T(arToCamInitial_(1, 1)), T(arToCamInitial_(1, 2)), T(arToCamInitial_(1, 3)),
                  T(arToCamInitial_(2, 0)), T(arToCamInitial_(2, 1)), T(arToCamInitial_(2, 2)), T(arToCamInitial_(2, 3)),
                  T(arToCamInitial_(3, 0)), T(arToCamInitial_(3, 1)), T(arToCamInitial_(3, 2)), T(arToCamInitial_(3, 3));

    Eigen::Quaternion<T> q = Map<const Eigen::Quaternion<T>>(quaternion);
    Eigen::Matrix<T, 3, 1> trans_vec = Map<const Eigen::Matrix<T, 3, 1>>(translation);

    Eigen::Matrix<T, 4, 1> res = (q * cam_image_corner + trans_vec).colwise().homogeneous();
    res = arToCamInitial * res;
    res(0) / res(3);
    res(1) / res(3);
    res(2) / res(3);
    res(3) / res(3);

    residuals[0] = res(0) - T(X_initial);
    residuals[1] = res(1) - T(Y_initial);
    residuals[2] = res(2) - T(Z_initial);

    return true;
  }

private:
  const Matrix4f &arToCamInitial_;
  float X_initial;
  float Y_initial;
  float Z_initial;
  float X_cur;
  float Y_cur;
  float Z_cur;

};
