#include <Eigen/Geometry>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

using namespace Eigen;
using namespace ceres;

class ARTagCostFunctor {

public:
  ARTagCostFunctor(const Matrix3f &intrinsics, const Matrix4f &transform, int parameters_index, const Vector2f &image_corners, const Vector3f &world_corner): intrinsics_(intrinsics), transform_(transform), parameters_index_(parameters_index) {
      X_world = world_corner[0];
      Y_world = world_corner[1];
      Z_world = world_corner[2];

      x_image = image_corners[0];
      y_image = image_corners[1];
  }

  template <typename T>
  bool operator()(T const* const* rt, T* residuals) const {

    T quaternion[4];
    quaternion[0] = rt[0][parameters_index_ + 0];
    quaternion[1] = rt[0][parameters_index_ + 1];
    quaternion[2] = rt[0][parameters_index_ + 2];
    quaternion[3] = rt[0][parameters_index_ + 3];

    T translation[3];
    translation[0] = rt[0][parameters_index_ + 4];
    translation[1] = rt[0][parameters_index_ + 5];
    translation[2] = rt[0][parameters_index_ + 6];

    Eigen::Matrix<T, 3, 1> world_corner;
    world_corner << T(X_world), T(Y_world), T(Z_world);

    Eigen::Matrix<T, 3, 3> intrinsics;
    intrinsics << T(intrinsics_(0, 0)), T(intrinsics_(0, 1)), T(intrinsics_(0, 2)),
                  T(intrinsics_(1, 0)), T(intrinsics_(1, 1)), T(intrinsics_(1, 2)),
                  T(intrinsics_(2, 0)), T(intrinsics_(2, 1)), T(intrinsics_(2, 2));

    Eigen::Matrix<T, 4, 4> transformMat;
    transformMat << T(transform_(0, 0)), T(transform_(0, 1)), T(transform_(0, 2)), T(transform_(0, 3)),
                  T(transform_(1, 0)), T(transform_(1, 1)), T(transform_(1, 2)), T(transform_(1, 3)),
                  T(transform_(2, 0)), T(transform_(2, 1)), T(transform_(2, 2)), T(transform_(2, 3)),
                  T(transform_(3, 0)), T(transform_(3, 1)), T(transform_(3, 2)), T(transform_(3, 3));
    Eigen::Transform<T, 3, Eigen::Affine> transform(transformMat);

    Eigen::Quaternion<T> q = Map<const Eigen::Quaternion<T>>(quaternion);
    Eigen::Matrix<T, 3, 1> trans_vec = Map<const Eigen::Matrix<T, 3, 1>>(translation);

    Eigen::Matrix<T, 3, 1> res = intrinsics * (transform * (q * world_corner + trans_vec));
    res /= res(2);

    residuals[0] = res(0) - T(x_image);
    residuals[1] = res(1) - T(y_image);

    return true;
  }

private:
  const Matrix3f &intrinsics_;
  Matrix4f transform_;
  int parameters_index_;
  float X_world;
  float Y_world;
  float Z_world;
  float x_image;
  float y_image;

};
