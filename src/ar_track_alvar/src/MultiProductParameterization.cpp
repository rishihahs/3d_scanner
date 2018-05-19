#include "ar_track_alvar/MultiProductParameterization.h"

#include <algorithm>
#include "Eigen/Geometry"
#include "ceres/internal/eigen.h"
#include "ceres/internal/fixed_array.h"

namespace ceres {

using std::vector;

MultiProductParameterization::MultiProductParameterization(
    LocalParameterization* local_param1,
    LocalParameterization* local_param2) {
  local_params_.push_back(local_param1);
  local_params_.push_back(local_param2);
  Init();
}

MultiProductParameterization::MultiProductParameterization(
    LocalParameterization* local_param1,
    LocalParameterization* local_param2,
    LocalParameterization* local_param3) {
  local_params_.push_back(local_param1);
  local_params_.push_back(local_param2);
  local_params_.push_back(local_param3);
  Init();
}

MultiProductParameterization::MultiProductParameterization(
    LocalParameterization* local_param1,
    LocalParameterization* local_param2,
    LocalParameterization* local_param3,
    LocalParameterization* local_param4) {
  local_params_.push_back(local_param1);
  local_params_.push_back(local_param2);
  local_params_.push_back(local_param3);
  local_params_.push_back(local_param4);
  Init();
}

MultiProductParameterization::MultiProductParameterization(vector<LocalParameterization*> &params) {
  for (auto param : params) {
    local_params_.push_back(param);
  }
  Init();
}


MultiProductParameterization::~MultiProductParameterization() {
  for (int i = 0; i < local_params_.size(); ++i) {
    delete local_params_[i];
  }
}

void MultiProductParameterization::Init() {
  global_size_ = 0;
  local_size_ = 0;
  buffer_size_ = 0;
  for (int i = 0; i < local_params_.size(); ++i) {
    const LocalParameterization* param = local_params_[i];
    buffer_size_ = std::max(buffer_size_,
                            param->LocalSize() * param->GlobalSize());
    global_size_ += param->GlobalSize();
    local_size_ += param->LocalSize();
  }
}

bool MultiProductParameterization::Plus(const double* x,
                                   const double* delta,
                                   double* x_plus_delta) const {
  int x_cursor = 0;
  int delta_cursor = 0;
  for (int i = 0; i < local_params_.size(); ++i) {
    const LocalParameterization* param = local_params_[i];
    if (!param->Plus(x + x_cursor,
                     delta + delta_cursor,
                     x_plus_delta + x_cursor)) {
      return false;
    }
    delta_cursor += param->LocalSize();
    x_cursor += param->GlobalSize();
  }

  return true;
}

bool MultiProductParameterization::ComputeJacobian(const double* x,
                                              double* jacobian_ptr) const {
  MatrixRef jacobian(jacobian_ptr, GlobalSize(), LocalSize());
  jacobian.setZero();
  internal::FixedArray<double> buffer(buffer_size_);

  int x_cursor = 0;
  int delta_cursor = 0;
  for (int i = 0; i < local_params_.size(); ++i) {
    const LocalParameterization* param = local_params_[i];
    const int local_size = param->LocalSize();
    const int global_size = param->GlobalSize();

    if (!param->ComputeJacobian(x + x_cursor, buffer.get())) {
      return false;
    }
    jacobian.block(x_cursor, delta_cursor, global_size, local_size)
        = MatrixRef(buffer.get(), global_size, local_size);

    delta_cursor += local_size;
    x_cursor += global_size;
  }

  return true;
}

} // namespace ceres
