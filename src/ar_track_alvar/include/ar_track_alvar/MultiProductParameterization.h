#ifndef MULTI_PRODUCT_PARAMETERIZATION_H_
#define MULTI_PRODUCT_PARAMETERIZATION_H_

#include <vector>
#include "ceres/local_parameterization.h"
#include "ceres/internal/disable_warnings.h"

namespace ceres {

// Construct a local parameterization by taking the Cartesian product
// of a number of other local parameterizations. This is useful, when
// a parameter block is the cartesian product of two or more
// manifolds. For example the parameters of a camera consist of a
// rotation and a translation, i.e., SO(3) x R^3.
//
// Currently this class supports taking the cartesian product of up to
// four local parameterizations.
//
// Example usage:
//
// ProductParameterization product_param(new QuaterionionParameterization(),
//                                       new IdentityParameterization(3));
//
// is the local parameterization for a rigid transformation, where the
// rotation is represented using a quaternion.
class CERES_EXPORT MultiProductParameterization : public LocalParameterization {
 public:
  //
  // NOTE: All the constructors take ownership of the input local
  // parameterizations.
  //
  MultiProductParameterization(LocalParameterization* local_param1,
                          LocalParameterization* local_param2);

  MultiProductParameterization(LocalParameterization* local_param1,
                          LocalParameterization* local_param2,
                          LocalParameterization* local_param3);

  MultiProductParameterization(LocalParameterization* local_param1,
                          LocalParameterization* local_param2,
                          LocalParameterization* local_param3,
                          LocalParameterization* local_param4);

  MultiProductParameterization(std::vector<LocalParameterization*> &params);

  virtual ~MultiProductParameterization();
  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const;
  virtual bool ComputeJacobian(const double* x,
                               double* jacobian) const;
  virtual int GlobalSize() const { return global_size_; }
  virtual int LocalSize() const { return local_size_; }

 private:
  void Init();

  std::vector<LocalParameterization*> local_params_;
  int local_size_;
  int global_size_;
  int buffer_size_;
};

} // namespace ceres

#include "ceres/internal/reenable_warnings.h"

#endif
