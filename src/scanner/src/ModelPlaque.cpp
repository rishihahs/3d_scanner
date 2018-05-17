#include "ModelPlaque.h"
#include "mapping_calibration/HomCartShortcuts.h"

using namespace Eigen;

ModelPlaque::ModelPlaque(double width, double height)
    : _modelPC2D(createModelPlaque(width, height)),
      _modelPH2D(addARowOfConst(_modelPC2D, 1.0)),
      _modelPC3D(addARowOfConst(_modelPC2D, 0.0)),
      _modelPH3D(addARowOfConst(_modelPC3D, 1.0)) {}

ModelPlaque::~ModelPlaque() {}

MatrixXd ModelPlaque::getModelPC2D() const { return _modelPC2D; }
MatrixXd ModelPlaque::getModelPH2D() const { return _modelPH2D; }
MatrixXd ModelPlaque::getModelPC3D() const { return _modelPC3D; }
MatrixXd ModelPlaque::getModelPH3D() const { return _modelPH3D; }

MatrixXd ModelPlaque::createModelPlaque(double width, double height) {
  MatrixXd m(2, 4);  // 4 corners

  m(0, 0) = 0.;
  m(1, 0) = 0.;

  m(0, 1) = width;
  m(1, 1) = 0.;

  m(0, 2) = width;
  m(1, 2) = height;

  m(0, 3) = 0.;
  m(1, 3) = height;

  return m;
}
