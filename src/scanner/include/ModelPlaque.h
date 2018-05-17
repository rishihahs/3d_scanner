#ifndef MODEL_PLAQUE_H
#define MODEL_PLAQUE_H

#include <Eigen/Dense>

class ModelPlaque {
public:
    ModelPlaque(double width, double height);
    ~ModelPlaque();

    Eigen::MatrixXd getModelPC2D() const;
    Eigen::MatrixXd getModelPH2D() const;
    Eigen::MatrixXd getModelPC3D() const;
    Eigen::MatrixXd getModelPH3D() const;

    static Eigen::MatrixXd createModelPlaque(double width, double height);

protected:
    Eigen::MatrixXd _modelPC2D, _modelPH2D, _modelPC3D, _modelPH3D;

};

#endif
