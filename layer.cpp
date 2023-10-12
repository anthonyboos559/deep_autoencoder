#include "layer.h"
#include <Eigen/Dense>

void Layer::set_deltas(const Eigen::VectorXd &lyr) {
    deltas = (layer.unaryExpr(sigmoid_d)).array() * lyr.array();
}

Eigen::VectorXd Layer::forwardprop(const Eigen::VectorXd &lyr) {
    layer = weights * lyr;
    return layer.unaryExpr(sigmoid);
}

template <typename Derived>
Eigen::VectorXd Layer::backprop(const Eigen::DenseBase<Derived> &lyr) {
    weight_changes += deltas * lyr.transpose();
    return weights.transpose() * deltas;
}