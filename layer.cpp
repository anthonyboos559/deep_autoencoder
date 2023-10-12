#include "layer.h"
#include "activations.h"

void Layer::set_deltas(const Eigen::VectorXd &lyr) {
    deltas = z_values.unaryExpr(std::ref(sigmoid_d)).array() * lyr.array();
    //deltas = deltas.array() * lyr.array();
}

Eigen::VectorXd Layer::forwardprop(const Eigen::VectorXd &lyr) {
    z_values = weights * lyr;
    layer.head(layer.rows()-1) = z_values.unaryExpr(std::ref(sigmoid));
    return layer;
}

Eigen::VectorXd Layer::backprop(const Eigen::VectorXd &lyr) {
    weight_changes += deltas * lyr.transpose();
    return weights.leftCols(weights.cols()-1).transpose() * deltas;
}