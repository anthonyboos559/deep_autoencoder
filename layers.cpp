#include "layers.h"

Eigen::VectorXd Linear_Layer::forwardprop(const Eigen::VectorXd &lyr) {
    //lyr is the previous activation layer
    layer.head(size) = weights * lyr;
    return layer;
}

/* WIP/testing
Eigen::VectorXd Activation_Layer::forwardprop(const Eigen::VectorXd &lyr) {
    //lyr is the previous linear-layer
    layer.head(size) = lyr.unaryExpr(activation->primary); <- "a pointer to a bound function may only be used to call the function", WIP
    return layer;
}
*/

Eigen::VectorXd Sigmoid_Layer::forwardprop(const Eigen::VectorXd &lyr) {
    //lyr is the previous linear-layer
    layer.head(size) = lyr.unaryExpr(sigmoid);
    return layer;
}

Eigen::VectorXd Relu_Layer::forwardprop(const Eigen::VectorXd &lyr) {
    //lyr is the previous linear-layer
    layer.head(size) = lyr.unaryExpr(relu);
    return layer;
}

Eigen::VectorXd Linear_Layer::backprop(const Eigen::VectorXd &delta) {
    //Input is the output of an activation layer backprop call
    return weights.leftCols(weights.cols()-1).transpose() * delta;
}

Eigen::VectorXd Sigmoid_Layer::backprop(const Eigen::VectorXd &lyr) {
    //lyr is the linear layer associated with this activation layer
    //Error is the output of a linear layer backprop call
    return lyr.unaryExpr(sigmoid_d).array() * error.array();
}

Eigen::VectorXd Relu_Layer::backprop(const Eigen::VectorXd &lyr) {
    //lyr is the linear layer associated with this activation layer
    //Error is the output of a linear layer backprop call
    return lyr.unaryExpr(relu_d).array() * error.array();
}