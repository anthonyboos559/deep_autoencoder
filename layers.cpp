#include "layers.h"

Eigen::VectorXd Layer::forwardprop(const Eigen::VectorXd &prv_lyr) {
    //lyr is the previous layer
    layer_values.head(size) = weights * prv_lyr;
    return activation->activate(layer_values);
}

Eigen::VectorXd Linear_Layer::forwardprop(const Eigen::VectorXd &prv_lyr) {
    //lyr is the previous layer
    layer_values.head(size) = weights * prv_lyr;
    return layer_values;
}

Eigen::VectorXd Sigmoid_Layer::forwardprop(const Eigen::VectorXd &prv_lyr) {
    //lyr is the previous layer
    layer_values.head(size) = weights * prv_lyr;
    return layer_values.head(size).unaryExpr(sigmoid);
}

Eigen::VectorXd Relu_Layer::forwardprop(const Eigen::VectorXd &prv_lyr) {
    //lyr is the previous layer
    layer_values.head(size) = (weights * prv_lyr);
    return layer_values.head(size).unaryExpr(relu);
}

Eigen::VectorXd Layer::backprop(const Eigen::VectorXd &error) {
    layer_gradients = activation->derivative(layer_values.head(size)).array() * error.array();
    return weights.leftCols(weights.cols()-1).transpose() * layer_gradients;
}

Eigen::VectorXd Linear_Layer::backprop(const Eigen::VectorXd &error) {
    layer_gradients = error;
    return weights.leftCols(weights.cols()-1).transpose() * layer_gradients;
}

Eigen::VectorXd Sigmoid_Layer::backprop(const Eigen::VectorXd &error) {
    layer_gradients = layer_values.head(size).unaryExpr(sigmoid_d).array() * error.array();
    return weights.leftCols(weights.cols()-1).transpose() * layer_gradients;
}

Eigen::VectorXd Relu_Layer::backprop(const Eigen::VectorXd &error) {
    layer_gradients = layer_values.head(size).unaryExpr(relu_d).array() * error.array();
    return weights.leftCols(weights.cols()-1).transpose() * layer_gradients;
}

Eigen::MatrixXd Layer::get_weight_gradient(const Eigen::VectorXd &prv_lyr) {
    return 
}