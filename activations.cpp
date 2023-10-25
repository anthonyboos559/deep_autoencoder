#include "activations.h"

Eigen::VectorXd Sigmoid::activate(const Eigen::VectorXd &lyr) { 
    Eigen::VectorXd activation = lyr;
    activation.head(lyr.rows()-1) = lyr.head(lyr.rows()-1).unaryExpr(sigmoid);
    return activation; 
}

Eigen::VectorXd Relu::activate(const Eigen::VectorXd &lyr) { 
    Eigen::VectorXd activation = lyr;
    activation.head(lyr.rows()-1) = lyr.head(lyr.rows()-1).unaryExpr(relu);
    return activation; 
}