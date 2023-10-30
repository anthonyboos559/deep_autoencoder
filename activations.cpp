#include "activations.h"
#include <iostream>

Eigen::VectorXd Sigmoid::activate(const Eigen::VectorXd &lyr) { 
    //std::cout << lyr << std::endl;
    Eigen::VectorXd activation = lyr;
    //std::cout << activation << std::endl;
    activation.head(lyr.rows()-1) = lyr.head(lyr.rows()-1).unaryExpr(std::function(sigmoid));
    //std::cout << activation << std::endl;
    return activation; 
}

Eigen::VectorXd Relu::activate(const Eigen::VectorXd &lyr) { 
    //std::cout << lyr.transpose() << std::endl;
    Eigen::VectorXd activation = lyr;
    //std::cout << activation.transpose() << std::endl;
    activation.head(lyr.rows()-1) = lyr.head(lyr.rows()-1).unaryExpr(std::function(relu));
    //std::cout << activation.transpose() << std::endl;
    return activation; 
}