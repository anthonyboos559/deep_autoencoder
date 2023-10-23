#include <Eigen/Dense>

class Activation {
public:
    Activation() {}
    virtual Eigen::VectorXd activate(const Eigen::VectorXd lyr) =0;
    virtual Eigen::VectorXd derivative(const Eigen::VectorXd lyr) =0;
};

class Sigmoid : public Activation {
public:
    Sigmoid() : Activation() {}
    Eigen::VectorXd activate(const Eigen::VectorXd &lyr) override { return lyr.unaryExpr(sigmoid); }
    Eigen::VectorXd derivative(const Eigen::VectorXd &lyr) override { return lyr.unaryExpr(sigmoid_d); }
    double sigmoid(double weighted_sum) { return 1 / (1 + exp(-weighted_sum)); }
    double sigmoid_d(double weighted_sum) { return exp(-weighted_sum) / pow((1 + exp(-weighted_sum)), 2); };
};

class Relu : public Activation {
public:
    Relu() : Activation() {}
    Eigen::VectorXd activate(const Eigen::VectorXd lyr) override { return lyr.unaryExpr(relu); }
    Eigen::VectorXd derivative(const Eigen::VectorXd lyr) override { return lyr.unaryExpr(relu_d); }
    double relu(double weighted_sum) { return weighted_sum > 0 ? weighted_sum : 0; }
    double relu_d(double weighted_sum) { return weighted_sum > 0 ? 1 : 0; };
};