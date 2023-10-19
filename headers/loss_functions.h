#include <Eigen/Dense>

//Just MSE for now
class Loss_Function {

public:
    Loss_Function() {}
    virtual Eigen::VectorXd get_loss(const Eigen::VectorXd &input, const Eigen::VectorXd &output) =0;
    virtual double get_error(const Eigen::VectorXd &input, const Eigen::VectorXd &output) =0;
};

class MSE : public Loss_Function {

public:
    MSE() : Loss_Function() {}
    Eigen::VectorXd get_loss(const Eigen::VectorXd &input, const Eigen::VectorXd &output) { return 2 * (output - input); }
    double get_error(const Eigen::VectorXd &input, const Eigen::VectorXd &output) { return (output - input).array().square().sum(); }
};