#include <Eigen/Dense>

//Just MSE for now
namespace Loss_Functions {
    struct MSE {
        Eigen::VectorXd derivative(const Eigen::VectorXd &expected, const Eigen::VectorXd &output) { return 2 * (expected - output); }
        Eigen::VectorXd loss(const Eigen::VectorXd &expected, const Eigen::VectorXd &output) { return expected - output; }
        double error(const Eigen::VectorXd &expected, const Eigen::VectorXd &output) { return (expected - output).array().square().sum(); }
    };
}