#include <Eigen/Dense>

//Just MSE for now
namespace Loss_Functions {
    struct MSE {
        Eigen::VectorXd derivative(const Eigen::VectorXd &input, const Eigen::VectorXd &output) { return 2 * (output - input); }
        double error(const Eigen::VectorXd &input, const Eigen::VectorXd &output) { return (output - input).array().square().sum(); }
    };
}