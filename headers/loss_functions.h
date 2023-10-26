#include <Eigen/Dense>

//Just MSE for now
namespace Loss_Functions {
    struct MSE {
        Eigen::VectorXd derivative(const Eigen::VectorXd &expected, const Eigen::VectorXd &output) { return 2 * (output - expected); }
        Eigen::VectorXd loss(const Eigen::VectorXd &expected, const Eigen::VectorXd &output) { return output - expected; }
        double error(const Eigen::VectorXd &expected, const Eigen::VectorXd &output) { return (output - expected).array().square().sum(); }
    };
}