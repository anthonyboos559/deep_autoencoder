#include "layers.h"

namespace Optimizers {
    class SGD {
        double learning_rate;

    public:
        SGD(double lr) : learning_rate(lr) {}
        void optimize(std::vector<Layer*> &layers, std::vector<Eigen::MatrixXd> &gradients);
    };

    class ADAM {
        double learning_rate = 0.001;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilion = 1e-08;
        int time_step = 0;

        std::vector<Eigen::MatrixXd> moment1;
        std::vector<Eigen::MatrixXd> moment2;

    public:
        ADAM() {}
        void optimize(std::vector<Layer*> &layers, std::vector<Eigen::MatrixXd> &gradients);
        void initalize_moments(std::vector<Eigen::MatrixXd> &gradients);
    };
}