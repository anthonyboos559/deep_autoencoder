#include "optimizers.h"

void Optimizers::ADAM::initalize_moments(std::vector<Eigen::MatrixXd> &gradients) {
    for (auto gradient : gradients) {
        moment1.push_back(Eigen::MatrixXd::Zero(gradient.rows(), gradient.cols()));
        moment2.push_back(Eigen::MatrixXd::Zero(gradient.rows(), gradient.cols()));
    }
}

void Optimizers::ADAM::optimize(std::vector<Layer*> &layers, std::vector<Eigen::MatrixXd> &gradients) {
    if (time_step == 0) {
        initalize_moments(gradients);
    }
    time_step++;
    for (int i = 0; i < gradients.size(); i++) {
        moment1.at(1) = (beta1 * moment1.at(i) + (1 - beta1) * gradients.at(i)) / (1 - pow(beta1, time_step));
        moment2.at(i) = (beta2 * moment2.at(i) + (1 - beta2) * (Eigen::MatrixXd)gradients.at(i).array().square()) / (1 - pow(beta2, time_step));
        layers.at(i+1)->update_weights(learning_rate * (moment1.at(i).array() / (moment2.at(i).array().sqrt() + epsilion).array()));
        gradients.at(i).setZero();
    }
}