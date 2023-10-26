#include "models.h"
#include <algorithm>

template <typename T, typename U>
Eigen::VectorXd Sequential_model<T, U>::feedforward(const Eigen::VectorXd &data) {
    layers.front()->set_layer_values(data);
    Eigen::VectorXd next_layer = data;
    for (Layer lyr : layers) {
        next_layer = lyr.forwardprop(next_layer);
    }
    return next_layer;
}

template <typename T, typename U>
void Sequential_model<T, U>::backpropagate(Eigen::VectorXd error) {
    for (int i = layers.size(); i >= 1; i--) {
        error = layers.at(i)->backprop(error);
        *gradients.at(i-1) += layers.at(i-1)->activation().transpose() * layers.at(i)->get_gradient();
    }
}

template <typename T, typename U>
void Sequential_model<T, U>::train() {
    Eigen::VectorXd output;
    while (!optimized) {
        std::random_shuffle(train_data.begin(), train_data.end());
        for (auto data : train_data) {
            output = feedforward(data);
            train_error += loss_function.error(data, output);
            backpropagate(loss_function.derivative(data, output))
            optimizer.optimize(layers, gradients);
        }
    }
}