#include "models.h"

void Sequential::initalize_gradients() {

}

void Sequential::fit(std::vector<Eigen::VectorXd> &train_data, std::vector<Eigen::VectorXd> &test_data) {
    for (int i = 0; i < epochs; i++) {
        std::random_shuffle(train_data.begin(), train_data.end());
        for (Eigen::VectorXd &data : train_data) {
            Eigen::VectorXd next_layer = data;
            for (Layer* lyr : layers) {
                next_layer = lyr->forwardprop(next_layer);
            }
            Eigen::VectorXd loss = loss_function->get_loss(data, next_layer);
            /*
            Messy, thinking it'd be better to separate the linear and activation layers rather than one 'layers' vector
            */
            for (int i = layers.size()-1; i >= 0; i -= 2) {
                layers.at(i)->set_error(loss);
                loss = layers.at(i)->backprop(layers.at(i-1)->get_layer());
                gradients.at(i) = layers.at(i-1)->get_weight_gradients(loss, layers.at(i-2)->get_layer());
                loss = layers.at(i-1)->backprop(loss);
            }
            //Optimizer call would come next
        }
    }
}