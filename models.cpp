#include "models.h"

void Sequential::fit() {
    for (int i = 0; i < epochs; i++) {
        std::random_shuffle(train_data.begin(), train_data.end());
        for (Eigen::VectorXd &data : train_data) {
            Eigen::VectorXd next_layer = data;
            for (Layer* lyr : layers) {
                next_layer = lyr->forwardprop(next_layer);
            }
            Eigen::VectorXd loss = loss_function->get_loss(data, next_layer);
            
        }
    }
}