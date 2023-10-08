#include "layer.h"

void Hidden_layer::set_z(Eigen::VectorXd* value) {
    z_values = *value;
}

Eigen::VectorXd* Hidden_layer::z_vals() {
    return &z_values;
}

void Hidden_layer::set_weights(Eigen::MatrixXd* prv_w, Eigen::MatrixXd* nxt_w) {
    prev_weights = prv_w;
    next_weights = nxt_w;
}

void Output_layer::set_z(Eigen::VectorXd* value) {
    z_values = *value;
}

Eigen::VectorXd* Output_layer::z_vals() {
    return &z_values;
}