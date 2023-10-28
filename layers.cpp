#include "layers.h"

void Layer::set_layer_values(const Eigen::VectorXd &lyr) {
    size = lyr.rows();
    layer_values = Eigen::VectorXd::Ones(size+1);
    layer_values.head(size) = lyr;
}