#include <Eigen/Dense>

class Layer {

    Eigen::VectorXd* const layer;
    Eigen::VectorXd* const z_values;
    Eigen::VectorXd* const deltas;
    Eigen::MatrixXd* const prev_weights;
    Eigen::MatrixXd* const next_weights;

public:
    Layer(Eigen::VectorXd* lyr, Eigen::MatrixXd* nxt_w = nullptr, Eigen::MatrixXd* prv_w = nullptr);


};