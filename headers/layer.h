#include <Eigen/Dense>

class Hidden_layer {
    Eigen::VectorXd layer;
    Eigen::VectorXd z_values;
    Eigen::VectorXd deltas;
    Eigen::MatrixXd* prev_weights;
    Eigen::MatrixXd* next_weights;

public:
    Hidden_layer(Eigen::VectorXd* lyr);

};

class Input_layer {

public:
    Input_layer(Eigen::VectorXd* lyr);
};

class Output_layer {

public:
    Output_layer(Eigen::VectorXd* lyr);
};