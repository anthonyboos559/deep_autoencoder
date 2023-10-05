#include <Eigen/Dense>

class Hidden_layer {
    Eigen::VectorXd layer;
    Eigen::VectorXd z_values;
    Eigen::VectorXd deltas;
    Eigen::MatrixXd* prev_weights;
    Eigen::MatrixXd* next_weights;

public:
    Hidden_layer(Eigen::VectorXd* lyr);
    void set_weights(Eigen::MatrixXd* prv_w, Eigen::MatrixXd* nxt_w);
    Eigen::MatrixXd* weights();
};

class Input_layer {
    Eigen::VectorXd layer;
    Eigen::MatrixXd* next_weights;

public:
    Input_layer(Eigen::VectorXd* lyr);
    void set_weights(Eigen::MatrixXd* nxt_w);
    Eigen::MatrixXd* weights();
    Eigen::VectorXd* glayer();
};

class Output_layer {
    Eigen::VectorXd layer;
    Eigen::VectorXd z_values;
    Eigen::VectorXd deltas;
    Eigen::MatrixXd* prev_weights;

public:
    Output_layer(Eigen::VectorXd* lyr);
    void set_weights(Eigen::MatrixXd* prv_w);
    Eigen::MatrixXd* weights();
};