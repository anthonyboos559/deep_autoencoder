#include "activations.h"
#include <Eigen/Dense>

class Layer {
protected:
    Eigen::VectorXd layer;
    Eigen::VectorXd deltas;
    Eigen::MatrixXd weights;
    Eigen::MatrixXd weight_changes;

public:
    Layer() {}
    template <typename Derived>
    Layer(const Eigen::DenseBase<Derived> &lyr) : layer(lyr) {}
    Eigen::VectorXd get_layer() { return layer; }
    void set_deltas(const Eigen::VectorXd &lyr);
    template <typename Derived>
    void set_weights(const Eigen::DenseBase<Derived> &wghts) { weights = wghts; }
    Eigen::VectorXd forwardprop(const Eigen::VectorXd &lyr);
    template <typename Derived>
    Eigen::VectorXd backprop(const Eigen::DenseBase<Derived> &lyr);


};
/*
class Input_layer: public Layer {
protected:
    Layer* next_layer;
    Eigen::MatrixXd* next_weights;
public:
    template <typename Derived>
    Input_layer(const Eigen::DenseBase<Derived>& lyr) : Layer(lyr) {}
    Input_layer() : Layer() {}
    Layer* get_next_layer() { return next_layer; }
    void set_next_layer(Layer* n_lyr) { next_layer = n_lyr; }
    Eigen::MatrixXd* get_next_weights() { return next_weights; }
    void set_next_weights(Eigen::MatrixXd* nxt_w) { next_weights = nxt_w; }
};

class Output_layer: public Layer {
protected:
    Layer* prev_layer;
    Eigen::MatrixXd* prev_weights;
    Eigen::VectorXd z_values;
    Eigen::VectorXd deltas;
public:
    template <typename Derived>
    Output_layer(const Eigen::DenseBase<Derived>& lyr) : Layer(lyr) {}
    Output_layer() : Layer() {}
    Layer* get_prev_layer() { return prev_layer; }
    void set_prev_Layer(Layer* p_lyr) { prev_layer = p_lyr; }
    Eigen::MatrixXd* get_prev_weights() { return prev_weights; }
    void set_prev_weights(Eigen::MatrixXd* prv_w) { prev_weights = prv_w; }
    Eigen::VectorXd* get_z() { return &z_values; }
    void set_z(const Eigen::Ref<const Eigen::VectorXd>& vals) { z_values = vals; };
    Eigen::VectorXd* get_delta() { return &deltas; }
    void set_delta(const Eigen::Ref<const Eigen::VectorXd>& vals) { deltas = vals; };
};

class Hidden_layer: public Input_layer, public Output_layer {
public:
    Hidden_layer() : Input_layer() {}
    template <typename Derived>
    Hidden_layer(const Eigen::DenseBase<Derived>& lyr) : Input_layer(lyr) {}
};
*/