#include <Eigen/Dense>

class Layer {
protected:
    Eigen::VectorXd layer;
    Layer* prev_layer;
    Layer* next_layer;
    Eigen::MatrixXd* prev_weights;
    Eigen::MatrixXd* next_weights;

public:
    template <typename Derived>
    Layer(const Eigen::DenseBase<Derived>& lyr) : layer(lyr) {}
    Eigen::VectorXd* get_layer() { return &layer; }
    Layer* get_prev_layer() { return prev_layer; }
    Layer* get_next_layer() { return next_layer; }
    void set_prev_Layer(Layer* p_lyr) { prev_layer = p_lyr; }
    void set_next_layer(Layer* n_lyr) { next_layer = n_lyr; }
    Eigen::MatrixXd* get_next_weights() { return next_weights; }
    Eigen::MatrixXd* get_prev_weights() { return prev_weights; }
    void set_next_weights(Eigen::MatrixXd* nxt_w) { next_weights = nxt_w; }
    void set_prev_weights(Eigen::MatrixXd* prv_w) { prev_weights = prv_w; }
    virtual void set_z(const Eigen::Ref<const Eigen::VectorXd>& vals) {};
    virtual Eigen::VectorXd* get_z() {};
    virtual void set_delta(const Eigen::Ref<const Eigen::VectorXd>& vals) {};
    virtual Eigen::VectorXd* get_delta() {};
};

class Input_layer: public Layer {
    const Layer* prev_layer;
    const Eigen::MatrixXd* prev_weights;

public:
    template <typename Derived>
    Input_layer(const Eigen::DenseBase<Derived>& lyr) : prev_layer(nullptr), prev_weights(nullptr), Layer(lyr) {}
};

class Hidden_layer: public Layer {
    Eigen::VectorXd z_values;
    Eigen::VectorXd deltas;
public:
    template <typename Derived>
    Hidden_layer(const Eigen::DenseBase<Derived>& lyr) : Layer(lyr) {}
    void set_z(const Eigen::Ref<const Eigen::VectorXd>& vals) { z_values = vals; };
    Eigen::VectorXd* get_z() { return &z_values; }
};

class Output_layer: public Hidden_layer {
    const Layer* next_layer;
    const Eigen::MatrixXd* next_weights;

public:
    template <typename Derived>
    Output_layer(const Eigen::DenseBase<Derived>& lyr) : next_layer(nullptr), next_weights(nullptr), Hidden_layer(lyr) {}
};