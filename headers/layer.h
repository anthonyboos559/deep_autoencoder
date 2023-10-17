#include <Eigen/Dense>

class Layer {
protected:
    Eigen::VectorXd layer;
    Eigen::VectorXd z_values;
    Eigen::VectorXd deltas;
    Eigen::MatrixXd weights;
    Eigen::MatrixXd weight_changes;

public:
    //Layer() {}
    template <typename Derived>
    Layer(const Eigen::DenseBase<Derived> &lyr) : layer(lyr) {}
    Eigen::VectorXd get_layer() { return layer; }
    void set_deltas(const Eigen::VectorXd &lyr);
    template <typename Derived>
    void set_weights(const Eigen::DenseBase<Derived> &wghts) { weights = wghts; }
    template <typename Derived>
    void set_weight_changes(const Eigen::DenseBase<Derived> &wght_chngs) { weight_changes = wght_chngs; }
    Eigen::MatrixXd get_weight_changes() { return weight_changes; }
    Eigen::VectorXd forwardprop(const Eigen::VectorXd &lyr);
    Eigen::VectorXd backprop(const Eigen::VectorXd &lyr);
    void update_weights(const double learning_rate, const int batch_size) { weights -= weight_changes * (learning_rate/batch_size); weight_changes *= 0; }

};