#include <Eigen/Dense>
#include "activations.h"

class Layer {
protected:    
    int size;
    Eigen::VectorXd layer_values;
    Eigen::VectorXd layer_gradients;
    Eigen::MatrixXd weights;
    Activation* activation;

public:
    //Init all layers to ones, size is +1 to account for bias term - size is saved for block expressions later
    Layer(const int size, Activation* act = &Linear()) : layer_values(Eigen::VectorXd::Ones(size+1)), size(size), activation(act) {}
    Eigen::VectorXd get_layer() { return layer_values; }
    virtual Eigen::VectorXd forwardprop(const Eigen::VectorXd &prv_lyr);
    virtual Eigen::VectorXd backprop(const Eigen::VectorXd &error);
    virtual Eigen::MatrixXd get_weight_gradient(const Eigen::VectorXd &prv_lyr);
    
};

class Linear_Layer : public Layer {
protected:
    
public:
    Linear_Layer(const int size) : Layer(size) {}
    void update_weights (const Eigen::MatrixXd &wghts) { weights = wghts; }
    Eigen::VectorXd forwardprop(const Eigen::VectorXd &prv_lyr) override;
    Eigen::VectorXd backprop(const Eigen::VectorXd &error) override;
    double linear(const double weighted_sum) { return weighted_sum; }
    double linear_d(const double weighted_sum) { return 1; }
};

class Sigmoid_Layer : public Layer {
public:
    Sigmoid_Layer(const int size) : Layer(size, &Sigmoid()) {}
    Eigen::VectorXd forwardprop(const Eigen::VectorXd &prv_lyr) override;
    Eigen::VectorXd backprop(const Eigen::VectorXd &error) override;
    double sigmoid(const double weighted_sum) { return 1 / (1 + exp(-weighted_sum)); }
    double sigmoid_d(const double weighted_sum) { return exp(-weighted_sum) / pow((1 + exp(-weighted_sum)), 2); }
};

class Relu_Layer : public Layer {
public:
    Relu_Layer(const int size) : Layer(size, &Relu()) {}
    Eigen::VectorXd forwardprop(const Eigen::VectorXd &prv_lyr) override;
    Eigen::VectorXd backprop(const Eigen::VectorXd &error) override;
    double relu(const double weighted_sum) { return weighted_sum > 0 ? weighted_sum : 0; }
    double relu_d(const double weighted_sum) { return weighted_sum > 0 ? 1 : 0; }
};