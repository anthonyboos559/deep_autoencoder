#include "activations.h"
#include <iostream>

class Layer {
protected:
    int size;
    Eigen::VectorXd layer_values;
    Eigen::VectorXd layer_gradients;
    Eigen::MatrixXd weights;
    
public:
    //Init all layers to ones, size is +1 to account for bias term - size is saved for block expressions later
    Layer() {}
    Layer(const int size) : layer_values(Eigen::VectorXd::Ones(size+1)), size(size) {}
    
    void set_layer_values(const Eigen::VectorXd &lyr);
    void initalize_weights(const int col_size) { weights = Eigen::MatrixXd::Random(size, col_size+1); }
    void update_weights(const Eigen::MatrixXd changes) { weights -= changes; }
    
    Eigen::VectorXd get_layer() { return layer_values; }
    int get_size() { return size; }
    Eigen::VectorXd get_gradient() { return layer_gradients; }
    Eigen::MatrixXd* get_weights() { return &weights; }


    virtual Eigen::VectorXd forwardprop(const Eigen::VectorXd &prv_lyr) =0;
    virtual Eigen::VectorXd backprop(const Eigen::VectorXd &error) =0;
    virtual Eigen::VectorXd get_activation() =0;
    
};

template <typename T>
class Activation_Layer : public Layer {
protected:
    T activation;

public:
    Activation_Layer(T act) : Layer() {}
    Activation_Layer(const int size, T act ) : activation(act), Layer(size) {}
    Eigen::VectorXd get_activation() { return activation.activate(layer_values); }
    Eigen::VectorXd forwardprop(const Eigen::VectorXd &prv_lyr) override {
        //std::cout << prv_lyr << '\n' << weights << std::endl;
        layer_values.head(size) = weights * prv_lyr;
        return get_activation();
    }
    Eigen::VectorXd backprop(const Eigen::VectorXd &error) override {
        layer_gradients = activation.derivative(layer_values.head(size)).array() * error.array();
        return weights.leftCols(weights.cols()-1).transpose() * layer_gradients;
    }
    
};

class Linear_Layer : public Activation_Layer<Linear> {
public:
    Linear_Layer() : Activation_Layer(Linear()) {}
    Linear_Layer(const int size) : Activation_Layer(size, Linear()) {}
};

class Input_Layer : public Linear_Layer {
public:
    Input_Layer() : Linear_Layer() {}
    Eigen::VectorXd forwardprop(const Eigen::VectorXd &prv_lyr) override { return layer_values; }
};

class Sigmoid_Layer : public Activation_Layer<Sigmoid> {
public:
    Sigmoid_Layer(const int size) : Activation_Layer(size, Sigmoid()) {}
};

class Relu_Layer : public Activation_Layer<Relu> {
public:
    Relu_Layer(const int size) : Activation_Layer(size, Relu()) {}
};