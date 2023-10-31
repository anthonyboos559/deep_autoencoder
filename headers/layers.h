#include "activations.h"
#include <iostream>

class Layer {
protected:
    int size;
    Eigen::VectorXd layer_values; //Unactivated values, last entry is always 1 to account for the bias
    Eigen::VectorXd layer_gradients; //Gradient of the layer with respect to the loss function
    Eigen::MatrixXd weights;
    
public:
    //Init all layers to ones, size is +1 to account for bias term - size is saved for block expressions later
    Layer(const int size) : size(size), layer_values(Eigen::VectorXd::Ones(size+1)) {}

    //Used for updating the Input layer with input data for the forward pass
    void set_layer_values(const Eigen::VectorXd &lyr); 
    
    //Getters
    int get_size() { return size; }
    Eigen::VectorXd get_layer() { return layer_values; }
    Eigen::VectorXd get_gradient() { return layer_gradients; }
    Eigen::MatrixXd* get_weights() { return &weights; } //Returns ptr since its used only to get info about the weights (Rows/Cols)

    //Input is the size of the previous/input layer
    void initalize_weights(const int col_size) { weights = Eigen::MatrixXd::Random(size, col_size+1); } 

    //Input comes from the optimizer
    void update_weights(const Eigen::MatrixXd changes) { weights -= changes; }

    virtual Eigen::VectorXd forwardprop(const Eigen::VectorXd &prv_lyr) =0;
    virtual Eigen::VectorXd backprop(const Eigen::VectorXd &error) =0;
    virtual Eigen::VectorXd get_activation() =0;
    
};

class Input_Layer : public Layer {
public:
    Input_Layer(const int size) : Layer(size) {}
    Eigen::VectorXd get_activation() override { return layer_values; }
    Eigen::VectorXd forwardprop(const Eigen::VectorXd &prv_lyr) override { return layer_values; }
    Eigen::VectorXd backprop(const Eigen::VectorXd &error) override {}
};

template <typename T>
class Activation_Layer : public Layer {
protected:
    T activation;

public:
    Activation_Layer(const int size, T act ) : Layer(size), activation(act) {}
    Eigen::VectorXd get_activation() override { return activation.activate(layer_values); }
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
    Linear_Layer(const int size) : Activation_Layer(size, Linear()) {}
};

class Sigmoid_Layer : public Activation_Layer<Sigmoid> {
public:
    Sigmoid_Layer(const int size) : Activation_Layer(size, Sigmoid()) {}
};

class Relu_Layer : public Activation_Layer<Relu> {
public:
    Relu_Layer(const int size) : Activation_Layer(size, Relu()) {}
};