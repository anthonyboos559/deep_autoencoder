#include <Eigen/Dense>
#include "activations.h"

class Layer {
protected:
    Eigen::VectorXd layer;
    int size;

public:
    //Init all layers to ones, size is +1 to account for bias term - size is saved for block expressions later
    Layer(const int size) : layer(Eigen::VectorXd::Ones(size+1)), size(size) {}
    Eigen::VectorXd get_layer() { return layer; }
    virtual Eigen::MatrixXd get_weight_gradients(const Eigen::VectorXd &delta, const Eigen::VectorXd &prv_lyr);
    virtual Eigen::VectorXd forwardprop(const Eigen::VectorXd &lyr) =0;
    virtual Eigen::VectorXd backprop(const Eigen::VectorXd &lyr) =0;
    virtual void set_error(const Eigen::VectorXd &lyr);
    virtual Eigen::VectorXd get_error();
};

class Linear_Layer : public Layer {
protected:
    Eigen::MatrixXd weights;
public:
    Linear_Layer(const int size) : Layer(size) {}
    //Init weights, thinking it'd be better to init with the layer
    void build(const int shape) { weights = Eigen::MatrixXd::Random(size, shape+1).cwiseAbs(); }
    void update_weights (const Eigen::MatrixXd &wghts) { weights = wghts; }
    //Return the deratives/gradeints of the weights rather than saving them in the layer like before, currently saved under the Model classes
    Eigen::MatrixXd get_weight_gradients(const Eigen::VectorXd &delta, const Eigen::VectorXd &prv_lyr) override;
    Eigen::VectorXd forwardprop(const Eigen::VectorXd &lyr) override;
    Eigen::VectorXd backprop(const Eigen::VectorXd &delta) override;
};

class Activation_Layer : public Layer {
protected:
    Eigen::VectorXd error;
    Activation* activation;
public:
    Activation_Layer(const int size, Activation &actv) : activation(&actv), Layer(size) {}
    //Activation_Layer(const int size) : Layer(size) {}
    void set_error(const Eigen::VectorXd &lyr) override { error = lyr; }
    Eigen::VectorXd get_error() override { return error; }
    Eigen::VectorXd forwardprop(const Eigen::VectorXd &lyr) override;
    Eigen::VectorXd backprop(const Eigen::VectorXd &delta) override;
};

class Sigmoid_Layer : public Activation_Layer {
public:
    Sigmoid_Layer(const int size) : Activation_Layer(size, Sigmoid()) {}
    Eigen::VectorXd forwardprop(const Eigen::VectorXd &lyr) override;
    Eigen::VectorXd backprop(const Eigen::VectorXd &lyr) override;
    double sigmoid(double weighted_sum) { return 1 / (1 + exp(-weighted_sum)); }
    double sigmoid_d(double weighted_sum) { return exp(-weighted_sum) / pow((1 + exp(-weighted_sum)), 2); }
};

class Relu_Layer : public Activation_Layer {
public:
    Relu_Layer(const int size) : Activation_Layer(size, Relu()) {}
    Eigen::VectorXd forwardprop(const Eigen::VectorXd &lyr) override;
    Eigen::VectorXd backprop(const Eigen::VectorXd &lyr) override;
    double relu(double weighted_sum) { return weighted_sum > 0 ? weighted_sum : 0; }
    double relu_d(double weighted_sum) { return weighted_sum > 0 ? 1 : 0; }
};