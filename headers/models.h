#include "layers.h"
#include "optimizers.h"
#include "loss_functions.h"

class Model {
protected:
    Optimizer optimizer;
    Loss_Function* loss_function;
    int epochs;
    
    //Unimplemented atm
    std::vector<Eigen::MatrixXd> gradients;

public:
    Model(Optimizer opt, Loss_Function* lf, int epoch) : optimizer(opt), loss_function(lf), epochs(epoch) {}
};

class Sequential : public Model {
protected:
    std::vector<Layer*> layers;
    std::vector<Linear_Layer> linera_layers;
    std::vector<Activation_Layer> activation_layers;
public:
    Sequential(Optimizer opt, Loss_Function* lf, int epoch) : Model(opt, lf, epoch) {}
    void add_layer(Layer &lyr) { layers.push_back(&lyr); }
    void pop_layer() { layers.pop_back(); }
    //Main training method
    void fit(std::vector<Eigen::VectorXd> &train_data, std::vector<Eigen::VectorXd> &test_data) {}
    //Unimplemented atm
    void initalize_gradients();
};