#include "layers.h"
#include "optimizers.h"
#include "loss_functions.h"

class Model {
protected:
    Optimizer optimizer;
    Loss_Function* loss_function;
    std::vector<Layer*> layers;

public:
    Model(Optimizer opt, Loss_Function* lf) : optimizer(opt), loss_function(lf) {}
};

class Sequential_model : public Model {
protected:

public:
    Sequential_model(Optimizer opt, Loss_Function* lf) : Model(opt, lf) {}
    void add_layer(Layer &lyr) { layers.push_back(&lyr); }
    void pop_layer() { layers.pop_back(); }
    //Main training method
    void train() {}
    void feedforward();
    void backpropagate();
    //Unimplemented atm
    void initalize_gradients();
};