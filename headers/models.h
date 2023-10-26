#include "layers.h"
#include "optimizers.h"
#include "loss_functions.h"

template <typename T, typename U>
class Model {
protected:
    T optimizer;
    U loss_function;
    std::vector<Layer*> layers;
    std::vector<Eigen::MatrixXd> gradients;
    std::vector<Eigen::VectorXd> train_data;
	std::vector<Eigen::VectorXd> test_data;
    int batch_size = 0;
    int epochs = 0;
    double train_error = 0;
    double test_error = 0;

public:
    Model(T opt, U lf) : optimizer(opt), loss_function(lf) {}
};

template <typename T, typename U>
class Sequential_model : public Model<T, U> {
protected:

public:
    Sequential_model(T opt, U lf) : Model(opt, lf) {}
    void add_layer(Layer &lyr) { layers.push_back(lyr); }
    void pop_layer() { layers.pop_back(); }

    Eigen::VectorXd feedforward(const Eigen::VectorXd &data) {
        layers.front()->set_layer_values(data);
        Eigen::VectorXd next_layer = data;
        for (Layer lyr : layers) {
            next_layer = lyr.forwardprop(next_layer);
        }
        return next_layer;
    }

    void backpropagate(Eigen::VectorXd error) {
        for (int i = layers.size(); i >= 1; i--) {
            error = layers.at(i)->backprop(error);
            *gradients.at(i-1) += layers.at(i-1)->activation().transpose() * layers.at(i)->get_gradient();
        }
    }

    void train() {
        for (int epoch = 1; epoch <= epochs; epoch++) {
            std::random_shuffle(train_data.begin(), train_data.end());
            for (int i = 0; i < train_data.size(); i++) {
                Eigen::VectorXd output = feedforward(train_data.at(i));
                train_error += loss_function.error(train_data.at(i), output);
                backpropagate(loss_function.derivative(train_data.at(i), output))
                if (i+1 % batch_size == 0) {
                    optimizer.optimize(layers, gradients);
                }
            }
        }
    }
    //Unimplemented atm
    void initalize_gradients();
};