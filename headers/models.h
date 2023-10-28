#include "optimizers.h"
#include "loss_functions.h"
#include <fstream>
#include <iostream>

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
    Model(T opt, U lf) : optimizer(opt), loss_function(lf) { layers.push_back(new Input_Layer()); }

    void set_batch_size(const int size) { batch_size = size; }
    void set_epochs(const int epoch) { epochs = epoch; }

    void load_train_data(std::string file_path) {
        std::ifstream data_file (file_path);
        std::string row_data;
        std::string value;
        int count = 0;
        while (data_file.peek() != EOF) {
            std::getline(data_file, row_data);
            std::stringstream row_stream(row_data);
            std::vector<double> image_data;
            while (std::getline(row_stream, value, ',')) {
                image_data.push_back(stod(value));
            }
            train_data.push_back(Eigen::Map<Eigen::Vector<double, 784>>(image_data.data()).array() / 255);
            count++;
            if (count % 1000 == 0) {
                printf("%i training images loaded\n", count);
            }
        }
    }

    void load_test_data(std::string file_path) {
        std::ifstream data_file(file_path);
        std::string row_data;
        std::string value;
        int count = 0;
        while (data_file.peek() != EOF) {
            std::getline(data_file, row_data);
            std::stringstream row_stream(row_data);
            std::vector<double> image_data;
            while (std::getline(row_stream, value, ',')) {
                image_data.push_back(stod(value));
            }
            test_data.push_back(Eigen::Map<Eigen::Vector<double, 784>>(image_data.data()).array() / 255);
            count++;
            if (count % 1000 == 0) {
                printf("%i training images loaded\n", count);
            }
        }
    }
};

template <typename T, typename U>
class Sequential_model : public Model<T, U> {
protected:

public:
    Sequential_model(T opt, U lf) : Model<T, U>(opt, lf) {}
    template <typename L>
    void add_layer(L &lyr) { this->layers.push_back(&lyr); }
    void pop_layer() { this->layers.pop_back(); }

    Eigen::VectorXd feedforward(const Eigen::VectorXd &data) {
        this->layers.front()->set_layer_values(data);
        Eigen::VectorXd next_layer = data;
        for (Layer* lyr : this->layers) {
            next_layer = lyr->forwardprop(next_layer);
        }
        return next_layer.head(next_layer.rows()-1);
    }

    void backpropagate(Eigen::VectorXd error) {
        for (int i = this->layers.size()-1; i >= 1; i--) {
            error = this->layers.at(i)->backprop(error);
            this->gradients.at(i-1) += this->layers.at(i)->get_gradient() * this->layers.at(i-1)->get_activation().transpose();
        }
    }

    void train() {
        for (int epoch = 1; epoch <= this->epochs; epoch++) {
            std::random_shuffle(this->train_data.begin(), this->train_data.end());
            for (int i = 0; i < this->train_data.size(); i++) {
                Eigen::VectorXd output = feedforward(this->train_data.at(i));
                this->train_error += this->loss_function.error(this->train_data.at(i), output);
                backpropagate(this->loss_function.derivative(this->train_data.at(i), output));
                if (i+1 % this->batch_size == 0) {
                    this->optimizer.optimize(this->layers, this->gradients);
                }
            }
            std::cout << "Epoch: " << epoch << '\n';
            std::cout << "Error: " << this->train_error << std::endl;
            this->train_error = 0;
        }
    }

    void initalize_weights() {
        int previous_size = this->train_data.front().rows()+1;
        for (std::vector<Layer*>::iterator itr = this->layers.begin(); itr != this->layers.end(); ++itr) {
            (*itr)->initalize_weights(previous_size);
            previous_size = (*itr)->get_size();
        }
    }

    void initalize_gradients() {
        for (std::vector<Layer*>::iterator itr = this->layers.begin(); itr != this->layers.end(); ++itr) {
            Eigen::MatrixXd* weights = (*itr)->get_weights();
            this->gradients.push_back(Eigen::MatrixXd::Zero(weights->rows(), weights->cols()));
        }
    }

    void build() {
        initalize_weights();
        initalize_gradients();
    }
};