#include "optimizers.h"
#include "loss_functions.h"
#include <fstream>
#include <iostream>

template <typename T, typename U>
class Model {
protected:
    //Templated to allow for varying optimizers and loss functions
    T optimizer;
    U loss_function;
    
    //Collections to store core model data
    std::vector<Layer*> layers;
    std::vector<Eigen::MatrixXd> gradients;
    std::vector<Eigen::VectorXd> train_data;
	std::vector<Eigen::VectorXd> test_data;
    
    int batch_size = 1; //Batch size controls how often the weights are updated
    int epochs = 1; //Train set iterations
    double train_error = 0;
    double test_error = 0;

public:
    Model(T opt, U lf) : optimizer(opt), loss_function(lf) { layers.push_back(new Input_Layer(0)); } //Ensures first layer is always an Input layer

    void set_batch_size(const int size) { batch_size = size; }
    void set_epochs(const int epoch) { epochs = epoch; }

    //Static data loading for now
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
            train_data.push_back(Eigen::Map<Eigen::Vector<double, 784>>(image_data.data()).array());
            count++;
            if (count % 1000 == 0) {
                printf("%i training images loaded\n", count);
            }
        }
    }
    //Static data loading for now
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
            test_data.push_back(Eigen::Map<Eigen::Vector<double, 784>>(image_data.data()).array());
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
    void add_layer(Layer* lyr) { this->layers.push_back(lyr); }
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
        Eigen::VectorXd output;
        for (int epoch = 1; epoch <= this->epochs; epoch++) {
            std::random_shuffle(this->train_data.begin(), this->train_data.end());
            for (int i = 0; i < this->train_data.size(); i++) {
                output = feedforward(this->train_data.at(i));
                this->train_error += this->loss_function.error(this->train_data.at(i), output);
                backpropagate(this->loss_function.derivative(this->train_data.at(i), output));
                if ((i+1) % this->batch_size == 0) {
                    average_gradients();
                    this->optimizer.optimize(this->layers, this->gradients);
                }
            }
            std::cout << "Epoch: " << epoch << '\n';
            std::cout << "Error: " << this->train_error / this->train_data.size() << std::endl;
            this->train_error = 0;
        }
    }

    void average_gradients() {
        for (auto gradient : this->gradients) {
            gradient /= this->batch_size;
        }
    }

    void initalize_weights() {
        int previous_size = this->train_data.front().rows();
        for (int i = 1; i < this->layers.size(); i++) {
            this->layers.at(i)->initalize_weights(previous_size);
            previous_size = this->layers.at(i)->get_size();
        }
    }

    void initalize_gradients() {
        for (int i = 1; i < this->layers.size(); i++) {
            Eigen::MatrixXd* weights = this->layers.at(i)->get_weights();
            this->gradients.push_back(Eigen::MatrixXd::Zero(weights->rows(), weights->cols()));
        }
    }

    void build() {
        initalize_weights();
        initalize_gradients();
    }
};