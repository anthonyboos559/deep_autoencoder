#include "deep_autoencoder.h"
#include "activations.h"
#include <iostream>
#include <fstream>
#include <algorithm>

Deep_autoencoder::Deep_autoencoder(std::string train_file_path, std::string test_file_path, std::vector<int> sizes, double learn_rate, int epoch, int batch) {
	num_layers = sizes.size();
	layer_sizes = sizes;
	learning_rate = learn_rate;
	epochs = epoch;
	batch_size = batch;
	io_size = sizes.front();

	//Initalize the hidden layers
	layers = new std::vector<Layer*>;
	h_layers = new std::vector<Hidden_layer>;
	layers->reserve(num_layers);
	h_layers->reserve(num_layers-2);
	input = Input_layer(Eigen::VectorXd(io_size+1));
	layers->push_back(&input);
	Layer* prv_layer = layers->back(); //input
	for (int i = 1; i < num_layers-1; i++) { 
		//Extra input is given to the vector size to account for the bias term
		h_layers->push_back(Hidden_layer(Eigen::VectorXd::Ones(layer_sizes[i]+1)));
		layers->push_back(&h_layers->back());
		prv_layer->set_next_layer(layers->back());
		layers->back()->set_prev_Layer(prv_layer);
		prv_layer = layers->back();
	}
	output = Output_layer(Eigen::VectorXd(io_size+1));
	layers->push_back(&output);
	prv_layer->set_next_layer(&output);
	output.set_prev_Layer(prv_layer);

	//Initalize the weights at random values
	weights = new std::vector<Eigen::MatrixXd>;
	weights->reserve(num_layers-1);
	prv_layer = &input;
	for (int i = 0; i < num_layers-1; i++) {
		//The Matrix dimensions are NextLayer x (PrevLayer +1)
		//The weights are given an extra column to account for the bias vector
		weights->push_back(Eigen::MatrixXd::Random(sizes[i+1], sizes[i]+1));
		prv_layer->set_next_weights(&weights->back());
		prv_layer = prv_layer->get_next_layer();
		prv_layer->set_prev_weights(&weights->back());
	}

	train_data = new std::vector<Eigen::VectorXd>;
	test_data = new std::vector<Eigen::VectorXd>;

	train_data->reserve(60000);
	test_data->reserve(10000);

	load_train_data(train_file_path);
	load_test_data(test_file_path);

}

void Deep_autoencoder::load_train_data(std::string file_path) {
	std::ifstream data_file (file_path);
	std::string row_data;
	std::string value;
	int count = 0;
	while (data_file.peek() != EOF) {
		std::getline(data_file, row_data);
		std::stringstream row_stream(row_data);
		std::vector<double> image_data;
		image_data.reserve(io_size);
		while (std::getline(row_stream, value, ',')) {
			image_data.push_back(stod(value));
		}
		image_data.push_back(1.0);
		//Eigen::VectorXd image = (Eigen::Map<Eigen::Vector<double, 785>>(image_data.data())).array() / 255;
		train_data->push_back(Eigen::Map<Eigen::Vector<double, 785>>(image_data.data()).array() / 255);
		count++;
		if (count % 1000 == 0) {
			printf("%i/%i training images loaded\n", count, train_data->capacity());
		}
	}
}

void Deep_autoencoder::load_test_data(std::string file_path) {
	std::ifstream data_file(file_path);
	std::string row_data;
	std::string value;
	int count = 0;
	while (data_file.peek() != EOF) {
		std::getline(data_file, row_data);
		std::stringstream row_stream(row_data);
		std::vector<double> image_data;
		image_data.reserve(io_size);
		while (std::getline(row_stream, value, ',')) {
			image_data.push_back(stod(value));
		}
		image_data.push_back(255);
		//Eigen::VectorXd image = (Eigen::Map<Eigen::Vector<double, 785>>(image_data.data())).array() / 255;
		test_data->push_back(Eigen::Map<Eigen::Vector<double, 785>>(image_data.data()).array() / 255);
		count++;
		if (count % 1000 == 0) {
			printf("%i/%i training images loaded\n", count, test_data->capacity());
		}
	}
}

void Deep_autoencoder::feed_fordward(Eigen::VectorXd& data) {
	*input.get_layer() = data;
	for (auto lyr : *layers) {
		if (lyr != layers->front()) {
			lyr->set_z(*lyr->get_prev_weights() * *lyr->get_prev_layer()->get_layer());
			lyr->get_layer()->head(lyr->get_layer()->rows()-1) = lyr->get_z()->unaryExpr(std::ref(sigmoid));
		}
	}
}

void Deep_autoencoder::backpropegate() {
	Eigen::VectorXd error = 2 * (*output.get_layer() - *input.get_layer());
	output.set_delta(error);
	for (auto lyr : *layers) {
		if (lyr != layers->back()) {
			lyr->set_delta(lyr->get_z()->unaryExpr(std::ref(sigmoid_d)).array() * lyr->get_delta()->array());
		}
	}
	
	deltas->back() = z_values->back().unaryExpr(std::function(sigmoid_d)).array() * error.array();
	for (int i = num_layers - 2; i >= 0; i--) {
		weight_changes->at(i) += deltas->at(i) * layers->at(i).transpose();
		bias_changes->at(i) += deltas->at(i);
		if (i != 0) {
			deltas->at(i-1) = (weights->at(i).transpose() * deltas->at(i)).array() * z_values->at(i-1).unaryExpr(std::function(sigmoid_d)).array();
		}
	}
}
/*
void Deep_autoencoder::update_weights() {
	for (int i = 0; i < num_layers - 1; i++) {
		weights->at(i) -= weight_changes->at(i) * (learning_rate / batch_size);
		biases->at(i) -= bias_changes->at(i) * (learning_rate / batch_size);
		weight_changes->at(i).array() *= 0;
		bias_changes->at(i).array() *= 0;
	}
}

void Deep_autoencoder::train_model() {
	train_error = 0;
	for (train_iter = 0; train_iter < train_data->size(); train_iter++) {
		feed_fordward(train_data[train_iter]);
		train_error += (layers->back() - layers->front()).array().square().sum();
		backpropegate();
		if ((train_iter + 1) % batch_size == 0) {
			update_weights();
		}
	}
	train_error /= train_data->size();
}

void Deep_autoencoder::test_model() {
	test_error = 0;
	for (test_iter = 0; test_iter < test_data->size(); test_iter++) {
		feed_fordward(test_data[test_iter]);
		test_error += (layers->back() - layers->front()).array().square().sum();
	}
	test_error /= test_data->size();
}

void Deep_autoencoder::mbgd() {
	for (int i = 1; i <= epochs; i++) {
		std::cout << "Epoch #" << i << std::endl;
		train_model();
		printf("Training error: %f\n", train_error);
		test_model();
		printf("Test error: %f\n", test_error);
	}
}

void Deep_autoencoder::adam() {
	int time_step = 1;
	double alpha = 0.01;
	double beta1 = 0.9;
	double beta2 = 0.999;
	double epsilon = pow(10, -8);
	std::vector<Eigen::MatrixXd>* m_weights = new std::vector<Eigen::MatrixXd>(num_layers-1);
	std::vector<Eigen::MatrixXd>* v_weights = new std::vector<Eigen::MatrixXd>(num_layers-1);

	for (int i = 0; i < num_layers-1; i++) {
		m_weights->at(i) = Eigen::MatrixXd::Zero(layer_sizes[i+1], layer_sizes[i]);
		v_weights->at(i) = Eigen::MatrixXd::Zero(layer_sizes[i+1], layer_sizes[i]);
	}
	while (time_step < 10000) {
		std::random_shuffle(train_data->begin(), train_data->end());
		train_error = 0;
		for (train_iter = 0; train_iter < train_data->size(); train_iter++) {
			feed_fordward(train_data->at(train_iter));
			train_error += (layers->back() - layers->front()).array().square().sum();
			backpropegate();
			if ((train_iter + 1) % batch_size == 0) {
				printf("Timestep %i:\n", time_step);
				train_error /= batch_size;
				printf("Training error: %f\n", train_error);
				train_error = 0;
				test_model();
				printf("Test error: %f\n", test_error);
				for (int i = 0; i < num_layers-1; i++) {
					weight_changes->at(i) /= batch_size;
					bias_changes->at(i) /= batch_size;

					m_weights->at(i) = beta1 * m_weights->at(i) + (1-beta1) * weight_changes->at(i);
					m_biases->at(i) = beta1 * m_biases->at(i) + (1-beta1) * bias_changes->at(i);
					v_weights->at(i) = beta2 * v_weights->at(i).array() + (1-beta2) * weight_changes->at(i).array().square();
					v_biases->at(i) = beta2 * v_biases->at(i).array() + (1-beta2) * bias_changes->at(i).array().square();

					alpha *= sqrt(1-pow(beta2, time_step)) / (1-pow(beta1, time_step));

					v_weights->at(i) = v_weights->at(i).array().sqrt();
					v_biases->at(i) = v_biases->at(i).array().sqrt();

					weights->at(i).array() -= alpha * (m_weights->at(i).array() / (v_weights->at(i).array() + epsilon));
					biases->at(i).array() -= alpha * (m_biases->at(i).array() / (v_biases->at(i).array() + epsilon));

					weight_changes->at(i) *= 0;
					bias_changes->at(i) *= 0;
				}
				time_step++;
			}
		}

	}
}
*/
void Deep_autoencoder::print_out() {

	for (auto &lyr : *layers) {
		std::cout << typeid(&lyr).name() << std::endl;
	}

	
	Eigen::VectorXd test_vec = Eigen::VectorXd::Random(785);
	feed_fordward(test_vec);
	//std::cout << "Before:\n" << *input->get_layer() << std::endl;
	//std::cout << "After:\n" << sigmoid(*input->get_layer()) << std::endl;
	
	std::cout << "Layers:" << std::endl;
	for (int i = 0; i < num_layers; i++) {
		std::cout << "#" << i << std::endl;
		std::cout << *layers->at(i)->get_layer() << std::endl;
	}
	/*
	std::cout << "Weights:" << std::endl;
	for (int i = 0; i < num_layers-1; i++) {
		std::cout << "#" << i << std::endl;
		std::cout << weights->at(i) << std::endl;
	}
	*/
}

int main()
{
	std::vector<int> size({784, 48, 24, 10, 24, 48, 784});
	Deep_autoencoder test = Deep_autoencoder("/home/tony/Documents/MNIST/mnist_train_no_label.csv", "/home/tony/Documents/MNIST/mnist_test_no_label.csv", size, 3.0, 30, 100);
	//test.mbgd();
	//test.adam();
	test.print_out();
}
