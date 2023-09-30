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

	train_data = new std::vector<Eigen::VectorXd>;
	test_data = new std::vector<Eigen::VectorXd>;
	layers = new std::vector<Eigen::VectorXd>(num_layers);
	weights = new std::vector<Eigen::MatrixXd>(num_layers-1);
	weight_changes = new std::vector<Eigen::MatrixXd>(num_layers-1);
	biases = new std::vector<Eigen::VectorXd>(num_layers-1);
	bias_changes = new std::vector<Eigen::VectorXd>(num_layers-1);
	z_values = new std::vector<Eigen::VectorXd>(num_layers-1);
	deltas = new std::vector<Eigen::VectorXd>(num_layers-1);

	train_data->reserve(60000);
	test_data->reserve(10000);

	load_train_data(train_file_path);
	load_test_data(test_file_path);
	for (int i = 0; i < num_layers-1; i++) {
		weights->at(i) = Eigen::MatrixXd::Random(sizes[i+1], sizes[i]);
		weight_changes->at(i) = Eigen::MatrixXd::Zero(sizes[i+1], sizes[i]);
		biases->at(i) = Eigen::VectorXd::Random(sizes[i+1]);
		bias_changes->at(i) = Eigen::VectorXd::Zero(sizes[i+1]);
	}
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
		Eigen::VectorXd image = Eigen::Map<Eigen::Vector<double, 784>>(image_data.data());
		train_data->push_back(Eigen::Map<Eigen::Vector<double, 784>>(image_data.data()).array() / 255);
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
		Eigen::VectorXd image = Eigen::Map<Eigen::Vector<double, 784>>(image_data.data());
		test_data->push_back(Eigen::Map<Eigen::Vector<double, 784>>(image_data.data()).array() / 255);
		count++;
		if (count % 1000 == 0) {
			printf("%i/%i training images loaded\n", count, test_data->capacity());
		}
	}
}

void Deep_autoencoder::feed_fordward(Eigen::VectorXd data) {
	layers->at(0) = data;
	for (int i = 0; i < num_layers-1; i++) {
		z_values->at(i) = (weights->at(i) * layers->at(i)) + biases->at(i);
		layers->at(i + 1) = z_values->at(i).unaryExpr(std::function(sigmoid));
	}
}

void Deep_autoencoder::backpropegate() {
	Eigen::VectorXd error = 2 * (layers->back() - layers->front());
	deltas->back() = z_values->back().unaryExpr(std::function(sigmoid_d)).array() * error.array();
	for (int i = num_layers - 2; i >= 0; i--) {
		weight_changes->at(i) += deltas->at(i) * layers->at(i).transpose();
		bias_changes->at(i) += deltas->at(i);
		if (i != 0) {
			deltas->at(i-1) = (weights->at(i).transpose() * deltas->at(i)).array() * z_values->at(i-1).unaryExpr(std::function(sigmoid_d)).array();
		}
	}
}

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
		feed_fordward(train_data->at(train_iter));
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
		feed_fordward(test_data->at(test_iter));
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
	std::vector<Eigen::VectorXd>* m_biases = new std::vector<Eigen::VectorXd>(num_layers-1);
	std::vector<Eigen::MatrixXd>* v_weights = new std::vector<Eigen::MatrixXd>(num_layers-1);
	std::vector<Eigen::VectorXd>* v_biases = new std::vector<Eigen::VectorXd>(num_layers-1);

	for (int i = 0; i < num_layers-1; i++) {
		m_weights->at(i) = Eigen::MatrixXd::Zero(layer_sizes[i+1], layer_sizes[i]);
		m_biases->at(i) = Eigen::VectorXd::Zero(layer_sizes[i+1]);
		v_weights->at(i) = Eigen::MatrixXd::Zero(layer_sizes[i+1], layer_sizes[i]);
		v_biases->at(i) = Eigen::VectorXd::Zero(layer_sizes[i+1]);
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

void Deep_autoencoder::print_out() {
	
	/*
	std::cout << "Layers:" << std::endl;
	for (int i = 0; i < num_layers; i++) {
		std::cout << "#" << i << std::endl;
		std::cout << layers->at(i) << std::endl;
	}
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
	test.adam();
	//test.print_out();
}
