#include "deep_autoencoder.h"
#include "activations.h"
#include <iostream>
#include <fstream>

Deep_autoencoder::Deep_autoencoder(std::string train_file_path, std::string test_file_path, std::vector<int> sizes, double learn_rate, int epoch, int batch) {
	num_layers = sizes.size();
	layer_sizes = sizes;
	learning_rate = learn_rate;
	epochs = epoch;
	batch_size = batch;
	io_size = sizes.front();

	train_data = new std::vector<Eigen::VectorXd>;
	train_labels = new std::vector<int>;
	test_data = new std::vector<Eigen::VectorXd>;
	test_labels = new std::vector<int>;
	layers = new std::vector<Eigen::VectorXd>(num_layers);
	weights = new std::vector<Eigen::MatrixXd>(num_layers-1);
	weight_changes = new std::vector<Eigen::MatrixXd>(num_layers-1);
	biases = new std::vector<Eigen::VectorXd>(num_layers-1);
	bias_changes = new std::vector<Eigen::VectorXd>(num_layers-1);
	z_values = new std::vector<Eigen::VectorXd>(num_layers-1);
	deltas = new std::vector<Eigen::VectorXd>(num_layers-1);

	train_data->reserve(60000);
	train_labels->reserve(60000);
	test_data->reserve(10000);
	test_labels->reserve(10000);

	load_train_data(train_file_path);
	load_test_data(test_file_path);
	for (int i = 0; i < num_layers-1; i++) {
		layers->at(i+1) = (Eigen::VectorXd::Zero(sizes[i+1]));
		weights->at(i) = (Eigen::MatrixXd::Random(sizes[i+1], sizes[i]));
		weight_changes->at(i) = (Eigen::MatrixXd::Zero(sizes[i+1], sizes[i]));
		biases->at(i) = (Eigen::VectorXd::Random(sizes[i+1]));
		bias_changes->at(i) = (Eigen::VectorXd::Zero(sizes[i+1]));
		z_values->at(i) = (Eigen::VectorXd::Zero(sizes[i+1]));
		deltas->at(i) = (Eigen::VectorXd::Zero(sizes[i+1]));
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
		while (std::getline(row_stream, value, ',')) {
			image_data.push_back(stod(value));
		}
		train_labels->push_back(image_data.back());
		image_data.pop_back();
		Eigen::VectorXd image = Eigen::Map<Eigen::Vector<double, 784>>(image_data.data());
		train_data->push_back(Eigen::Map<Eigen::Vector<double, 784>>(image_data.data()).array() / 255);
		count++;
		if (count % 1000 == 0) {
			std::cout << count << "/60000 training images loaded" << std::endl;
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
		while (std::getline(row_stream, value, ',')) {
			image_data.push_back(stod(value));
		}
		test_labels->push_back(image_data.back());
		image_data.pop_back();
		Eigen::VectorXd image = Eigen::Map<Eigen::Vector<double, 784>>(image_data.data());
		test_data->push_back(Eigen::Map<Eigen::Vector<double, 784>>(image_data.data()).array() / 255);
		count++;
		if (count % 1000 == 0) {
			std::cout << count << "/10000 testing images loaded" << std::endl;
		}
	}
}

void Deep_autoencoder::feed_fordward(Eigen::VectorXd data) {
	layers->at(0) = data;
	for (int i = 0; i < num_layers-1; i++) {
		z_values->at(i) = (weights->at(i) * layers->at(i)) + biases->at(i);
		layers->at(i + 1) = z_values->at(i).unaryExpr(std::ptr_fun(sigmoid));
	}
}

void Deep_autoencoder::backpropegate() {
	Eigen::VectorXd error = layers->back() - layers->front();
	deltas->back() = z_values->back().unaryExpr(std::ptr_fun(sigmoid_d)).array() * error.array();
	for (int i = num_layers - 2; i >= 0; i--) {
		weight_changes->at(i) += deltas->at(i) * layers->at(i).transpose();
		bias_changes->at(i) += deltas->at(i);
		if (i != 0) {
			deltas->at(i-1) = (weights->at(i).transpose() * deltas->at(i)).array() * z_values->at(i-1).unaryExpr(std::ptr_fun(sigmoid_d)).array();
		}
	}
}

void Deep_autoencoder::update_weights() {
	for (int i = 0; i < num_layers - 1; i++) {
		//std::cout << "Before:\n" << weights->at(i) << std::endl;
		weights->at(i) -= weight_changes->at(i) * (learning_rate / batch_size);
		biases->at(i) -= bias_changes->at(i) * (learning_rate / batch_size);
		//std::cout << "After:\n" << weights->at(i) << std::endl;
		weight_changes->at(i).array() *= 0;
		bias_changes->at(i).array() *= 0;
	}
}

void Deep_autoencoder::train_model() {
	for (train_iter = 0; train_iter < train_data->size(); train_iter++) {
		feed_fordward(train_data->at(train_iter));
		backpropegate();
		if ((train_iter + 1) % batch_size == 0) {
			update_weights();
		}
	}
}

void Deep_autoencoder::test_model() {
	error = 0;
	for (test_iter = 0; test_iter < test_data->size(); test_iter++) {
		feed_fordward(test_data->at(test_iter));
		error += (layers->back() - layers->front()).array().square().sum();
	}
	error /= 10000;
}

void Deep_autoencoder::sgd() {
	for (int i = 1; i <= epochs; i++) {
		std::cout << "Epoch #" << i << std::endl;
		train_model();
		test_model();
		std::cout << "Mean squared error: " << error << std::endl;
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
	Deep_autoencoder test = Deep_autoencoder("C:\\Users\\Tony\\Desktop\\MNIST\\mnist_train.csv", "C:\\Users\\Tony\\Desktop\\MNIST\\mnist_test.csv", size, 3.0, 30, 100);
	test.sgd();
	//test.print_out();
}
