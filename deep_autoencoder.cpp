#include "deep_autoencoder.h"
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

	for (int i = 0; i < num_layers-1; i++) {
		layers.push_back(Layer(Eigen::VectorXd::Ones(layer_sizes[i+1]+1)));
		layers.at(i).set_weights(Eigen::MatrixXd::Random(layer_sizes[i+1], layer_sizes[i]+1));
		layers.at(i).set_weight_changes(Eigen::MatrixXd::Zero(layer_sizes[i+1], layer_sizes[i]+1));
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

void Deep_autoencoder::save_data() {
	std::ofstream datafile;
	datafile.open("/home/tony/programming/deep_autoencoder/model_data.csv");
	for (auto i : training_errors) {
		datafile << i << ",";
	}
	datafile << "\n";
	for (auto i : test_errors) {
		datafile << i << ",";
	}
	datafile.close();
}

void Deep_autoencoder::write_image(int epoch, Eigen::VectorXd before, Eigen::VectorXd after) {
	std::ofstream datafile;
	datafile.open("/home/tony/programming/deep_autoencoder/model_images.csv", std::ios_base::app);
	datafile << epoch;
	for (auto i : before) {
		datafile << "," << i;
	}
	datafile << "\n" << epoch;
	for (auto i : after) {
		datafile  << "," << i;
	}
	datafile << "\n";
	datafile.close();
}

Eigen::VectorXd Deep_autoencoder::feed_fordward(Eigen::VectorXd &data) {
	Eigen::VectorXd next_layer = data;
	for (Layer &lyr : layers) {
		next_layer = lyr.forwardprop(next_layer);
	}
	return layers.back().get_layer().head(io_size) - data.head(io_size);
}

void Deep_autoencoder::backpropegate(Eigen::VectorXd &error) {
	error *= 2;
	for (std::vector<Layer>::reverse_iterator lyr = layers.rbegin(); lyr != --layers.rend();) {
		lyr->set_deltas(error);
		error = lyr->backprop((++lyr)->get_layer());
	}
	layers.front().set_deltas(error);
	layers.front().backprop(*input);
}

void Deep_autoencoder::update_weights() {
	for (Layer &lyr : layers) {
		lyr.update_weights(learning_rate, batch_size);
	}
}

double Deep_autoencoder::train_model(Eigen::VectorXd &data) {
	input = &data;
	Eigen::VectorXd loss = feed_fordward(data);
	double squared_error = loss.array().square().sum();
	backpropegate(loss);
	return squared_error;
}

void Deep_autoencoder::test_model() {
	for (Eigen::VectorXd &data : *test_data) {
		Eigen::VectorXd loss = feed_fordward(data);
		test_error += loss.array().square().sum();
	}
}

void Deep_autoencoder::mbgd() {
	for (int i = 1; i <= epochs; i++) {
		train_error = 0; 
		test_error = 0;
		std::cout << "Epoch #" << i << std::endl;
		std::random_shuffle(train_data->begin(), train_data->end());
		for (int i = 0; i < train_data->size(); i++) {
			train_error += train_model(train_data->at(i));
			if ((i+1) % batch_size == 0) {
				update_weights();
			}
		}
		write_image(i, train_data->back().head(train_data->back().rows()-1), layers.back().get_layer().head(layers.back().get_layer().rows()-1));
		training_errors.push_back(train_error / train_data->size());
		std::cout << "Training error: " << training_errors.back() << std::endl;
		test_model();
		test_errors.push_back(test_error / test_data->size());
		std::cout << "Test error: " << test_errors.back() << std::endl;
	}
	//save_data();
}
/*
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

	Eigen::VectorXd test_vec = Eigen::VectorXd::Random(785).cwiseAbs();
	Eigen::VectorXd loss = feed_fordward(test_vec);
	std::cout << "Before:\n" << layers.at(1).get_weight_changes().transpose() << std::endl;
	backpropegate(loss);
	std::cout << "After:\n" <<  layers.at(1).get_weight_changes().transpose() << std::endl;
	
	
	/*
	std::cout << "Layers:" << std::endl;
	for (int i = 0; i < num_layers; i++) {
		std::cout << "#" << i << std::endl;
		std::cout << *(layers.at(i)->get_layer()) << std::endl;
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
	Deep_autoencoder test = Deep_autoencoder("/home/tony/Documents/MNIST/mnist_train_no_label.csv", "/home/tony/Documents/MNIST/mnist_test_no_label.csv", size, 0.1, 30, 100);
	test.mbgd();
	//test.adam();
	//test.print_out();
}
