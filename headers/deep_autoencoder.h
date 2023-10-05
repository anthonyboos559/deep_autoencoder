#include "layer.h"
#include <Eigen/Dense>
#include <vector>
#include <string>


class Deep_autoencoder {
	//Storing each component in a vector, the indexs will indicate what layer they correspond to
	std::vector<Input_layer>* train_data;
	std::vector<Input_layer>* test_data;
	std::vector<Eigen::MatrixXd>* weights;
	std::vector<Hidden_layer>* hidden_layers;

	Input_layer* input;
	Output_layer* output;

	int io_size;
	int num_layers;
	std::vector<int> layer_sizes;

	double learning_rate;
	int epochs;
	int batch_size;
	int train_iter;
	int test_iter;
	double train_error;
	double test_error;

public:
	Deep_autoencoder(std::string train_file_path, std::string test_file_path, std::vector<int> sizes, double learn_rate, int epoch, int batch);
	void load_train_data(std::string file_path);
	void load_test_data(std::string file_path);
	void feed_fordward(Input_layer* data);
	void backpropegate();
	void update_weights();
	void train_model();
	void test_model();
	void mbgd();
	void adam();
	void print_out();
};