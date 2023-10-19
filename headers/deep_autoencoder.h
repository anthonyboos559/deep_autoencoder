#include "layers.h"
#include <vector>
#include <string>


class Deep_autoencoder {

	std::vector<Eigen::VectorXd>* train_data;
	std::vector<Eigen::VectorXd>* test_data;
	std::vector<Layer> layers;

	std::vector<double> training_errors;
	std::vector<double> test_errors;

	Eigen::VectorXd* input;

	int io_size;
	int num_layers;
	std::vector<int> layer_sizes;

	double learning_rate;
	int epochs;
	int batch_size;
	double train_error;
	double test_error;

public:
	Deep_autoencoder(std::string train_file_path, std::string test_file_path, std::vector<int> sizes, double learn_rate, int epoch, int batch);
	void load_train_data(std::string file_path);
	void load_test_data(std::string file_path);
	Eigen::VectorXd feed_fordward(Eigen::VectorXd &data);
	void backpropegate(Eigen::VectorXd &error);
	void update_weights();
	double train_model(Eigen::VectorXd &data);
	void test_model();
	void mbgd();
	void adam();
	void print_out();
	void save_data();
	void write_image(int epoch, Eigen::VectorXd before, Eigen::VectorXd after);
};