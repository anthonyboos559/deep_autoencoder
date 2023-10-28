#include "models.h"
#include <iostream>

int main()
{
	std::vector<int> sizes({256, 128, 64, 128, 256, 784});
    Loss_Functions::MSE loss_fun;
	Sequential_model test = Sequential_model(Optimizers::ADAM(), loss_fun);
	test.load_train_data("/home/tony/Documents/MNIST/mnist_train_no_label.csv");
    test.load_test_data("/home/tony/Documents/MNIST/mnist_test_no_label.csv");
    for (int size : sizes) {
        Sigmoid_Layer lyr = Sigmoid_Layer(size);
        test.add_layer(lyr);
    }
    test.set_batch_size(100);
    test.set_epochs(30);
    test.build();
    test.train();
	//test.adam();
	//test.print_out();
}