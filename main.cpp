#include "models.h"
#include <iostream>

int main()
{
	std::vector<int> sizes({64, 32, 10, 32, 64, 784});
    Loss_Functions::MSE loss_fun;
	Sequential_model test = Sequential_model(Optimizers::SGD(3), loss_fun);
	test.load_train_data("/home/tony/Documents/MNIST/mnist_train_no_label.csv");
    test.load_test_data("/home/tony/Documents/MNIST/mnist_test_no_label.csv");
    for (int size : sizes) {
        test.add_layer(new Sigmoid_Layer(size));
    }
    test.set_batch_size(100);
    test.set_epochs(30);
    test.build();
    //test.print_data();
    test.train();
	//test.adam();
	//test.print_out();
}