#include "models.h"

int main()
{
	std::vector<int> sizes({256, 128, 64, 128, 256, 784});
    Loss_Functions::MSE loss_fun;
	Sequential_model test = Sequential_model(Optimizers::SGD(0.1), loss_fun);
	test.load_train_data("/home/tony/programming/data/MNIST/mnist_train_no_label.csv");
    test.load_test_data("/home/tony/programming/data/MNIST/mnist_test_no_label.csv");
    for (int size : sizes) {
        test.add_layer(new Relu_Layer(size));
    }
    test.set_batch_size(100);
    test.set_epochs(30);
    test.build();
    test.train();
}