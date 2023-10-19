
//Optimizers currently take weight gradients
class Optimizer {
protected:
    double learning_rate;
public:
    Optimizer(double lr) : learning_rate(lr) {}
    void optimize(std::vector<Eigen::VectorXd> &gradients);
};

class MBGD : public Optimizer {
protected:
    int batch_size;
public:
    MBGD(double lr, int bs) : batch_size(bs), Optimizer(lr) {}
    void optimize(std::vector<Eigen::MatrixXd> &gradients);
};

class ADAM : public Optimizer {

};