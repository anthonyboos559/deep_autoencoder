#include <Eigen/Dense>

inline double sigmoid(double weighted_sum) { return 1 / (1 + exp(-weighted_sum)); }
inline double sigmoid_d(double weighted_sum) { return exp(-weighted_sum) / pow((1 + exp(-weighted_sum)), 2); };
inline double relu(double weighted_sum) { return weighted_sum > 0 ? weighted_sum : 0; }
inline double relu_d(double weighted_sum) { return weighted_sum > 0 ? 1 : 0; };

class Activation {
public:
    Activation() {}
    virtual Eigen::VectorXd activate(const Eigen::VectorXd &lyr) =0;
    virtual Eigen::VectorXd derivative(const Eigen::VectorXd &lyr) =0;
};

class Sigmoid : public Activation {
public:
    Sigmoid() : Activation() {}
    Eigen::VectorXd activate(const Eigen::VectorXd &lyr) override;
    Eigen::VectorXd derivative(const Eigen::VectorXd &lyr) override { return lyr.unaryExpr(std::function(sigmoid_d)); }

};

class Relu : public Activation {
public:
    Relu() : Activation() {}
    Eigen::VectorXd activate(const Eigen::VectorXd &lyr) override;
    Eigen::VectorXd derivative(const Eigen::VectorXd &lyr) override { return lyr.unaryExpr(std::function(relu_d)); }
};

class Linear : public Activation {
public:
    Linear() : Activation() {}
    Eigen::VectorXd activate(const Eigen::VectorXd &lyr) override { return lyr; }
    Eigen::VectorXd derivative(const Eigen::VectorXd &lyr) override { return Eigen::VectorXd::Ones(lyr.rows()); }
};