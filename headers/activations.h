#include <cmath>

//Trying to make the activation functions an object, having trouble getting it to pass to a unaryexpr in the layer class

class Activation {
public:
    Activation() {}
    virtual double primary(double weighted_sum) =0;
    virtual double derivative(double weighted_sum) =0;
};

class Sigmoid : public Activation {
public:
    Sigmoid() : Activation() {}
    double primary(double weighted_sum) override { return 1 / (1 + exp(-weighted_sum)); }
    double derivative(double weighted_sum) override { return exp(-weighted_sum) / pow((1 + exp(-weighted_sum)), 2); };
};

class Relu : public Activation {
public:
    Relu() : Activation() {}
    double primary(double weighted_sum) override { return weighted_sum > 0 ? weighted_sum : 0; }
    double derivative(double weighted_sum) override { return weighted_sum > 0 ? 1 : 0; };
};