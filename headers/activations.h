#include <Eigen/Dense>

double relu(double weighted_sum) { return weighted_sum > 0 ? weighted_sum : 0; }
double relu_d(double weighted_sum) { return weighted_sum > 0 ? 1 : 0; }
double sigmoid(double weighted_sum) { return 1 / (1 + exp(-weighted_sum)); }
double sigmoid_d(double weighted_sum) { return exp(-weighted_sum) / pow((1 + exp(-weighted_sum)), 2); }
