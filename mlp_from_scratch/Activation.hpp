#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "Matrix.hpp"
#include <string>

enum class ActivationType {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX
};

class Activation {
public:
    static Matrix forward(const Matrix& Z, ActivationType type);
    static Matrix backward(const Matrix& Z, const Matrix& A, const Matrix& dA, ActivationType type);
    static std::string toString(ActivationType type);

private:
    static double sigmoid(double x);
    static double sigmoid_derivative(double a); // derivative in terms of output a = sigmoid(x)
    static double relu(double x);
    static double relu_derivative(double x);
    static double tanh_func(double x);
    static double tanh_derivative(double a); // derivative in terms of output a = tanh(x)

    static Matrix softmax(const Matrix& Z);
};

#endif // ACTIVATION_HPP
