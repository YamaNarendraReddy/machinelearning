#include "Activation.hpp"
#include <cmath>
#include <algorithm>

double Activation::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double Activation::sigmoid_derivative(double a) {
    // a is already sigmoid(x)
    return a * (1.0 - a);
}

double Activation::relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double Activation::relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

double Activation::tanh_func(double x) {
    return std::tanh(x);
}

double Activation::tanh_derivative(double a) {
    // a is already tanh(x)
    return 1.0 - a * a;
}

Matrix Activation::softmax(const Matrix& Z) {
    int rows = Z.getRows();
    int cols = Z.getCols();
    Matrix A(rows, cols);

    for (int i = 0; i < rows; ++i) {
        // Find max element in the row for numerical stability
        double max_val = Z(i, 0);
        for (int j = 1; j < cols; ++j) {
            max_val = std::max(max_val, Z(i, j));
        }

        double sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            double exp_val = std::exp(Z(i, j) - max_val);
            A(i, j) = exp_val;
            sum += exp_val;
        }

        for (int j = 0; j < cols; ++j) {
            A(i, j) /= (sum + 1e-15);
        }
    }
    return A;
}

Matrix Activation::forward(const Matrix& Z, ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID:
            return Z.apply([](double x) { return sigmoid(x); });
        case ActivationType::RELU:
            return Z.apply([](double x) { return relu(x); });
        case ActivationType::TANH:
            return Z.apply([](double x) { return tanh_func(x); });
        case ActivationType::SOFTMAX:
            return softmax(Z);
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

Matrix Activation::backward(const Matrix& Z, const Matrix& A, const Matrix& dA, ActivationType type) {
    // Note: Z is the input to the activation function, A is the output of the activation function
    // dA is the derivative of the loss with respect to A.
    switch (type) {
        case ActivationType::SIGMOID:
            // dL/dZ = dL/dA * dA/dZ = dA * A * (1 - A)
            return dA.multiply(A.apply([](double a) { return sigmoid_derivative(a); }));
        case ActivationType::RELU:
            // dL/dZ = dL/dA * dA/dZ = dA * (Z > 0 ? 1 : 0)
            return dA.multiply(Z.apply([](double z) { return relu_derivative(z); }));
        case ActivationType::TANH:
            // dL/dZ = dL/dA * dA/dZ = dA * (1 - A^2)
            return dA.multiply(A.apply([](double a) { return tanh_derivative(a); }));
        case ActivationType::SOFTMAX:
            // Softmax derivative is complex when isolated, but usually we combine Softmax with Cross Entropy
            // in the loss layer. If we need to compute it independently:
            // dL/dZ_i = A_i * (dA_i - sum_k(dA_k * A_k))
            {
                int rows = Z.getRows();
                int cols = Z.getCols();
                Matrix dZ(rows, cols);
                for (int i = 0; i < rows; ++i) {
                    double dot_product = 0.0;
                    for (int j = 0; j < cols; ++j) {
                        dot_product += dA(i, j) * A(i, j);
                    }
                    for (int j = 0; j < cols; ++j) {
                        dZ(i, j) = A(i, j) * (dA(i, j) - dot_product);
                    }
                }
                return dZ;
            }
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

std::string Activation::toString(ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID: return "Sigmoid";
        case ActivationType::RELU: return "ReLU";
        case ActivationType::TANH: return "Tanh";
        case ActivationType::SOFTMAX: return "Softmax";
        default: return "Unknown";
    }
}
