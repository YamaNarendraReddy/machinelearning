#ifndef LOSS_HPP
#define LOSS_HPP

#include "Matrix.hpp"
#include <string>

enum class LossType {
    MSE,          // Mean Squared Error (Regression)
    CROSS_ENTROPY, // Categorical Cross Entropy (Multi-class Classification)
    BINARY_CROSS_ENTROPY // Binary Cross Entropy (Binary Classification)
};

class Loss {
public:
    static double compute(const Matrix& Y_pred, const Matrix& Y_true, LossType type) {
        int m = Y_pred.getRows();
        int n = Y_pred.getCols();
        double loss = 0.0;

        switch (type) {
            case LossType::MSE: {
                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < n; ++j) {
                        double diff = Y_pred(i, j) - Y_true(i, j);
                        loss += diff * diff;
                    }
                }
                return loss / (m * n);
            }
            case LossType::CROSS_ENTROPY: {
                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < n; ++j) {
                        // clip value to prevent log(0)
                        double val = std::max(1e-15, std::min(1.0 - 1e-15, Y_pred(i, j)));
                        loss -= Y_true(i, j) * std::log(val);
                    }
                }
                return loss / m;
            }
            case LossType::BINARY_CROSS_ENTROPY: {
                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < n; ++j) {
                        double pred = std::max(1e-15, std::min(1.0 - 1e-15, Y_pred(i, j)));
                        double target = Y_true(i, j);
                        loss -= (target * std::log(pred) + (1.0 - target) * std::log(1.0 - pred));
                    }
                }
                return loss / m;
            }
            default:
                throw std::invalid_argument("Unknown loss type");
        }
    }

    static Matrix gradient(const Matrix& Y_pred, const Matrix& Y_true, LossType type) {
        int m = Y_pred.getRows();
        int n = Y_pred.getCols();
        Matrix grad(m, n);

        switch (type) {
            case LossType::MSE: {
                // dL/dY_pred = 2 * (Y_pred - Y_true) / (m * n)
                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < n; ++j) {
                        grad(i, j) = 2.0 * (Y_pred(i, j) - Y_true(i, j)) / (m * n);
                    }
                }
                return grad;
            }
            case LossType::CROSS_ENTROPY: {
                // dL/dY_pred = -Y_true / (Y_pred * m)
                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < n; ++j) {
                        double val = std::max(1e-15, std::min(1.0 - 1e-15, Y_pred(i, j)));
                        grad(i, j) = -Y_true(i, j) / (val * m);
                    }
                }
                return grad;
            }
            case LossType::BINARY_CROSS_ENTROPY: {
                // dL/dY_pred = -( (Y_true / Y_pred) - ((1 - Y_true) / (1 - Y_pred)) ) / m
                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < n; ++j) {
                        double pred = std::max(1e-15, std::min(1.0 - 1e-15, Y_pred(i, j)));
                        double target = Y_true(i, j);
                        grad(i, j) = -((target / pred) - ((1.0 - target) / (1.0 - pred))) / m;
                    }
                }
                return grad;
            }
            default:
                throw std::invalid_argument("Unknown loss type");
        }
    }

    static std::string toString(LossType type) {
        switch (type) {
            case LossType::MSE: return "Mean Squared Error (MSE)";
            case LossType::CROSS_ENTROPY: return "Categorical Cross Entropy";
            case LossType::BINARY_CROSS_ENTROPY: return "Binary Cross Entropy";
            default: return "Unknown";
        }
    }
};

#endif // LOSS_HPP
