#ifndef LAYER_HPP
#define LAYER_HPP

#include "Matrix.hpp"
#include "Activation.hpp"

class Layer {
public:
    Matrix W; // Weights
    Matrix b; // Biases
    Matrix dW; // Weight gradients
    Matrix db; // Bias gradients

    // Velocity vectors for SGD with momentum
    Matrix vW;
    Matrix vb;

    // Cache for backpropagation
    Matrix A_prev;
    Matrix Z;
    Matrix A;

    ActivationType activation_type;

    Layer(int in_features, int out_features, ActivationType act_type)
        : activation_type(act_type) {
        
        // Use Xavier/He initialization depending on activation type
        if (act_type == ActivationType::RELU) {
            W = Matrix::he(in_features, out_features, in_features);
        } else {
            W = Matrix::xavier(in_features, out_features, in_features, out_features);
        }
        b = Matrix(1, out_features, 0.0);
        
        vW = Matrix(in_features, out_features, 0.0);
        vb = Matrix(1, out_features, 0.0);
    }

    Matrix forward(const Matrix& input) {
        A_prev = input;
        Z = A_prev.dot(W).addBias(b);
        A = Activation::forward(Z, activation_type);
        return A;
    }

    Matrix backward(const Matrix& dA) {
        // dA has shape [batch_size, out_features]
        // Compute dZ (gradient of Loss w.r.t Z)
        Matrix dZ = Activation::backward(Z, A, dA, activation_type);

        int batch_size = A_prev.getRows();

        // dW = A_prev^T * dZ
        dW = A_prev.transpose().dot(dZ);

        // db = sum of dZ columns (across batch size)
        db = Matrix(1, Z.getCols(), 0.0);
        for (int j = 0; j < dZ.getCols(); ++j) {
            double sum = 0.0;
            for (int i = 0; i < batch_size; ++i) {
                sum += dZ(i, j);
            }
            db(0, j) = sum / batch_size; // average gradient over batch
        }
        
        // Also average dW over the batch
        dW = dW.multiply(1.0 / batch_size);

        // dA_prev = dZ * W^T
        Matrix dA_prev = dZ.dot(W.transpose());
        return dA_prev;
    }

    void updateParameters(double learning_rate, double momentum = 0.9, double weight_decay = 0.0) {
        // Apply weight decay (L2 regularization) gradient if present: dW = dW + weight_decay * W
        if (weight_decay > 0.0) {
            dW = dW.add(W.multiply(weight_decay));
        }

        // vW = momentum * vW + learning_rate * dW
        vW = vW.multiply(momentum).add(dW.multiply(learning_rate));
        // W = W - vW
        W = W.subtract(vW);

        // vb = momentum * vb + learning_rate * db
        vb = vb.multiply(momentum).add(db.multiply(learning_rate));
        // b = b - vb
        b = b.subtract(vb);
    }

    void save(std::ostream& os) const {
        int act = static_cast<int>(activation_type);
        os.write(reinterpret_cast<const char*>(&act), sizeof(act));
        W.save(os);
        b.save(os);
    }

    void load(std::istream& is) {
        int act;
        is.read(reinterpret_cast<char*>(&act), sizeof(act));
        activation_type = static_cast<ActivationType>(act);
        W.load(is);
        b.load(is);
    }
};

#endif // LAYER_HPP
