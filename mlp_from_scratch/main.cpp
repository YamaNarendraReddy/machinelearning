#include <iostream>
#include <vector>
#include <iomanip>
#include "Matrix.hpp"
#include "Activation.hpp"
#include "Loss.hpp"
#include "MLP.hpp"
#include "Dataset.hpp"

// Utility to print ASCII scatter plot / decision boundary of concentric circles
void printDecisionBoundary(MLP& mlp) {
    std::cout << "\n================ DECISION BOUNDARY (ASCII VISUALIZATION) ================\n";
    std::cout << "Inner ring (class 0) -> '.'  Outer ring (class 1) -> '#'\n\n";
    
    int width = 50;
    int height = 20;
    for (int y = 0; y < height; ++y) {
        double py = -1.2 + 2.4 * y / (height - 1);
        for (int x = 0; x < width; ++x) {
            double px = -1.2 + 2.4 * x / (width - 1);
            Matrix sample(std::vector<std::vector<double>>{{px, py}});
            Matrix pred = mlp.forward(sample);
            double val = pred(0, 0);
            if (val < 0.5) {
                std::cout << ".";
            } else {
                std::cout << "#";
            }
        }
        std::cout << "\n";
    }
    std::cout << "=========================================================================\n\n";
}

void runXORDemo() {
    std::cout << "========================================\n";
    std::cout << "         1. TRAINING ON XOR DATA        \n";
    std::cout << "========================================\n";

    auto [X, Y] = Dataset::generateXOR();

    // Create MLP: Input(2) -> Hidden(4, ReLU) -> Output(1, Sigmoid)
    MLP mlp;
    mlp.addLayer(2, 4, ActivationType::RELU);
    mlp.addLayer(4, 1, ActivationType::SIGMOID);

    double lr = 0.1;
    int epochs = 1000;

    std::cout << "Initial Predictions on XOR:\n";
    Matrix initial_pred = mlp.forward(X);
    for (int i = 0; i < 4; ++i) {
        std::cout << "X: [" << X(i, 0) << ", " << X(i, 1) << "] -> Target: " << Y(i, 0) 
                  << " | Pred: " << std::fixed << std::setprecision(4) << initial_pred(i, 0) << "\n";
    }

    std::cout << "\nTraining for " << epochs << " epochs...\n";
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        Matrix Y_pred = mlp.forward(X);
        double loss = Loss::compute(Y_pred, Y, LossType::BINARY_CROSS_ENTROPY);
        
        Matrix dLoss = Loss::gradient(Y_pred, Y, LossType::BINARY_CROSS_ENTROPY);
        mlp.backward(dLoss);
        mlp.update(lr, 0.9, 1e-4); // learning_rate, momentum, weight_decay

        if (epoch % 200 == 0 || epoch == 1) {
            std::cout << "Epoch " << epoch << "/" << epochs << " | Loss: " << std::fixed << std::setprecision(6) << loss << "\n";
        }
    }

    std::cout << "\nFinal Predictions on XOR:\n";
    Matrix final_pred = mlp.forward(X);
    for (int i = 0; i < 4; ++i) {
        std::cout << "X: [" << X(i, 0) << ", " << X(i, 1) << "] -> Target: " << Y(i, 0) 
                  << " | Pred: " << std::fixed << std::setprecision(4) << final_pred(i, 0) 
                  << (std::abs(final_pred(i, 0) - Y(i, 0)) < 0.1 ? " (CORRECT)" : " (INCORRECT)") << "\n";
    }
    std::cout << "\n";
}

void runConcentricCirclesDemo() {
    std::cout << "========================================\n";
    std::cout << "  2. TRAINING ON CONCENTRIC CIRCLES     \n";
    std::cout << "========================================\n";

    // Generate dataset
    int train_samples = 500;
    int test_samples = 100;
    auto [X_train, Y_train] = Dataset::generateConcentricCircles(train_samples, 0.05);
    auto [X_test, Y_test] = Dataset::generateConcentricCircles(test_samples, 0.05);

    // Create MLP: Input(2) -> Hidden(16, Tanh) -> Hidden(8, Tanh) -> Output(1, Sigmoid)
    MLP mlp;
    mlp.addLayer(2, 16, ActivationType::TANH);
    mlp.addLayer(16, 8, ActivationType::TANH);
    mlp.addLayer(8, 1, ActivationType::SIGMOID);

    double lr = 0.05;
    int epochs = 800;
    int batch_size = 32;

    std::cout << "Dataset generated: " << train_samples << " train samples, " << test_samples << " test samples.\n";
    
    std::cout << "\nInitial decision boundary visualization before training:\n";
    printDecisionBoundary(mlp);

    std::cout << "Training model using mini-batch gradient descent with momentum...\n";
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto batches = Dataset::getMiniBatches(X_train, Y_train, batch_size);
        double epoch_loss = 0.0;

        for (auto& batch : batches) {
            Matrix X_batch = batch.first;
            Matrix Y_batch = batch.second;

            Matrix Y_pred = mlp.forward(X_batch);
            epoch_loss += Loss::compute(Y_pred, Y_batch, LossType::BINARY_CROSS_ENTROPY);

            Matrix dLoss = Loss::gradient(Y_pred, Y_batch, LossType::BINARY_CROSS_ENTROPY);
            mlp.backward(dLoss);
            mlp.update(lr, 0.9, 1e-4);
        }
        epoch_loss /= batches.size();

        if (epoch % 100 == 0 || epoch == 1) {
            // Evaluate on test set
            Matrix Y_test_pred = mlp.forward(X_test);
            double test_loss = Loss::compute(Y_test_pred, Y_test, LossType::BINARY_CROSS_ENTROPY);
            
            // Calculate accuracy
            int correct = 0;
            for (int i = 0; i < test_samples; ++i) {
                double val = Y_test_pred(i, 0);
                double pred_class = val >= 0.5 ? 1.0 : 0.0;
                if (pred_class == Y_test(i, 0)) {
                    correct++;
                }
            }
            double test_acc = (double)correct / test_samples * 100.0;

            std::cout << "Epoch " << std::setw(3) << epoch << "/" << epochs 
                      << " | Train Loss: " << std::fixed << std::setprecision(6) << epoch_loss 
                      << " | Test Loss: " << test_loss 
                      << " | Test Accuracy: " << std::fixed << std::setprecision(2) << test_acc << "%\n";
        }
    }

    std::cout << "\nDecision boundary boundary visualization after training:\n";
    printDecisionBoundary(mlp);
}

void runIrisDemo() {
    std::cout << "========================================\n";
    std::cout << "  3. TRAINING ON IRIS DATASET (REAL DATA)\n";
    std::cout << "========================================\n";

    try {
        auto [X, Y] = Dataset::loadIrisCSV("iris.csv");
        int total_samples = X.getRows();
        std::cout << "Successfully loaded " << total_samples << " samples from iris.csv\n";

        // MLP for Iris: Input(4) -> Hidden(8, ReLU) -> Output(3, Softmax)
        MLP mlp;
        mlp.addLayer(4, 8, ActivationType::RELU);
        mlp.addLayer(8, 3, ActivationType::SOFTMAX);

        double lr = 0.05;
        int epochs = 200;
        int batch_size = 16;

        std::cout << "Training model on Iris dataset...\n";
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            auto batches = Dataset::getMiniBatches(X, Y, batch_size);
            double epoch_loss = 0.0;

            for (auto& batch : batches) {
                Matrix X_batch = batch.first;
                Matrix Y_batch = batch.second;

                Matrix Y_pred = mlp.forward(X_batch);
                epoch_loss += Loss::compute(Y_pred, Y_batch, LossType::CROSS_ENTROPY);

                Matrix dLoss = Loss::gradient(Y_pred, Y_batch, LossType::CROSS_ENTROPY);
                mlp.backward(dLoss);
                mlp.update(lr, 0.9, 1e-4);
            }
            epoch_loss /= batches.size();

            if (epoch % 50 == 0 || epoch == 1) {
                Matrix Y_pred_all = mlp.forward(X);
                int correct = 0;
                for (int i = 0; i < total_samples; ++i) {
                    int pred_class = 0;
                    double max_val = Y_pred_all(i, 0);
                    int true_class = 0;
                    double max_true = Y(i, 0);
                    
                    for (int j = 1; j < 3; ++j) {
                        if (Y_pred_all(i, j) > max_val) {
                            max_val = Y_pred_all(i, j);
                            pred_class = j;
                        }
                        if (Y(i, j) > max_true) {
                            max_true = Y(i, j);
                            true_class = j;
                        }
                    }
                    if (pred_class == true_class) correct++;
                }
                double acc = (double)correct / total_samples * 100.0;
                std::cout << "Epoch " << std::setw(3) << epoch << "/" << epochs 
                          << " | Loss: " << std::fixed << std::setprecision(6) << epoch_loss 
                          << " | Accuracy: " << std::fixed << std::setprecision(2) << acc << "%\n";
            }
        }

        std::cout << "Saving trained weights to 'iris_model.bin'...\n";
        mlp.save("iris_model.bin");

        std::cout << "Loading weights from 'iris_model.bin' to a new model to verify serialization...\n";
        MLP mlp_loaded;
        mlp_loaded.addLayer(4, 8, ActivationType::RELU);
        mlp_loaded.addLayer(8, 3, ActivationType::SOFTMAX);
        mlp_loaded.load("iris_model.bin");

        Matrix Y_pred_all = mlp_loaded.forward(X);
        int correct = 0;
        for (int i = 0; i < total_samples; ++i) {
            int pred_class = 0;
            double max_val = Y_pred_all(i, 0);
            int true_class = 0;
            double max_true = Y(i, 0);
            for (int j = 1; j < 3; ++j) {
                if (Y_pred_all(i, j) > max_val) { max_val = Y_pred_all(i, j); pred_class = j; }
                if (Y(i, j) > max_true) { max_true = Y(i, j); true_class = j; }
            }
            if (pred_class == true_class) correct++;
        }
        std::cout << "Loaded model accuracy on entire dataset: " << ((double)correct / total_samples * 100.0) << "%\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Failed to run Iris demo: " << e.what() << "\n";
    }
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "       MULTIPLAYER PERCEPTRON (MLP) FROM SCRATCH IN C++  \n";
    std::cout << "=========================================================\n\n";

    runXORDemo();
    runConcentricCirclesDemo();
    runIrisDemo();

    return 0;
}
