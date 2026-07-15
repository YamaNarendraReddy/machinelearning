#ifndef DATASET_HPP
#define DATASET_HPP

#include "Matrix.hpp"
#include <utility>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

class Dataset {
public:
    // Generate XOR dataset
    static std::pair<Matrix, Matrix> generateXOR() {
        std::vector<std::vector<double>> X_data = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        };
        std::vector<std::vector<double>> Y_data = {
            {0.0},
            {1.0},
            {1.0},
            {0.0}
        };
        return {Matrix(X_data), Matrix(Y_data)};
    }

    // Generate concentric circles dataset for non-linear classification demo
    static std::pair<Matrix, Matrix> generateConcentricCircles(int samples, double noise = 0.05) {
        std::vector<std::vector<double>> X_data;
        std::vector<std::vector<double>> Y_data;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist_noise(0.0, noise);
        std::uniform_real_distribution<double> dist_angle(0.0, 2.0 * M_PI);

        int samples_per_ring = samples / 2;

        // Inner circle (class 0)
        for (int i = 0; i < samples_per_ring; ++i) {
            double angle = dist_angle(gen);
            double r = 0.4 + dist_noise(gen);
            X_data.push_back({r * std::cos(angle), r * std::sin(angle)});
            Y_data.push_back({0.0});
        }

        // Outer circle (class 1)
        for (int i = 0; i < samples_per_ring; ++i) {
            double angle = dist_angle(gen);
            double r = 0.8 + dist_noise(gen);
            X_data.push_back({r * std::cos(angle), r * std::sin(angle)});
            Y_data.push_back({1.0});
        }

        return {Matrix(X_data), Matrix(Y_data)};
    }

    // Shuffle and batch helper
    static std::vector<std::pair<Matrix, Matrix>> getMiniBatches(const Matrix& X, const Matrix& Y, int batch_size) {
        int m = X.getRows();
        std::vector<int> indices(m);
        for (int i = 0; i < m; ++i) indices[i] = i;

        // Shuffle indices
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        std::vector<std::pair<Matrix, Matrix>> batches;
        for (int i = 0; i < m; i += batch_size) {
            int current_batch_size = std::min(batch_size, m - i);
            std::vector<std::vector<double>> batch_X_data(current_batch_size);
            std::vector<std::vector<double>> batch_Y_data(current_batch_size);

            for (int j = 0; j < current_batch_size; ++j) {
                int idx = indices[i + j];
                batch_X_data[j] = X.getData()[idx];
                batch_Y_data[j] = Y.getData()[idx];
            }

            batches.push_back({Matrix(batch_X_data), Matrix(batch_Y_data)});
        }

        return batches;
    }

    // Load Iris dataset from CSV
    // Returns pair of (Features, One-Hot Targets)
    static std::pair<Matrix, Matrix> loadIrisCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open Iris dataset CSV file: " + filename);
        }

        std::vector<std::vector<double>> X_data;
        std::vector<std::vector<double>> Y_data;
        std::string line, token;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::vector<double> features;
            for (int i = 0; i < 4; ++i) {
                if (std::getline(ss, token, ',')) {
                    features.push_back(std::stod(token));
                }
            }
            if (features.size() != 4) continue;

            std::string label;
            std::getline(ss, label, ',');
            
            // One-hot encode the target
            std::vector<double> target = {0.0, 0.0, 0.0};
            if (label == "Iris-setosa") {
                target[0] = 1.0;
            } else if (label == "Iris-versicolor") {
                target[1] = 1.0;
            } else if (label == "Iris-virginica") {
                target[2] = 1.0;
            } else {
                continue; // invalid label
            }

            X_data.push_back(features);
            Y_data.push_back(target);
        }

        return {Matrix(X_data), Matrix(Y_data)};
    }
};

#endif // DATASET_HPP
