#ifndef MLP_HPP
#define MLP_HPP

#include <vector>
#include <memory>
#include <fstream>
#include <string>
#include "Layer.hpp"

class MLP {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    MLP() = default;

    void addLayer(int in_features, int out_features, ActivationType act_type) {
        layers.push_back(std::make_unique<Layer>(in_features, out_features, act_type));
    }

    Matrix forward(const Matrix& X) {
        Matrix current_activation = X;
        for (auto& layer : layers) {
            current_activation = layer->forward(current_activation);
        }
        return current_activation;
    }

    void backward(const Matrix& dLoss) {
        Matrix current_grad = dLoss;
        // Traverse layers in reverse order for backpropagation
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            current_grad = (*it)->backward(current_grad);
        }
    }

    void update(double learning_rate, double momentum = 0.9, double weight_decay = 0.0) {
        for (auto& layer : layers) {
            layer->updateParameters(learning_rate, momentum, weight_decay);
        }
    }

    const std::vector<std::unique_ptr<Layer>>& getLayers() const {
        return layers;
    }

    void save(const std::string& filename) const {
        std::ofstream os(filename, std::ios::binary);
        if (!os) throw std::runtime_error("Could not open file for saving: " + filename);
        int num_layers = layers.size();
        os.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
        for (const auto& layer : layers) {
            layer->save(os);
        }
    }

    void load(const std::string& filename) {
        std::ifstream is(filename, std::ios::binary);
        if (!is) throw std::runtime_error("Could not open file for loading: " + filename);
        int num_layers;
        is.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
        // Note: For simplicity, this assumes the architecture (addLayer) is already set up identically 
        // before calling load(). A more robust implementation would recreate layers dynamically.
        if (num_layers != (int)layers.size()) {
            throw std::runtime_error("Architecture mismatch when loading weights.");
        }
        for (auto& layer : layers) {
            layer->load(is);
        }
    }
};

#endif // MLP_HPP
