#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>
#include <random>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <fstream>

class Matrix {
private:
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

public:
    // Constructors
    Matrix();
    Matrix(int r, int c, double init_val = 0.0);
    Matrix(const std::vector<std::vector<double>>& d);

    // Getters
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    const std::vector<std::vector<double>>& getData() const { return data; }
    std::vector<std::vector<double>>& getDataRef() { return data; }

    // Accessors
    double& operator()(int r, int c);
    double operator()(int r, int c) const;

    // Static Initialization helper functions
    static Matrix randn(int r, int c, double mean = 0.0, double stddev = 1.0);
    static Matrix xavier(int r, int c, int n_in, int n_out);
    static Matrix he(int r, int c, int n_in);

    // Basic Operations
    Matrix transpose() const;
    Matrix dot(const Matrix& other) const; // Matrix multiplication
    Matrix add(const Matrix& other) const; // Element-wise addition
    Matrix addBias(const Matrix& bias) const; // Adding row vector bias [1, cols] to each row
    Matrix subtract(const Matrix& other) const; // Element-wise subtraction
    Matrix multiply(const Matrix& other) const; // Hadamard (element-wise) product
    Matrix multiply(double scalar) const; // Scalar multiplication

    // Apply element-wise function
    Matrix apply(std::function<double(double)> func) const;
    Matrix apply(std::function<double(double, double)> func, const Matrix& other) const;

    // Serialization
    void save(std::ostream& os) const;
    void load(std::istream& is);

    // Helper functions
    void print() const;
};

#endif // MATRIX_HPP
