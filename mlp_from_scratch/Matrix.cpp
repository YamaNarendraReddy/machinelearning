#include "Matrix.hpp"
#include <iomanip>

Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(int r, int c, double init_val) : rows(r), cols(c), data(r, std::vector<double>(c, init_val)) {}

Matrix::Matrix(const std::vector<std::vector<double>>& d) {
    rows = d.size();
    cols = rows > 0 ? d[0].size() : 0;
    data = d;
}

double& Matrix::operator()(int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[r][c];
}

double Matrix::operator()(int r, int c) const {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[r][c];
}

Matrix Matrix::randn(int r, int c, double mean, double stddev) {
    Matrix m(r, c);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, stddev);

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            m(i, j) = dist(gen);
        }
    }
    return m;
}

Matrix Matrix::xavier(int r, int c, int n_in, int n_out) {
    double limit = std::sqrt(6.0 / (n_in + n_out));
    Matrix m(r, c);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-limit, limit);

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            m(i, j) = dist(gen);
        }
    }
    return m;
}

Matrix Matrix::he(int r, int c, int n_in) {
    double stddev = std::sqrt(2.0 / n_in);
    return randn(r, c, 0.0, stddev);
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(j, i) = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::dot(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dot product dimensions mismatch: " + 
                                    std::to_string(rows) + "x" + std::to_string(cols) + 
                                    " and " + std::to_string(other.rows) + "x" + std::to_string(other.cols));
    }
    Matrix result(rows, other.cols, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < cols; ++k) {
            double temp = data[i][k];
            for (int j = 0; j < other.cols; ++j) {
                result(i, j) += temp * other.data[k][j];
            }
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix add dimensions mismatch");
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::addBias(const Matrix& bias) const {
    if (bias.rows != 1 || bias.cols != cols) {
        throw std::invalid_argument("addBias: Bias must be a row vector matching the matrix column count");
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + bias.data[0][j];
        }
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix subtract dimensions mismatch");
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::multiply(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix multiply dimensions mismatch");
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::multiply(double scalar) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = func(data[i][j]);
        }
    }
    return result;
}

Matrix Matrix::apply(std::function<double(double, double)> func, const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix apply dimensions mismatch");
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = func(data[i][j], other.data[i][j]);
        }
    }
    return result;
}

void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << data[i][j] << " ";
        }
        std::cout << "\n";
    }
}

void Matrix::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    os.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    for (int i = 0; i < rows; ++i) {
        os.write(reinterpret_cast<const char*>(data[i].data()), cols * sizeof(double));
    }
}

void Matrix::load(std::istream& is) {
    is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    data.assign(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        is.read(reinterpret_cast<char*>(data[i].data()), cols * sizeof(double));
    }
}
