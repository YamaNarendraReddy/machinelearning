# Multilayer Perceptron (MLP) from Scratch in C++

Welcome to the **MLP from Scratch** repository! This project implements a fully functional deep learning Neural Network—specifically a Multilayer Perceptron (MLP)—entirely from the ground up using pure C++17. 

**No external Machine Learning libraries (like PyTorch or TensorFlow) or math libraries (like Eigen) were used.** Everything from matrix multiplication to backpropagation calculus is implemented by hand.

---

## 🧠 What is this project and why does it matter?

For many, Machine Learning feels like "magic." You feed data into a black box, and it gives you intelligent predictions. 

This project **opens the black box**. 

By building the neural network entirely from scratch, it demonstrates exactly how an AI "learns":
1. **Forward Pass**: Data is multiplied by matrices (representing "weights" or the network's knowledge) and passed through non-linear activation functions (like ReLU or Sigmoid).
2. **Loss Calculation**: The network compares its prediction to the real answer and calculates the error.
3. **Backpropagation**: Using multivariable calculus (the Chain Rule), the network figures out exactly which weights caused the error.
4. **Optimization**: The network adjusts its weights slightly to be more accurate next time (using Stochastic Gradient Descent with Momentum).

**For Non-ML Evaluators:** This project proves strong software engineering capabilities (Object-Oriented C++, memory management, OpenMP multi-threading) paired with rigorous applied mathematics (Linear Algebra, Calculus).

---

## 📂 Directory Structure

The code is highly modular, separating the math from the network architecture.

```text
mlp_from_scratch/
├── Makefile           # Compilation instructions for easy building
├── main.cpp           # The main execution file containing the demos
├── Matrix.hpp/.cpp    # Custom 2D Matrix engine (handles math, OpenMP threading, and serialization)
├── Layer.hpp          # Represents a single layer of neurons (stores weights and biases)
├── Activation.hpp/.cpp# Mathematical functions that allow the network to learn complex patterns
├── Loss.hpp           # Functions to calculate how "wrong" the network's predictions are
├── Dataset.hpp        # Utility to generate synthetic data and load CSV files
├── MLP.hpp            # The main container that chains Layers together into a full network
└── iris.csv           # A classic real-world dataset for flower classification
```

---

## 🚀 How to Run and Test the Code

Building and running the project is incredibly simple. You only need a standard C++ compiler (like `g++`) and `make` installed on your machine.

1. **Open a terminal** and navigate to this directory.
2. **Compile and run** the project by typing:
   ```bash
   make clean && make run
   ```

The program will automatically compile using OpenMP optimizations and execute three distinct tests.

---

## 📊 How to Read the Test Results

When you run the code, it will output three demonstrations. Here is how to understand what the AI is doing:

### Demo 1: The XOR Problem
* **What it is:** The AI is trying to learn basic boolean logic (Exclusive OR).
* **What to look for:** Initially, the predictions will hover around `0.5` (guessing). After 1000 epochs (training rounds), look at the `Final Predictions`. The network should successfully output `~0.0` or `~1.0`, matching the target perfectly with 100% accuracy.

### Demo 2: Concentric Circles (ASCII Visualization)
* **What it is:** The AI must separate data points arranged in two circles (an inner ring and an outer ring). This proves the network can understand complex, non-linear boundaries.
* **What to look for:** As the epochs progress, watch the `Test Accuracy` climb to 100%. 
* **The Magic:** Look at the ASCII art output in your terminal! The network actually "draws" its understanding of the data. You will see a cluster of `.` characters representing the inner circle, perfectly surrounded by `#` characters representing the outer circle. 

### Demo 3: The Iris Dataset (Real World Data)
* **What it is:** The network parses a real-world CSV file containing flower measurements and tries to classify the exact species of Iris flower.
* **What to look for:** The network will train incredibly fast and achieve ~95-98% accuracy. Immediately after, it will **save its brain (weights) to a file (`iris_model.bin`)**, create a brand new empty network, and load the brain back in to prove that the knowledge was successfully preserved.

---

## 🌍 Real-World Applications of MLPs

While this project trains on smaller datasets for speed and demonstration purposes, Multilayer Perceptrons form the foundation of modern Deep Learning. The exact same mathematical principles used in this code are used to build systems that:

- **Predict Financial Markets:** Analyzing historical data to predict stock trends.
- **Medical Diagnosis:** Looking at patient vitals and bloodwork to classify disease risk.
- **Fraud Detection:** Identifying unusual patterns in credit card transactions.
- **Recommendation Systems:** Suggesting the next movie or product you might like based on user history. 

This repository serves as a foundational proof-of-concept that scales directly to those massive, world-changing applications.
