# 🤖 Machine Learning Portfolio

Welcome to my Machine Learning repository! This workspace is dedicated to building, understanding, and implementing core artificial intelligence and deep learning algorithms. 

Instead of relying solely on high-level APIs or pre-built frameworks, the focus of this repository is on the underlying mathematics, efficient software engineering, and raw algorithmic implementation.

---

## 📂 Projects in this Repository

### 1. [Multilayer Perceptron (MLP) from Scratch in C++](./mlp_from_scratch/)
A fully functional deep learning Neural Network implemented entirely from the ground up using pure C++17. 
* **The Goal:** Open the "black box" of Deep Learning by writing the forward pass, backpropagation calculus, and optimization algorithms by hand.
* **Tech Stack:** C++17, OpenMP (Multi-threading), GNU Make.
* **Key Features:** No external ML or math libraries used. Features a custom matrix computing engine, multiple activation functions, binary model serialization (saving/loading weights), and terminal-based ASCII data visualization.
* **Demos Included:** Boolean XOR logic, non-linear concentric circle classification, and the real-world Iris flower dataset.

*(More projects will be added here as the repository grows!)*

---

## 🧠 Engineering Philosophy

This repository is built on a few core principles:
1. **Math to Code:** Bridging the gap between theoretical multivariable calculus / linear algebra and actual, compiling software.
2. **Performance First:** Utilizing system-level languages (like C++) and hardware optimizations (like OpenMP) to understand how ML models execute efficiently at the CPU level.
3. **No Magic:** You don't truly understand an algorithm until you can build it from an empty text file.

---

## 🚀 How to Navigate

Each project in this repository is self-contained in its own directory with its own detailed `README.md`, source code, and build instructions. 

To explore a project, simply navigate to its folder. For example:
```bash
cd mlp_from_scratch
cat README.md