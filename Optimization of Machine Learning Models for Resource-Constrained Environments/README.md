# Optimization of Machine Learning Models for Resource-Constrained Environments

This repository contains the code, research materials, and results from a comprehensive study on optimizing machine learning models for deployment in resource-constrained environments, such as IoT devices and embedded systems.

## Overview

The study explores advanced compression and quantization techniques to enhance the efficiency of machine learning (ML) models while minimizing resource requirements. It benchmarks performance across CIFAR-10, MNIST, and Fashion MNIST datasets, focusing on metrics like:
- **Accuracy**
- **Latency**
- **Model Size**
- **Energy Efficiency**

Key findings demonstrate that significant reductions in latency and model size can be achieved with minimal accuracy loss, enabling real-world applications in edge computing and battery-powered devices.

---

## Features
- Structured pruning, Huffman encoding, and quantization techniques
- Comprehensive benchmarking on CIFAR-10, MNIST, and Fashion MNIST datasets
- Analysis of trade-offs between accuracy, latency, model size, and energy consumption
- Compatibility with edge devices like Raspberry Pi and Nvidia Jetson Nano

---

## Project Structure
```
ml-model-optimization-resource-constraints/
├── main.ipynb                      # Primary code notebook
├── Optimization_ML_Models.pdf      # Research paper
├── README.md                       # Project overview and instructions
├── LICENSE                         # License file
├── requirements.txt                # Python dependencies
├── data/                           # Datasets
│   ├── cifar10/
│   ├── mnist/
│   └── fashion_mnist/
├── models/                         # Trained models
│   ├── compressed/
│   ├── quantized/
│   └── uncompressed/
├── results/                        # Results and plots
│   ├── accuracy.csv
│   ├── latency.csv
│   ├── model_size.csv
│   ├── energy_efficiency.csv
│   └── plots/
├── src/                            # Python scripts
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── optimization_techniques.py
│   ├── evaluation_metrics.py
│   └── utils.py
└── CONTRIBUTING.md                 # Contribution guidelines
```

---

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. Recommended environment: Anaconda or virtualenv.

### Installation
Clone the repository:
```bash
git clone https://github.com/<your-username>/ml-model-optimization-resource-constraints.git
cd ml-model-optimization-resource-constraints
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Datasets
Place datasets in the `data/` directory:
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)

### Usage
1. Open the `main.ipynb` notebook for step-by-step analysis.
2. Run specific tasks using the scripts in the `src/` folder.
3. View results in the `results/` folder.

---

## Results
### Key Metrics
| Metric           | CIFAR-10 | MNIST  | Fashion MNIST |
|------------------|----------|--------|---------------|
| Accuracy         | 89.77%   | 98.96% | 91.50%        |
| Latency (ms)     | 37.0     | 34.0   | 35.0          |
| Model Size (KB)  | 3.75     | 3.57   | 3.57          |
| Energy Reduction | 40%      | 35%    | 38%           |

Plots visualizing these results can be found in the `results/plots/` directory.

---

## Contributions
Contributions are welcome! Please see the `CONTRIBUTING.md` file for guidelines.

---

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Contact
For any questions or feedback, please contact **Karan** at karan.codes22@gmail.com.

