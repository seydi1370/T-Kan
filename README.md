# T-KAN: Kolmogorov-Arnold Based Network Using Trigonometric Polynomials

This repository contains the implementation of T-KAN, a novel neural network architecture inspired by Kolmogorov-Arnold Networks (KANs) but using trigonometric polynomials instead of spline coefficients.

## Overview

T-KAN leverages the power of trigonometric polynomials to learn complex patterns and relationships in data. The proposed architecture is evaluated on the MNIST dataset, demonstrating its effectiveness in image classification tasks.

## Key Features

- Based on the Kolmogorov-Arnold representation theorem [1], which states that any continuous function can be represented as a composition of simple functions.
- Utilizes trigonometric polynomials instead of spline coefficients, enabling efficient learning of complex functions.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Usage

1. Clone the repository.
2. Install the required dependencies.
3. Run the `T-KAN.ipynb` script to train and evaluate the T-KAN model on the MNIST dataset.

## References

[1] Z. Liu et al., "KAN: Kolmogorov-Arnold Networks," arXiv preprint arXiv:2404.19756, 2024.

[2] S. SS, "Chebyshev Polynomial-Based Kolmogorov-Arnold Networks: An Efficient Architecture for Nonlinear Function Approximation," arXiv preprint arXiv:2405.07200, 2024.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

We would like to thank the authors of the related works [1,2] for their contributions and insights that inspired the development of T-KAN.
