# Stochastic Programming Algorithms

Welcome to the repository for implementing various algorithms to solve stochastic programming problems. In this repository, you'll find implementations for:

- Single Cut L-Shape Method
- Multi-Cut L-Shape Method
- Regularized Decomposition
- Nested Decomposition
- Scenario Reduction Algorithms

These algorithms are implemented using the [CVXPY](https://www.cvxpy.org/) library introduced by Prof. Stephen P. Boyd and the [PuLP](https://coin-or.github.io/pulp/) library.

## Table of Contents

- [Introduction](#introduction)
- [Algorithms](#algorithms)
  - [Single Cut L-Shape Method](#single-cut-l-shape-method)
  - [Multi-Cut L-Shape Method](#multi-cut-l-shape-method)
  - [Regularized Decomposition](#regularized-decomposition)
  - [Nested Decomposition](#nested-decomposition)
  - [Scenario Reduction Algorithms](#scenario-reduction-algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Stochastic programming is a framework for modeling optimization problems that involve uncertainty. This repository provides implementations of several key algorithms used to solve stochastic programming problems. Each algorithm is implemented in Python using the CVXPY and PuLP libraries, which allow for the definition and solving of convex optimization problems.

## Algorithms

### Single Cut L-Shape Method

The Single Cut L-shape method is a decomposition algorithm used to solve two-stage stochastic linear programs. This method iteratively refines the solution by adding cuts to approximate the recourse function.

### Multi-Cut L-Shape Method

The Multi Cut L-shape method is an extension of the Single Cut method that generates multiple cuts per iteration, potentially leading to faster convergence by providing a better approximation of the recourse function.

### Regularized Decomposition

Regularized Decomposition introduces a regularization term to the decomposition algorithm to improve convergence properties and stability. It is particularly useful for large-scale stochastic programming problems.

### Nested Decomposition

Nested Decomposition is an approach used to solve multi-stage stochastic programming problems. It extends the L-Shape method by considering multiple stages of decisions and uncertainties.

### Scenario Reduction Algorithms

Scenario Reduction Algorithms are techniques used to generate scenarios for stochastic programming. These scenarios represent possible realizations of uncertainties and are crucial for accurately modeling and solving stochastic problems.

## Installation

You need to have Python installed to use the codes in this repository. You can install the required dependencies using `pip`. Here are the steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/ErfanAmani2000/Stochastic-Programming.git
    cd Stochastic-Programming
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Each algorithm has its own script in the repository. You can run these scripts to see how the algorithms work and to solve example stochastic programming problems.

For example, to run the Single Cut L-Shape Method, use the following command:
```bash
python single_cut_l_shape.py
