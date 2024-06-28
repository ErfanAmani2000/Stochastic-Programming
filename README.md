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
- [Mathematical Model](#mathematical-model)
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

Stochastic programming is a framework for modeling optimization problems that involve uncertainty. This repository implements several key algorithms used to solve stochastic programming problems. Each algorithm is implemented in Python using the CVXPY and PuLP libraries, which allow for the definition and solving of convex optimization problems.

## Notation

### Sets
- \( i \): Index for product type (1: wheat, 2: corn, 3: high-price sugar beet, 4: low-price sugar beet, \( i \in \{1, 2, 3, 4\} \))
- \( t \): Index for time period (\( t \in \{1, 2, 3\} \))
- \( k \): Index for scenario

### Parameters
- \( C_i \): Planting cost for product type \( i \)
- \( B_i \): Purchase cost for product type \( i \)
- \( S_i \): Selling cost for product type \( i \)
- \( d_i \): Demand for product type \( i \)
- \( p_k \): Probability of scenario \( k \)
- \( R_k^i \): Random yield of planting product type \( i \) under scenario \( k \)

### Decision Variables
- \( x_{it} \): Hectares of land allocated to planting product type \( i \) in period \( t \)
- \( y_{itk} \): Amount of product type \( i \) purchased at the end of period \( t \) under scenario \( k \)
- \( w_{itk} \): Amount of product type \( i \) sold at the end of period \( t \) under scenario \( k \)

## Mathematical Model

The mathematical model solved using these algorithms is a simple stochastic farmer yield problem, which is given as follows:

$$
\min \left( \sum_{i=1}^{3} \sum_{t=1}^{2} C_i x_{it} + \sum_{k} p_k \left( \sum_{i=1}^{2} \sum_{t=1}^{2} B_i y_{itk} - \sum_{i=1}^{4} \sum_{t=1}^{2} S_i w_{itk} \right) \right)
$$

subject to:

$$
\begin{align}
(2) & \quad \sum_{i=1}^{3} x_{it} \leq 500 \quad \forall t \\
(3) & \quad R_k^i x_{it} + y_{itk} - w_{itk} \geq d_i \quad \forall i \in \{1, 2\}, \, \forall t, \, \forall k \\
(4) & \quad R_k^3 x_{3,t} \geq w_{3,t,k} + w_{4,t,k} \quad \forall t, \, \forall k \\
(5) & \quad w_{3,t,k} \leq 6000 \quad \forall t, \, \forall k \\
(6) & \quad x_{3,1} + x_{3,2} \leq 500 \\
(7) & \quad x_{it}, y_{itk}, w_{itk} \geq 0 \quad \forall i, \, \forall t, \, \forall k
\end{align}
$$

## Constraints Description

In the multi-stage farmer problem with crop rotation, the objective is to minimize the total system cost, which includes planting, purchasing, and subtracting the revenue from sales, as given in the objective function (1). 

- Constraint (2) ensures that the available land for planting in each period is 500 hectares.
- Constraint (3) guarantees the balanced relationship between the land yield, the amount purchased and sold, and the demand for each product (wheat and corn) in each period.
- Constraint (4) represents the same balance relationship as constraint (3) for sugar beet, for which purchasing is impossible.
- Constraint (5) indicates the limit on high-price sugar beet sales capacity.
- Constraint (6) states that no more than 500 hectares can be allocated to sugar beet over the two planting periods.
- Finally, constraint (7) specifies the non-negativity of the decision variables.

## Algorithms

### Single Cut L-Shape Method

The Single Cut L-shape method is a decomposition algorithm for solving two-stage stochastic linear programs. This method iteratively refines the solution by adding cuts to approximate the recourse function.

### Multi-Cut L-Shape Method

The Multi-Cut L-shape method is an extension of the Single Cut method that generates multiple cuts per iteration, potentially leading to faster convergence by better approximating the recourse function.

### Regularized Decomposition

Regularized Decomposition introduces a regularization term to the decomposition algorithm to improve convergence properties and stability. It is particularly useful for large-scale stochastic programming problems.

### Nested Decomposition

Nested Decomposition is an approach used to solve multi-stage stochastic programming problems. It extends the L-shape method by considering multiple stages of decisions and uncertainties.

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

Each algorithm has its script in the repository. You can run these scripts to see how the algorithms work and to solve, for example, stochastic programming problems.
