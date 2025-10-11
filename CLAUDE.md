# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Quantathon V2 hackathon challenge focused on implementing **Quantum Annealing for Robust Principal Component Analysis (QAPCA)** for financial data analysis. The challenge involves building L1-PCA classically and then implementing QAPCA-R to visualize natural groupings of stock tickers into sectors based on price movements.

**Key Paper:** [Quantum Annealing for Robust Principal Component Analysis](https://arxiv.org/pdf/2501.10431) (December 2024)

## Environment Setup

**Python Version:** 3.13

**Package Manager:** UV (uv.lock present)

**Install dependencies:**
```bash
uv sync
```

**Activate virtual environment:**
```bash
source .venv/bin/activate
```

## Working with Notebooks

The main work is done in Jupyter notebooks. To run notebooks:

```bash
jupyter notebook
```

**Primary notebooks:**
- `shaw_circle_challenge.ipynb` - Main challenge implementation (L1-PCA and QAPCA-R)
- `challenge_2.ipynb` - Additional challenge work
- `streamlit.ipynb` - Visualization/dashboard work

## Core Dependencies

- **yfinance** - Stock price data fetching
- **numpy** - Numerical operations
- **pandas** - Data manipulation
- **scipy** - Optimization algorithms (dual_annealing)
- **dwave-neal** - Simulated Annealing Sampler for quantum-inspired optimization
- **matplotlib** - Data visualization

## Data

Stock price data is stored in `data/prices.csv` (2017-01-01 to 2017-03-01). The data includes tickers from different sectors:
- **Technology:** AAPL, AMZN, GOOG, MSFT
- **Energy:** XOM
- **Finance:** GLD
- **Utility:** AEP, DUK, SO

## Architecture Overview

### L1-PCA Classical Implementation

The classical implementation follows this workflow:

1. **Data Preparation:** Calculate daily log returns from stock prices
2. **Covariance Matrix:** Compute `J = X @ X.T` where X is the feature matrix
3. **Optimization:** Solve the binary quadratic problem using `scipy.optimize.dual_annealing`:
   ```
   b_opt = argmin_{b ∈ {±1}^N} b^T(-J)b
   ```
4. **Component Extraction:** Recursively update covariance matrix to find orthogonal components
5. **Projection:** Project data onto principal components using `Phi` (nearest orthonormal matrix via SVD)

**Key Functions:**
- `solve_covariance_matrix(sample_data)` - Computes covariance matrix
- `l1_objective(b, J)` - L1 PCA objective function
- `Phi(T)` - Returns nearest orthonormal matrix via SVD
- `solve_l1_classical_component(J)` - Solves for single component using dual annealing
- `do_l1_pca(sample_data, K, get_component_func)` - Main PCA pipeline

### QAPCA-R Quantum-Inspired Implementation

The quantum annealing implementation:

1. **Ising Model Conversion:** Convert covariance matrix J to Ising coupling dictionary
2. **Quantum Annealing:** Use `neal.SimulatedAnnealingSampler` to solve Ising model
3. **Sample Selection:** Extract best sample from quantum annealer results
4. **Component Update:** Same recursive update as classical approach

**Key Functions:**
- `convert_J_to_ising_model(X)` - Converts covariance to Ising couplings
- `solve_l1_qapca_r_component(J)` - Solves for component using quantum annealing

### Recursive Component Update Formula

After finding component `b_opt`, the covariance matrix is updated:

```python
J_new = J - (2/r_norm_sqrd) * J @ bbT @ J + (J @ bbT @ J @ bbT @ J) / (r_norm_sqrd**2)
```

where `r_norm_sqrd = b_opt.T @ J @ b_opt` and `bbT = outer(b_opt, b_opt)`

This ensures subsequent components are orthogonal to previous ones.

## Important Implementation Details

1. **Binary Optimization:** The core problem is NP-hard binary quadratic optimization converted to Ising form
2. **Robustness:** The L1-norm formulation is more robust to outliers than traditional L2-PCA
3. **Neal Sampler:** Uses classical simulated annealing to approximate quantum annealing behavior
4. **Component Orthogonality:** Achieved through recursive covariance matrix updates, not explicit orthogonalization

## Challenge Workflow

1. **Step 0:** Download stock price data (cached in `data/prices.csv`)
2. **Step 1:** Prepare data - compute daily log returns
3. **Step 2:** Apply classical L1-PCA
4. **Step 3:** Analyze results - visualize sector groupings
5. **Step 4:** Construct Ising Model for QAPCA-R
6. **Step 5:** Analyze quantum results and compare with classical
7. **Bonus:** Extended experiments (larger datasets, outliers, custom annealing, multi-component PCA, real quantum hardware)

## Submission Requirements

- Public GitHub repository link
- Presentation (PowerPoint or PDF)
- **No edits allowed after hackathon deadline**

## Judging Criteria

- Fundamentals Understanding (20%)
- Creativity & Innovation (30%)
- Accuracy and Technical Soundness (25%)
- Code Quality & Reproducibility (10%)
- Presentation & Communication (15%)
