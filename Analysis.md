
# Step 3 & Step 5 Analysis — Quantum PCA for Financial Dataset

## Step 3 — Analysis of Classical L1-PCA Results

**What we did.**  
We computed daily **log-return vectors** per ticker, stacked them into a matrix $$X\in\mathbb{R}^{T\times N}$$ (time $$\times$$ tickers), formed the Gram/covariance-like matrix $$J = XX^\top$$ across tickers, and solved the **L1-PCA** $$K=1$$ problem by maximizing $$\|Xb\|_2$$ over binary $$b\in\{\pm1\}^N$$ (equivalently, minimizing $$-b^\top X^\top X b$$). The sign of each ticker’s embedding along the first L1 principal component partitions tickers into two groups.

> **Observed partition (L1-PC1):**  
> Group A: `AAPL, AMZN, GOOG` (large-cap tech/growth)  
> Group B: `AEP, DUK, SO` (utilities), `XOM` (energy), `GLD` (gold ETF / defensive)

**Do sectors naturally emerge?**  
Yes—**a clear sectoral/“risk-factor” split** emerges. The first L1 component separates **high beta, growth-oriented tech** names (AAPL/AMZN/GOOG) from a basket of **defensive/interest-rate-sensitive** (utilities), **commodity-linked** (XOM), and **safe-haven** (GLD) exposures. This is consistent with well-known market structure: tech names co-move with growth/market momentum; utilities and gold often load on different macro factors (rates/defensiveness), and energy loads on oil/commodity moves. The fact that GLD aligns with the second cluster is especially intuitive.

**Why PCA was needed to visualize this?**  
- The raw time series are **high-dimensional and noisy**. Their sector structure lives in their **co-movement patterns**, not in single-day returns.  
- PCA finds **directions of maximal shared variation**. With **L1-PCA**, we emphasize **alignment robustly** (sum of absolute projections) rather than variance alone, so the first component **extracts the dominant co-movement mode**.  
- The **sign of each ticker’s score** on PC1 yields a simple, interpretable **bipartition** that corresponds to sectors/factors. (Note: the global sign is arbitrary; flipping all signs leaves the partition unchanged.)

**What does “robust” mean here and why does it matter?**  
- **Robustness** means the solution is **less sensitive to outliers/heavy tails**. This is a critical feature for financial returns, which frequently show **non-Gaussian tails** and episodic jumps.  
- **L2-based PCA** can over-weight outlier days. On the other hand, **L1-PCA** reduces that sensitivity by optimizing absolute projections. The result is a **cleaner, more stable sector split** that better reflects persistent structure instead of one-off shocks.

---

## Step 5 — Analysis of QAPCA-R (Ising / Annealing) Results

**How Quantum Annealing (QA) works.**  
Quantum Annealing (QA) is a quantum optimization process based on the adiabatic theorem, governed by the time-dependent Schrödinger equation

$$
i\hbar \frac{\partial}{\partial t} \Psi(\mathbf{r}, t) = \hat{H(t)} \Psi(\mathbf{r}, t)
$$

where the system’s Hamiltonian evolves as

$$
H(t) = (1-s(t)) H_B + s(t)H_p, \quad \quad s(t) = \frac{t}{T}
$$


whose ground state represents the optimal solution. Quantum fluctuations induced by $$H_B$$ enable tunneling through high but narrow energy barriers, helping the system escape local minima. If the evolution is sufficiently slow and decoherence is minimal, the system ends close to the global minimum of $$H_P$$, yielding the optimal spin configuration $$b^*$$.

Starting from the known ground state of an easy Hamiltonian $$H_B$$, the system is evolved slowly so that it remains in its instantaneous ground state while the Hamiltonian is gradually transformed into the problem Hamiltonian $$H_P$$. The optimization problem is encoded as an **Ising Hamiltonian**,

$$
H(s) =  \sum_{i<j} J_{ij} s_i s_j + \sum_i h_i s_i, \quad \quad s_i \in \{\pm 1\}
$$

**What is the Ising model here?**  
For $$K=1$$, L1-PCA reduces to maximizing $$b^\top X^\top X b$$ over $$b\in\{\pm1\}^N$$.  
This is equivalent to **minimizing** $$-b^\top X^\top X b$$.  
We map that to an Ising model by setting pairwise couplings proportional to $$- (X^\top X)_{ij}$$ (diagonals can be absorbed or ignored for argmin). Sampling low-energy spin configurations $$s=b$$ approximately solves the binary optimization.

**Observed QAPCA-R partition (annealing result).** 
We again recover the **same sectoral split** as the classical L1-PCA run:  
`AAPL, AMZN, GOOG` vs. `AEP, DUK, SO, XOM, GLD`.  
That agreement indicates both methods found (or got close to) **the same dominant co-movement mode** for this dataset/period.

**Key differences: classical vs quantum route (results & implementation).**  
- **Search mechanism.** Classical (our dual-annealing) is a **stochastic thermal** optimizer in a continuous box relaxed to $$[-1,1]^N$$ and then snapped; QA attacks the **native discrete** $$\{\pm1\}^N$$ problem via **quantum/thermal sampling**.  
- **Landscape navigation.** QA can, in principle, **tunnel through tall/narrow barriers**, helping avoid some local minima; classical annealing relies on temperature schedules to jump valleys.  
- **Robustness to outliers.** Both are solving **L1-PCA’s robust objective**; robustness stems from the **L1 formulation**, not the solver.  
- **Determinism.** Neither route guarantees the **global optimum** on finite time/shots; both are **probabilistic**. That’s why we use **multiple reads/restarts** and check solution quality.  
- **Hardware path.** The Ising/QUBO form is **hardware-native** for annealers and also a good fit for **gate-model QAOA** via cost Hamiltonians; classical L1-PCA doesn’t provide that direct hardware mapping.

**Are the results guaranteed to be the same?**  
No. The objective is **non-convex and NP-hard**; different optimizers, seeds, anneal schedules, or noise levels can land on **different (near-)optima**. In this run, both methods produced **consistent sector partitions**, which boosts confidence—but equivalence is **not guaranteed** in general.

**Why QAPCA-R can be attractive to a financial team.**  
- **Interpretability:** Binary loadings $$b\in\{\pm1\}^N$$ give **crisp groupings** (sign-based clustering) that map naturally to **sector/factor views**.  
- **Robustness:** L1 formulation tempers outlier influence, making analyses **less whipsawed by tail events**.  
- **Scalability via hardware:** As ticker count grows, Ising/QUBO formulations let you **tap specialized hardware** (quantum annealers or QAOA on gate devices) for faster/broader exploration of the combinatorial space.  
- **Risk perspective:** The extracted components align with **macro risk factors** (growth/tech vs defensives/commodities), aiding **hedging, stress testing, and portfolio construction**.

---
