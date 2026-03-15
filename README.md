# 2D Ising Model: Phase Transition Analysis via Unsupervised ML

This repository contains a computational pipeline that simulates the 2D Ising model using Markov Chain Monte Carlo (MCMC) and applies unsupervised machine learning (PCA and K-Means) to identify thermodynamic phase transitions.

## Project Overview
The objective of this project is to determine if dimensionality reduction and clustering algorithms can identify the critical phase transition and order parameter of a ferromagnet without prior integration of physical equations of state.

## Repository Structure
* `src/`: Physics engine (2D lattice initialization, Metropolis dynamics, data sampling).
* `ml/`: Machine learning modules (PCA dimensionality reduction, K-Means clustering).
* `experiments/`: Jupyter notebook for data analysis and physical validation.
* `figures/`: Visual outputs of the ML and physical analysis.
* `report/`: LaTeX manuscript detailing the theoretical background and methodology.

## Methodology
1. **Simulation (Data Generation):** A $30 \times 30$ lattice of interacting spins is simulated across a temperature range ($T=1.0$ to $T=4.0$). The system is thermalized for 2,000 sweeps using the Metropolis algorithm. 200 independent configurations are sampled per temperature step to generate the dataset.
2. **Machine Learning:** The spin configurations are flattened and processed using Principal Component Analysis (PCA). K-Means clustering is applied to the latent space to classify the system into distinct macroscopic states.
3. **Analysis:** The primary principal component (`PC1`) is evaluated against the computationally derived physical Magnetization ($\langle |M| \rangle$). 

## Results
* **Order Parameter:** The dominant latent variable (`PC1`) correlates directly with the physical magnetization of the system across the temperature gradient.
* **Critical Temperature Estimation:** By calculating the peak variance of `PC1` (serving as an analog to magnetic susceptibility), the unsupervised pipeline estimates a pseudo-critical temperature of $T_c \approx 2.241$ for the $N=30$ lattice. This is consistent with finite-size scaling expectations compared to the Onsager exact solution for an infinite lattice ($T_c \approx 2.269$).

## Usage
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt