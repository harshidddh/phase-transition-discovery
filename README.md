# Unsupervised ML Discovery of Phase Transitions in the 2D Ising Model

## Project Overview
This project unites Statistical Mechanics and Machine Learning by showing how unsupervised dimensionality reduction and clustering algorithms can autonomously "discover" thermodynamic phase transitions without any prior physical labels.

The dataset is generated from scratch using a custom python script of the Metropolis-Hastings Markov Chain Monte Carlo (MCMC) algorithm to simulate the Ferromagnetic Ising Model

## Physics Engine/Data Generation
The system is modeled on an $N \times N$ square lattice with periodic boundary conditions. The Hamiltonian governing the interacting spins $s_i \in \{-1, +1\}$ is given by:

$$H = -J \sum_{\langle i, j \rangle} s_i s_j$$

Using the Metropolis-Hastings algorithm, the system is allowed to reach thermal equilibrium across a temperature sweep from $T=1.0$ to $T=4.0$. At each temperature, microstate configurations are flattened into high-dimensional vectors (e.g., $3600$ dimensions for a $60 \times 60$ lattice) to serve as the dataset for the ML pipeline.

## Machine Learning Pipeline
Rather than explicitly calculating physical observables to find the critical temperature $T_c$, I deployed Unsupervised Machine Learning to let the geometry of the data reveal the physics:
* **Dimensionality Reduction (PCA):** Principal Component Analysis was used to project the high-dimensional spin states into a 2D latent space. The first principal component (PC1) autonomously learned to represent the broken symmetry of the system, perfectly mimicking the physical Order Parameter (Magnetization).
* **Unsupervised Clustering (K-Means):** The latent projections were fed into a K-Means clustering algorithm ($k=3$). The algorithm successfully partitioned the data into Spin-Up (Ferromagnetic), Spin-Down (Ferromagnetic), and Disordered (Paramagnetic) phases without being fed temperature labels.

## Observability & Validation
To verify what the AI is actually predicting, we cross-checked the results against some classic statistical mechanics benchmarks:

1. **We superimposed the AI’s latent signal onto the theory’s order parameter. On a two-axis plot, the normalized PC1 vector correlated almost perfectly to the magnitude of the magnetization vector $\langle |M| \rangle$ over time. This shows a high degree of correlation.

2. **- We calculated the magnetic susceptibility via the rate of change of magnetization over time. The formula used is as follows:
     $$\chi = \frac{N^2}{T} (\langle |M|^2 \rangle - \langle |M| \rangle^2)$$
    The resulting plot correctly showed a massive divergence exactly at Lars Onsager's analytical solution for the critical temperature ($T_c \approx 2.269$).

## Technologies Used
* **Python:** Core simulation logic.
* **NumPy:** Vectorized lattice operations and efficient Boltzmann probability calculations.
* **Scikit-Learn:** PCA and K-Means implementation.
* **Matplotlib:** Visualization of phase transitions, latent spaces, and thermodynamic observables.