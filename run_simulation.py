import numpy as np
import os
from src.dataset import build_dataset, save_dataset

def main():
    # Simulation Hyperparameters
    N = 15                  # Lattice size (N x N) - keeping it small for speed
    T_min, T_max = 1.0, 4.0 # Temperature bounds
    T_steps = 30            # Number of temperature points
    therm_steps = 150       # Sweeps to reach equilibrium
    n_samples = 10          # Samples to capture per temperature
    interval = 5            # Sweeps between samples
    
    temps = np.linspace(T_min, T_max, T_steps)
    
    print("Initializing Monte Carlo Data Generation...")
    X, y = build_dataset(N, temps, therm_steps, n_samples, interval)
    save_dataset(X, y)
    print("Data generation complete. You can now run the analysis notebook.")

if __name__ == "__main__":
    main()