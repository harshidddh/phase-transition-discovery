import numpy as np
import os
from src.dataset import build_dataset, save_dataset
from src.observables import susceptibility

def main():
    # Simulation Hyperparameters
    N = 30                  # Lattice size (N x N) - keeping it small for speed
    T_min, T_max = 1.0, 4.0 # Temperature bounds
    T_steps = 30            # Number of temperature points
    therm_steps = 2000       # Sweeps to reach equilibrium
    n_samples = 200          # Samples to capture per temperature
    interval = 10           # Sweeps between samples
    
    temps = np.linspace(T_min, T_max, T_steps)
    
    print("Initializing Monte Carlo Data Generation...")
    X, y, mag_data = build_dataset(N, temps, therm_steps, n_samples, interval)
    save_dataset(X, y)
    magnetization_curve = []
    susceptibility_curve = []

    for T, mags in zip(temps, mag_data):

        mags = np.array(mags)

        m_avg = np.mean(np.abs(mags))
        chi = susceptibility(mags, T, N)

        magnetization_curve.append(m_avg)
        susceptibility_curve.append(chi)

    os.makedirs("data", exist_ok=True)
    np.save("data/magnetization.npy", magnetization_curve)
    np.save("data/susceptibility.npy", susceptibility_curve)
    np.save("data/temperatures.npy", temps)
    
    print("Data generation complete. You can now run the analysis notebook.")

if __name__ == "__main__":
    main()