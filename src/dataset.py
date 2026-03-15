import numpy as np
import os
from tqdm import tqdm
from src.ising_model import initialize_lattice
from src.simulation import run_thermalization, generate_configurations

def build_dataset(N: int, T_range: np.ndarray, therm_steps: int, n_samples: int, interval: int) -> tuple:
    """
    Iterates over a temperature range to build a full dataset of microstates.
    """
    X = []
    y_temp = []
    magnetization_data = []
    
    print(f"Generating dataset for {len(T_range)} temperatures...")
    for T in tqdm(T_range):
        grid = initialize_lattice(N)
        grid = run_thermalization(grid, T, therm_steps)
        
        configs, mags = generate_configurations(grid, T, n_samples, interval)
        X.extend(configs)
        y_temp.extend([T] * n_samples)
        magnetization_data.append(mags)
        
    return np.array(X), np.array(y_temp), np.array(magnetization_data)

def save_dataset(X: np.ndarray, y: np.ndarray, filepath: str = "data/"):
    """Saves the generated features and labels to disk."""
    os.makedirs(filepath, exist_ok=True)
    np.save(os.path.join(filepath, "X_configs.npy"), X)
    np.save(os.path.join(filepath, "y_temps.npy"), y)
    print(f"Dataset saved to {filepath}")