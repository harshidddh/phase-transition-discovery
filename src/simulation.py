import numpy as np
from src.metropolis import metropolis_step

def run_thermalization(grid: np.ndarray, T: float, steps: int) -> np.ndarray:
    """
    Advances the system to thermal equilibrium.
    
    Args:
        grid (np.ndarray): The initial lattice.
        T (float): System temperature.
        steps (int): Number of sweeps to perform.
        
    Returns:
        np.ndarray: The thermalized lattice.
    """
    for _ in range(steps):
        grid = metropolis_step(grid, T)
    return grid

def generate_configurations(grid: np.ndarray, T: float, n_samples: int, sample_interval: int) -> list:
    """
    Samples microstates from the system after it has reached equilibrium.
    
    Args:
        grid (np.ndarray): The thermalized lattice.
        T (float): System temperature.
        n_samples (int): Number of independent configurations to capture.
        sample_interval (int): Sweeps between captures to reduce autocorrelation.
        
    Returns:
        list: A list of flattened 1D arrays representing the microstates.
    """
    samples = []
    for _ in range(n_samples):
        for _ in range(sample_interval):
            grid = metropolis_step(grid, T)
        samples.append(grid.flatten())
    return samples