import numpy as np
from typing import Tuple

def initialize_lattice(N: int) -> np.ndarray:
    """
    Initializes a 2D square lattice with random spins (+1 or -1).
    
    Args:
        N (int): The dimension of the N x N grid.
        
    Returns:
        np.ndarray: An N x N array of spins.
    """
    return np.random.choice([-1, 1], size=(N, N))

def calculate_energy(grid: np.ndarray) -> float:
    """
    Calculates the total energy of the lattice using periodic boundary conditions.
    
    Args:
        grid (np.ndarray): The 2D lattice of spins.
        
    Returns:
        float: Total energy of the system.
    """
    energy = 0
    N = grid.shape[0]
    for i in range(N):
        for j in range(N):
            S = grid[i, j]
            neighbors = grid[(i+1)%N, j] + grid[(i-1)%N, j] + \
                        grid[i, (j+1)%N] + grid[i, (j-1)%N]
            energy += -S * neighbors
    return energy / 2.0  # Divide by 2 to avoid double counting

def calculate_magnetization(grid: np.ndarray) -> float:
    """
    Calculates the absolute magnetization per spin.
    
    Args:
        grid (np.ndarray): The 2D lattice of spins.
        
    Returns:
        float: Absolute magnetization bounded between [0, 1].
    """
    return np.abs(np.mean(grid))