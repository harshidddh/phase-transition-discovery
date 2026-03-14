import numpy as np

def metropolis_step(grid: np.ndarray, T: float) -> np.ndarray:
    """
    Performs one full Monte Carlo sweep using the Metropolis-Hastings algorithm.
    
    Args:
        grid (np.ndarray): The current state of the 2D lattice.
        T (float): The current temperature of the system.
        
    Returns:
        np.ndarray: The updated lattice after one sweep.
    """
    N = grid.shape[0]
    for _ in range(N * N):
        x, y = np.random.randint(0, N, 2)
        S = grid[x, y]
        
        # Periodic boundary conditions
        neighbors = grid[(x+1)%N, y] + grid[(x-1)%N, y] + \
                    grid[x, (y+1)%N] + grid[x, (y-1)%N]
        
        dE = 2 * S * neighbors
        
        # Metropolis acceptance criterion
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            grid[x, y] = -S
            
    return grid