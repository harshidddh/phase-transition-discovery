import numpy as np

def magnetization(lattice):
    """
    Compute magnetization of lattice
    """
    return np.sum(lattice) / lattice.size

def susceptibility(magnetizations, temperature, lattice_size):
    """
    Magnetic susceptibility from magnetization fluctuations
    """
    N = lattice_size**2
    m_mean = np.mean(magnetizations)
    m2_mean = np.mean(magnetizations**2)

    chi = (N / temperature) * (m2_mean - m_mean**2)

    return chi
