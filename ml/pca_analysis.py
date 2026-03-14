import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple

def perform_pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """
    Applies Principal Component Analysis to the dataset of microstates.
    
    Args:
        X (np.ndarray): Flattened configuration dataset (n_samples, N*N).
        n_components (int): Number of principal components to extract.
        
    Returns:
        Tuple[np.ndarray, PCA]: The transformed latent space and the fitted PCA object.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca