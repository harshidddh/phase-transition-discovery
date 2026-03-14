import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pca_clusters(X_pca: np.ndarray, labels: np.ndarray, save_path: str = None):
    """Visualizes the phase transition symmetry breaking in the 2D PCA space."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title("Symmetry Breaking: PCA Latent Space Clustering")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster ID")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

def plot_observability(temps: np.ndarray, pc1: np.ndarray, mag: np.ndarray, save_path: str = None):
    """Dual-axis plot proving the AI discovered the Order Parameter."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Temperature (T)')
    ax1.set_ylabel('Physical Magnetization |M|', color=color)
    ax1.scatter(temps, mag, color=color, alpha=0.5, label="Physics: Magnetization")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('AI Prediction: |PC1|', color=color)  
    ax2.scatter(temps, np.abs(pc1), color=color, alpha=0.5, marker='x', label="AI: Principal Component 1")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.axvline(x=2.269, color='k', linestyle='--', label='Critical Temp (Onsager)')
    plt.title("Observability: Unsupervised AI Recovers Thermodynamic Order Parameter")
    fig.tight_layout() 
    
    if save_path:
        plt.savefig(save_path)
    plt.show()