import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_heat_equation_data(alpha=0.1, L=1.0, T=1.0, n_points=100):
    """
    Generate training data for 1D heat equation with analytical solution.
    """
    # Spatial and temporal domains
    x = np.linspace(0, L, n_points)
    t = np.linspace(0, T, n_points)
    X, T_mesh = np.meshgrid(x, t)
    
    # Analytical solution
    u_analytical = np.exp(-alpha * np.pi**2 * T_mesh) * np.sin(np.pi * X)
    
    # Generate training points
    # Boundary conditions (x=0 and x=L)
    x_bc = np.concatenate([
        np.zeros(n_points),  # x=0 boundary
        np.ones(n_points)    # x=L boundary
    ])
    t_bc = np.concatenate([t, t])
    u_bc = np.zeros(2 * n_points)  # u=0 at boundaries
    
    # Initial condition (t=0)
    x_ic = x
    t_ic = np.zeros(n_points)
    u_ic = np.sin(np.pi * x_ic)  # u(x,0) = sin(πx)
    
    # Collocation points (random points in domain)
    n_col = 2000
    x_col = np.random.uniform(0, L, n_col)
    t_col = np.random.uniform(0, T, n_col)
    
    # Convert to PyTorch tensors
    # Note: We don't set requires_grad here for data tensors
    # The physics_loss method will handle it
    data_dict = {
        'boundary_data': (
            torch.tensor(x_bc.reshape(-1, 1), dtype=torch.float32),
            torch.tensor(t_bc.reshape(-1, 1), dtype=torch.float32), 
            torch.tensor(u_bc.reshape(-1, 1), dtype=torch.float32)
        ),
        'initial_data': (
            torch.tensor(x_ic.reshape(-1, 1), dtype=torch.float32),
            torch.tensor(t_ic.reshape(-1, 1), dtype=torch.float32),
            torch.tensor(u_ic.reshape(-1, 1), dtype=torch.float32)
        ),
        'collocation_points': (
            torch.tensor(x_col.reshape(-1, 1), dtype=torch.float32),
            torch.tensor(t_col.reshape(-1, 1), dtype=torch.float32)
        ),
        'analytical_solution': (X, T_mesh, u_analytical)
    }
    
    return data_dict

def plot_results(model, analytical_data, epoch=0, save_path=None):
    """
    Plot comparison between PINN prediction and analytical solution.
    """
    X, T_mesh, u_analytical = analytical_data
    
    # Create grid for prediction
    x_flat = X.flatten().reshape(-1, 1)
    t_flat = T_mesh.flatten().reshape(-1, 1)
    xt = torch.tensor(np.hstack([x_flat, t_flat]), dtype=torch.float32)
    
    # Predict with model
    with torch.no_grad():
        u_pred = model(xt).numpy().reshape(X.shape)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 3D surface - Analytical
    ax = axes[0, 0]
    surf = ax.plot_surface(X, T_mesh, u_analytical, cmap='viridis', alpha=0.8)
    ax.set_title('Analytical Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    
    # 3D surface - PINN
    ax = axes[0, 1]
    surf = ax.plot_surface(X, T_mesh, u_pred, cmap='viridis', alpha=0.8)
    ax.set_title('PINN Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('t') 
    ax.set_zlabel('u(x,t)')
    
    # Error surface
    ax = axes[1, 0]
    error = np.abs(u_analytical - u_pred)
    surf = ax.plot_surface(X, T_mesh, error, cmap='hot', alpha=0.8)
    ax.set_title('Absolute Error')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('Error')
    
    # Slice at specific times
    ax = axes[1, 1]
    time_indices = [0, len(T_mesh)//4, len(T_mesh)//2, 3*len(T_mesh)//4, -1]
    colors = ['b', 'r', 'g', 'c', 'm']
    
    for i, idx in enumerate(time_indices):
        t_val = T_mesh[idx, 0]
        ax.plot(X[0], u_analytical[idx], '--', color=colors[i], 
                label=f'Analytical, t={t_val:.2f}')
        ax.plot(X[0], u_pred[idx], '-', color=colors[i], 
                label=f'PINN, t={t_val:.2f}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Time Slices Comparison')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return u_pred, error