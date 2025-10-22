import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving PDEs.
    
    Args:
        layers (list): List of layer sizes [input_dim, hidden1, ..., hiddenN, output_dim]
        activation (str): Activation function ('tanh', 'relu', 'sigmoid')
    """
    def __init__(self, layers, activation='tanh'):
        super(PINN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.activation = self._get_activation(activation)
        
        # Build the network
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
    
    def _get_activation(self, activation):
        """Get activation function"""
        if activation == 'tanh':
            return torch.tanh
        elif activation == 'relu':
            return torch.relu
        elif activation == 'sigmoid':
            return torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        """Forward pass through the network"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on output layer
                x = self.activation(x)
        return x

class HeatEquationPINN(PINN):
    """
    Specialized PINN for 1D Heat Equation: u_t = alpha * u_xx
    """
    def __init__(self, layers, alpha=0.1):
        super().__init__(layers)
        self.alpha = alpha  # Thermal diffusivity
        
    def physics_loss(self, x, t):
        """
        Compute physics residual: u_t - alpha * u_xx
        
        Args:
            x: Spatial coordinates (requires_grad=True)
            t: Temporal coordinates (requires_grad=True)
        """
         # Ensure inputs require gradients
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        
        # Combine inputs
        xt = torch.cat([x, t], dim=1)
        
        # Forward pass
        u = self.forward(xt)
        
        # Compute gradients: u_t = ∂u/∂t
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute gradients: u_x = ∂u/∂x
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute second derivative: u_xx = ∂²u/∂x²
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]
        
        # Physics residual: u_t - alpha * u_xx
        residual = u_t - self.alpha * u_xx
        """# Combine inputs
        xt = torch.cat([x, t], dim=1)
        xt.requires_grad_(True)
        
        # Forward pass
        u = self.forward(xt)
        
        # Compute gradients: u_t = ∂u/∂t
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute gradients: u_x = ∂u/∂x
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute second derivative: u_xx = ∂²u/∂x²
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]
        
        # Physics residual: u_t - alpha * u_xx
        residual = u_t - self.alpha * u_xx"""
        
        return residual, u