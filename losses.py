import torch

class PINNLoss:
    """
    Combined loss function for PINN: data_loss + physics_loss
    """
    def __init__(self, physics_weight=1.0):
        self.physics_weight = physics_weight
        self.mse_loss = torch.nn.MSELoss()
    
    def __call__(self, model, data_dict):
        """
        Compute total loss = data_loss + physics_weight * physics_loss
        
        Args:
            model: PINN model
            data_dict: Dictionary containing:
                - 'boundary_data': (x_bc, t_bc, u_bc) for boundary conditions
                - 'initial_data': (x_ic, t_ic, u_ic) for initial conditions  
                - 'collocation_points': (x_col, t_col) for physics residual
        """
        total_loss = 0.0
        
        # Boundary condition loss
        if 'boundary_data' in data_dict:
            x_bc, t_bc, u_bc_true = data_dict['boundary_data']
            u_bc_pred = model(torch.cat([x_bc, t_bc], dim=1))
            bc_loss = self.mse_loss(u_bc_pred, u_bc_true)
            total_loss += bc_loss
        
        # Initial condition loss  
        if 'initial_data' in data_dict:
            x_ic, t_ic, u_ic_true = data_dict['initial_data']
            u_ic_pred = model(torch.cat([x_ic, t_ic], dim=1))
            ic_loss = self.mse_loss(u_ic_pred, u_ic_true)
            total_loss += ic_loss
        
        # Physics residual loss
        if 'collocation_points' in data_dict:
            x_col, t_col = data_dict['collocation_points']
            residual, _ = model.physics_loss(x_col, t_col)
            physics_loss = torch.mean(residual**2)
            total_loss += self.physics_weight * physics_loss
        
        return total_loss