import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, loss_fn, optimizer, device='cpu'):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.loss_history = []
    
    def train(self, data_dict, epochs=10000, print_every=1000):
        """Training loop"""
        self.model.train()
        
        # Move data to device
        data_dict_device = {}
        for key, value in data_dict.items():
            if key == 'analytical_solution':
                data_dict_device[key] = value  # Leave as numpy arrays
            else:
                data_dict_device[key] = tuple(
                    tensor.to(self.device) for tensor in value
                )
        
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.optimizer.zero_grad()
            
            # Compute loss
            total_loss = self.loss_fn(self.model, data_dict_device)
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            self.loss_history.append(total_loss.item())
            
            if epoch % print_every == 0:
                pbar.set_description(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
    
    def plot_loss_history(self):
        """Plot training loss history"""
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss History')
        plt.grid(True)
        plt.show()

def train_heat_equation():
    """Complete training pipeline for heat equation"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    from utils import generate_heat_equation_data
    data_dict = generate_heat_equation_data(alpha=0.1, n_points=50)
    
    # Create model
    from models import HeatEquationPINN
    layers = [2, 50, 50, 50, 50, 1]  # Input: (x,t), Output: u
    model = HeatEquationPINN(layers, alpha=0.1)
    
    # Loss function and optimizer
    from losses import PINNLoss
    loss_fn = PINNLoss(physics_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Trainer
    trainer = Trainer(model, loss_fn, optimizer, device)
    
    # Train
    trainer.train(data_dict, epochs=10000, print_every=1000)
    
    # Plot loss
    trainer.plot_loss_history()
    
    return model, data_dict

if __name__ == "__main__":
    model, data_dict = train_heat_equation()