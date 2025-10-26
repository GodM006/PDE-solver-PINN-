# PINN PDE Solver

## Overview

The project demonstrates how a neural network can be trained to approximate solutions of the heat equation by minimizing a **combined loss** of observed data and the PDE residual.  

- **HeatEquationPINN**: Fully connected neural network taking `(x, t)` as input and predicting the solution `u(x, t)`.  
- **PINNLoss**: Computes the total loss as a combination of mean squared error on data points and the PDE residual.  
- **Training pipeline**: Uses Adam optimizer and supports GPU acceleration if available.  
- **Visualization**: Provides a function to plot the loss history over training.  

## Project Structure

```
PINN_PDE_solver/
│
├─ models.py           # Defines the PINN and HeatEquationPINN classes
├─ losses.py           # Defines PINNLoss class for combined data + physics loss
├─ utils.py            # Data generation and plotting functions
├─ train.py            # Main script to train the PINN
├─ requirements.txt    # Python dependencies
└─ README.md           # This file
```
