import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []

        # zero_grad is for gradients, not needed here — use no_grad context instead
        with torch.no_grad():
            out = x

            for layer in model.children():
                out = layer(out)

                if isinstance(layer, nn.Linear):
                    stats.append({
                        "mean":          round(out.mean().item(), 4),
                        "std":           round(out.std().item(), 4),
                        #.all(dim=0) you first reduce across the batch dimension (dim=0), asking "is this neuron dead for every single sample?".
                        "dead_fraction": round(((out <= 0).all(dim=0)).float().mean().item(), 4),
                    })
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        stats = []
        model.zero_grad()

        logits = model(x)
        # MSELoss() creates the criterion, then call it with (pred, target)
        # not MSELoss(logits, y) directly
        loss = nn.MSELoss()(logits, y)
        loss.backward()


        for layer in model.children():
            if isinstance(layer, nn.Linear):
                # layer.weight.grad holds the gradient, not loss[i]
                grad = layer.weight.grad
                stats.append({
                    # average activation value across all neurons
                    "mean": round(grad.mean().item(), 4),
                    # spread of activation values — low std can indicate vanishing signals
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(),4),
                })
        return stats
        

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        # these are lists of dicts, not numpy arrays — can't use .max() directly
        # extract values manually
        dead_fractions  = [s["dead_fraction"] for s in activation_stats]
        grad_norms      = [s["norm"]          for s in gradient_stats]
        act_stds       = [s["std"]           for s in activation_stats]

        if max(dead_fractions) > 0.5:
            return 'dead_neurons'
        elif max(grad_norms) > 1000:
            return 'exploding_gradients'
        elif grad_norms[-1] < 1e-5:
            return 'vanishing_gradients'
        
        for i in act_stds:
            if i < 0.1:
                return 'vanishing_gradients'
            elif i > 10:
                return 'exploding_gradients'
        
        return 'healthy'
