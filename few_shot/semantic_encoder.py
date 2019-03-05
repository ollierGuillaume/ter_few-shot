import torch
from torch.nn import Module
from typing import Callable, List, Union
from torch.optim import Optimizer

def hamming(x: torch.Tensor, n: int, k: int):
    """Compute the mean of hamming distances between all samples of the same class in the set.
    # Arguments
        x: Query samples. A binary tensor of shape (n_x, d) where d is the embedding dimension

    TODO simplify
    """
    d_x = x.shape[1]
    sum = 0
    for i in range(0, x.shape[0], n):
        classe_x = x[i:i+n]
        sum += torch.sum(torch.Tensor([[torch.sum(a != b) for a in classe_x] for b in classe_x]))/d_x
    return sum/k


def semantic_loss(base_loss, n, k):
    def _loss(output, target, bin_x):
        return base_loss(output, target) + hamming(bin_x, n, k)
    return _loss

def gradient_step_semantic_encoder(model: Module, optimiser: Optimizer, semantic_loss: Callable, x: torch.Tensor, y: torch.Tensor, **kwargs):
    """Takes a single gradient step.

    # Arguments
        model: Model to be fitted
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples
        y: Input targets
    """
    model.train()
    optimiser.zero_grad()
    y_pred, bin_layer = model(x)
    loss = semantic_loss(y_pred, y, semantic_loss)
    loss.backward()
    optimiser.step()

    return loss, y_pred