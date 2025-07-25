import torch
from torch import nn

from torch.optim import Optimizer
from utils.losses import LogCoshLoss


def get_loss_function(type: str) : #TODO ret type
        
        if type == "logcosh":
            loss_func = LogCoshLoss()
        elif type == "l1":
             loss_func = nn.L1Loss()
        
        return loss_func
    
def get_optimizer(type: str, 
                  model: torch.nn.Module,
                  lr: float,
                  weight_decay: float,
                  momentum: float = None) -> Optimizer:

    if type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    elif type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def get_lr_scheduler(type: str, epochs: int, optimizer):
    
    if type == "cos":
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=epochs,
                                                                eta_min=0.000005,
                                                                last_epoch=-1)

    return lr_sched

import math

def general_cosine_scheduler(min_val, max_val, epochs, direction='down'):
    """
    Returns a list of values scheduled over `epochs` using a cosine curve.

    Args:
        min_val (float): Minimum value of the schedule.
        max_val (float): Maximum value of the schedule.
        epochs (int): Total number of steps (e.g., training epochs).
        direction (str): 'down' (default) or 'up'.

    Returns:
        List[float]: Scheduled values of length `epochs`.
    """
    assert direction in {'up', 'down'}, "Direction must be 'up' or 'down'"

    values = []
    for i in range(epochs):
        cosine = 0.5 * (1 + math.cos(math.pi * i / (epochs - 1)))
        val = min_val + (max_val - min_val) * cosine if direction == 'down' else \
              max_val - (max_val - min_val) * cosine
        values.append(val)

    return values
