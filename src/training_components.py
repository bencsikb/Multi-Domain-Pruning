import torch

from torch.optim import Optimizer
from utils.losses import LogCoshLoss


def get_loss_function(type: str) : #TODO ret type
        
        if type == "logcosh":
            loss_func = LogCoshLoss()
        
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