import torch

from torch.optim import Optimizer
from utils.losses import LogCoshLoss


def get_loss_function(self) : #TODO ret type
        
        if self._model_conf.loss == "logcosh":
            loss_func = LogCoshLoss()
        
        return loss_func
    
def get_optimizer(self) -> Optimizer:

    if self._model_conf.optimizer == "adam":
        optimizer = torch.optim.Adam(self._model.parameters(), 
                                        lr=self._model_conf.start_lr,
                                        weight_decay=self._model_conf.weight_decay)
    elif self._model_conf.optimizer == "sgd":
        optimizer = torch.optim.SGD(self._model.parameters(),
                            lr=self._model_conf.start_lr,
                            momentum=self._model_conf.momentum,
                            weight_decay=self._model_conf.weight_decay)

    elif self._model_conf.optimizer == "adamw":
        optimizer = torch.optim.AdamW(self._model.parameters(),
                                lr=self._model_conf.start_lr,
                                weight_decay=self._model_conf.weight_decay)

    return optimizer


def get_lr_scheduler(self):
    
    if self._model_conf.lr_scheduler == "cos":
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer,
                                                                T_max=self._conf.train.epochs,
                                                                eta_min=0.000005,
                                                                last_epoch=-1)

    return lr_sched