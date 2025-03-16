import torch
import torch.nn as nn
from torch.optim import Optimizer

from state_predictor.model import SPN
from utils.losses import LogCoshLoss


class SPNHandler:
    def __init__(self, conf) -> None:

        self._conf = conf
        self._model_conf = conf.model
        self._device = self._conf.train.device

                
    
    def _load_pretrained(self, weights) -> nn.Module:
        pass

    def _get_loss_function(self) : #TODO ret type
        
        if self._model_conf.loss == "logcosh":
            loss_func = LogCoshLoss()
        
        return loss_func
    
    def _get_optimizer(self) -> Optimizer:

        if self._model_conf.optimizer == "adam":
            optimizer = torch.optim.Adam(self._model.parameters, 
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


    def _get_lr_scheduler(self):
        pass

    def create(self) -> None:

        if self._model_conf.pretrained is not None:
            self._model = self._load_pretrained(self._model_conf.pretrained)
        else:
            self._model = SPN(self._model_conf.input_size, self._model_conf.output_size)
            self._optimizer = self._get_optimizer()
            self._loss_func = self._get_loss_function()
            self._lr_scheduler = self._lr_scheduler()          



    
    def train(self):
        pass

    def evaluate(self) -> list:
        pass

    def save_metrics():
        pass


    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> str:
        return self._device





    



