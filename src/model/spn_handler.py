import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from state_predictor.model import SPN
from utils.losses import LogCoshLoss


class SPNHandler:
    def __init__(self, conf) -> None:

        self._conf = conf
        self._model_conf = conf.model
        self._device = self._conf.train.device

        self._model = None

                
    
    def _load_pretrained(self, weights) -> nn.Module:
        pass

    def _get_loss_function(self) : #TODO ret type
        
        if self._model_conf.loss == "logcosh":
            loss_func = LogCoshLoss()
        
        return loss_func
    
    def _get_optimizer(self) -> Optimizer:

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


    def _get_lr_scheduler(self):
        
        if self._model_conf.lr_scheduler == "cos":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer,
                                                                  T_max=self._conf.train.epochs,
                                                                  eta_min=0.000005,
                                                                  last_epoch=-1)

        return lr_sched

    
    def create(self) -> None:

        if self._model_conf.pretrained is not None:
            self._model = self._load_pretrained(self._model_conf.pretrained)
        else:
            self._model = SPN(self._model_conf.input_size, self._model_conf.output_size)
            self._optimizer = self._get_optimizer()
            self._loss_func = self._get_loss_function()
            self._lr_scheduler = self._get_lr_scheduler()     

        self._model.to(self._device)

    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        
        epochs = self._conf.train.epochs

        epoch = 0

        while epoch < epochs:         
                  
            self._model.train()

            running_loss = 0
            cnnt = 0


            for batch_i, (data_gt, label_gt) in enumerate(train_dataloader):
                print(f"batch {batch_i}")
                
                data_gt = data_gt.type(torch.float32).to(self._device)
                label_gt = label_gt.type(torch.float32).to(self._device)

                self._optimizer.zero_grad()
                outs = self._model(data_gt)
                loss = self._loss_func(outs, label_gt)
                loss.backward()
                self._optimizer.step()

                running_loss += loss.cpu().item()
                print(f"{running_loss = }")


            # Calculate training metrics
            running_loss /= len(train_dataloader)


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





    



