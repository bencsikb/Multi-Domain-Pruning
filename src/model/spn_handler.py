import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from state_predictor.model import SPN
from state_predictor.utils import calculate_metrics
from utils.losses import LogCoshLoss


class SPNHandler:
    def __init__(self, conf) -> None:

        self._conf = conf
        self._model_conf = conf.model
        self._device = self._conf.train.device

        self._label_keys = {"spars": 0, "dmap": 1}

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
            running_metrics = {"spars": np.zeros(5), "dmap": np.zeros(5)}

            for batch_i, (data_gt, label_gt) in enumerate(train_dataloader):
                
                data_gt = data_gt.type(torch.float32).to(self._device)
                label_gt = label_gt.type(torch.float32).to(self._device)

                self._optimizer.zero_grad()
                outs = self._model(data_gt)
                loss = self._loss_func(outs, label_gt)
                loss.backward()
                self._optimizer.step()

                running_loss += loss.cpu().item()
                for key, idx in self._label_keys.items():
                    running_metrics[key] += calculate_metrics(outs[:, idx:idx+1], label_gt[:, idx:idx+1])

            # Calculate training metrics
            running_loss /= len(train_dataloader)
            running_metrics = {k: v / len(train_dataloader) for k, v in running_metrics.items()}


            # Validation
            if epoch % self._conf.train.val_freq == 0:
                val_loss, val_metrics = self.evaluate(val_dataloader)

            # Logging
                

            print(f"{epoch = }, {running_loss = }, {val_loss = }")
            print(f"{running_metrics = }, {val_metrics = }")
            self._lr_scheduler.step()
            epoch += 1


    def evaluate(self, dataloader: DataLoader) -> list:

        self._model.eval()

        running_loss = 0
        running_metrics = {"spars": np.zeros(5), "dmap": np.zeros(5)}

        for batch_i, (data_gt, label_gt) in enumerate(dataloader):
                
            data_gt = data_gt.type(torch.float32).to(self._device)
            label_gt = label_gt.type(torch.float32).to(self._device)

            self._optimizer.zero_grad()
            outs = self._model(data_gt)
            loss = self._loss_func(outs, label_gt)

            running_loss += loss.cpu().item()
            for key, idx in self._label_keys.items():
                running_metrics[key] += calculate_metrics(outs[:, idx:idx+1], label_gt[:, idx:idx+1])

        # Calculate training metrics
        running_loss /= len(dataloader)
        running_metrics = {k: v / len(dataloader) for k, v in running_metrics.items()}

        return running_loss, running_metrics


    

    def save_metrics():
        pass


    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> str:
        return self._device





    



