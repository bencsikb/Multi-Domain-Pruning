import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from utils.tensorboard_handler import TensorboardHandler

from state_predictor.model import SPN
from state_predictor.utils import calculate_metrics
from utils.losses import LogCoshLoss


class SPNHandler:
    def __init__(self, conf, log_dir) -> None:

        self._conf = conf
        self._log_dir_path = log_dir
        self._model_conf = conf.model
        self._device = self._conf.train.device

        self._label_keys = {"spars": 0, "dmap": 0} # dummy

        self._model = None
        self._tb_handler = TensorboardHandler(log_dir=self._log_dir_path)
                
    
    def create(self, checkpoint_path=None) -> None:

        if checkpoint_path is not None:
            self.load_checkpoint()
        else:
            self._model = SPN(self._model_conf.input_size, self._model_conf.output_size)
            self._optimizer = self._get_optimizer()
            self._loss_func = self._get_loss_function()
            self._lr_scheduler = self._get_lr_scheduler()     

        self._model.to(self._device)


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

    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        epochs = self._conf.train.epochs
        epoch = 0

        while epoch < epochs:         
            self._model.train()

            running_loss = 0.0
            # Properly initialize running_metrics with sub-dicts for each label key
            running_metrics = {key: {} for key in self._label_keys}

            for batch_i, (data_gt, label_gt) in enumerate(train_dataloader):
                data_gt = data_gt.type(torch.float32).to(self._device)
                label_gt = label_gt.type(torch.float32).to(self._device)

                self._optimizer.zero_grad()
                outs = self._model(data_gt)
                loss = self._loss_func(outs[0], label_gt[0]) + self._loss_func(outs[1], label_gt[1])
                loss.backward()
                self._optimizer.step()

                running_loss += loss.cpu().item()
                running_metrics = self._calculate_metrics(outs, label_gt, running_metrics)

            # Average loss and metrics
            self._train_loss = running_loss / len(train_dataloader)
            self._train_metrics = self._average_metrics(running_metrics, train_dataloader)           

            # Validation
            if epoch % self._conf.train.val_freq == 0:
                self._val_loss, self._val_metrics = self.evaluate(val_dataloader)

            # Tensorboard logging
            self._epoch = epoch
            self._log_results_to_tensorboard()
            self.save_checkpoint()

            # Epoch step
            self._lr_scheduler.step()
            epoch += 1

    def evaluate(self, dataloader: DataLoader) -> tuple:
        self._model.eval()

        running_loss = 0.0
        # Initialize metric containers for each label key
        running_metrics = {key: {} for key in self._label_keys}

        for batch_i, (data_gt, label_gt) in enumerate(dataloader):
            data_gt = data_gt.type(torch.float32).to(self._device)
            label_gt = label_gt.type(torch.float32).to(self._device)

            with torch.no_grad():
                outs = self._model(data_gt)
                loss = self._loss_func(outs, label_gt)

            running_loss += loss.cpu().item()
            running_metrics = self._calculate_metrics(outs, label_gt, running_metrics)

        # Compute average loss and metrics over the dataset
        running_loss /= len(dataloader)
        running_metrics = self._average_metrics(running_metrics, dataloader)     

        return running_loss, running_metrics

    
    
    def _calculate_metrics(self, outs, label_gt, running_metrics):
        for key, idx in self._label_keys.items():
            metrics_dict = calculate_metrics(outs[:, idx:idx+1], label_gt[:, idx:idx+1])

            # Initialize nested metric names if missing
            for metric_name, value in metrics_dict.items():
                if metric_name not in running_metrics[key]:
                    running_metrics[key][metric_name] = 0.0
                running_metrics[key][metric_name] += value
        
        return running_metrics

    def _average_metrics(self, running_metrics, dataloader):
        # Should be called at the end of the epoch
        
        for key in running_metrics:
                for metric_name in running_metrics[key]:
                    running_metrics[key][metric_name] /= len(dataloader)
        
        return running_metrics
    

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self._epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._lr_scheduler.state_dict(),
            'loss': self._loss_func,
        }
        torch.save(checkpoint, os.path.join(self._log_dir_path, "checkpoint"))
    
    
    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self._log_dir_path, "checkpoint")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self._epoch = checkpoint.get('epoch', 0)
        self._loss_func = checkpoint.get('loss', self._loss_func)
    
    def _log_results_to_tensorboard(self):

        self._tb_handler.log_scalar(self._train_loss, self._epoch, name="loss", tag_ext="train")
        self._tb_handler.log_scalar(self._val_loss, self._epoch, name="loss", tag_ext="val")
        self._tb_handler.log_dict_as_scalars(self._train_metrics, self._epoch, tag_ext="train")
        self._tb_handler.log_dict_as_scalars(self._val_metrics, self._epoch, tag_ext="val")


    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> str:
        return self._device





    



