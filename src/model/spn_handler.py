import os
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple        
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils.losses import LogCoshLoss
from utils.common_utils import set_seed, denormalize
from utils.tensorboard_handler import TensorboardHandler
from state_predictor.model import SPN, SPNMultihead, SPNTransformer
from state_predictor.metrics import calculate_metrics, calculate_transformer_metrics


class SPNHandler:
    def __init__(self, conf, run_name: str, tb_handler: TensorboardHandler) -> None:

        self._conf = conf
        self._run_name = run_name
        self._tb_handler = tb_handler
        self._log_dir_path = os.path.join(conf.save.root, run_name)
        self._model_conf = conf.model
        self._device = self._conf.train.device

        self._label_keys = {"spars": 0, "dmap": 1} 

        self._model = None
        set_seed(self._conf.train.seed)
                
    
    def create(self, is_pretrained = False) -> None:

        input_size = self._model_conf.n_prunable_layers * len(self._model_conf.state_features)

        # Select model type
        if self._model_conf.type == "basic":
            self._model = SPN(input_size, self._model_conf.output_size)
        elif self._model_conf.type == "multihead":
            self._model = SPNMultihead(input_size)
        elif self._model_conf.type == "transformer":
            self._model = SPNTransformer(
                input_dim=3, 
                model_dim=self._model_conf.model_dim, 
                num_heads=self._model_conf.num_heads, 
                num_layers=self._model_conf.num_layers, 
                dropout=self._model_conf.dropout,
            )
        else:
            print(f"Invalid model type {self._model_conf.type}!")
 

        self._model.to(self._device)

        self._freeze_model_parts_if_specified()

        self._optimizer = self._get_optimizer()
        self._loss_func = self._get_loss_function()
        self._lr_scheduler = self._get_lr_scheduler()  

        self._loss_func.to(self._device)

        if len(self._model_conf.pretrained) or is_pretrained:
            log_path = self._log_dir_path if is_pretrained else self._model_conf.pretrained 
            self.load_checkpoint(log_path)        
        
    
    def _freeze_model_parts_if_specified(self):
        if getattr(self._model_conf, "do_freeze_backend", False):
            for param in self._model.backend.parameters():
                param.requires_grad = False

        if getattr(self._model_conf, "do_freeze_head_spars", False):
            for param in self._model.head_spars.parameters():
                param.requires_grad = False

        if getattr(self._model_conf, "do_freeze_head_dmap", False):
            for param in self._model.head_dmap.parameters():
                param.requires_grad = False
    

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self._epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._lr_scheduler.state_dict(),
            'loss': self._loss_func,
        }
        torch.save(checkpoint, os.path.join(self._log_dir_path, "checkpoint.pt"))
    
    
    def load_checkpoint(self, path):

        checkpoint_path = os.path.join(path, "checkpoint.pt")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self._model.load_state_dict(checkpoint['model_state_dict'])
        if self._model_conf.do_resume:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self._epoch = checkpoint.get('epoch', 0)
            self._loss_func = checkpoint.get('loss', self._loss_func)


    def _get_loss_function(self) : #TODO ret type
        
        if self._model_conf.loss == "logcosh":
            loss_func = LogCoshLoss()
        elif self._model_conf.loss == "mse":
            loss_func = nn.MSELoss()
        elif self._model_conf.loss == "l1":
            loss_func = nn.L1Loss()
        
        return loss_func
    
    def _get_optimizer(self) -> Optimizer:
        trainable_params = filter(lambda p: p.requires_grad, self._model.parameters())

        if self._model_conf.optimizer == "adam":
            optimizer = torch.optim.Adam(trainable_params, 
                                        lr=self._model_conf.start_lr,
                                        weight_decay=self._model_conf.weight_decay)
        elif self._model_conf.optimizer == "sgd":
            optimizer = torch.optim.SGD(trainable_params,
                                        lr=self._model_conf.start_lr,
                                        momentum=self._model_conf.momentum,
                                        weight_decay=self._model_conf.weight_decay)
        elif self._model_conf.optimizer == "adamw":
            optimizer = torch.optim.AdamW(trainable_params,
                                        lr=self._model_conf.start_lr,
                                        weight_decay=self._model_conf.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self._model_conf.optimizer}")

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
            running_spars_loss = 0.0
            running_dmap_loss = 0.0
            running_metrics = {key: {} for key in self._label_keys}

            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}")

            for batch_i, (data_gt, label_gt) in pbar:
                data_gt = data_gt.type(torch.float32).to(self._device)
                label_gt = label_gt.type(torch.float32).to(self._device)

                self._optimizer.zero_grad()
                outs = self._model(data_gt)

                spars_loss = self._loss_func(outs[:,:,0], label_gt[:,:,0])
                dmap_loss = self._loss_func(outs[:,:,1], label_gt[:,:,1])
                loss = spars_loss + dmap_loss
                weighted_loss = self._conf.model.spars_loss_weight * spars_loss + self._conf.model.dmap_loss_weight * dmap_loss

                weighted_loss.backward()
                self._optimizer.step()

                running_loss += loss.cpu().item()
                running_spars_loss += spars_loss.cpu().item()
                running_dmap_loss += dmap_loss.cpu().item()
                running_metrics = self._calculate_metrics(outs, label_gt, running_metrics)

                pbar.set_postfix(loss=f"{loss.cpu().item():.4f}", spars_loss=f"{spars_loss.cpu().item():.4f}", dmap_loss=f"{dmap_loss.cpu().item():.4f}")

            # Average loss and metrics
            self._train_loss = running_loss / len(train_dataloader)
            self._train_spars_loss = running_spars_loss / len(train_dataloader)
            self._train_dmap_loss = running_dmap_loss / len(train_dataloader)
            self._train_metrics = self._average_metrics(running_metrics, train_dataloader)

            # Validation
            if epoch % self._conf.train.val_freq == 0:
                self._val_loss, self._val_metrics = self.evaluate(val_dataloader)

            # Logging and checkpoint
            self._epoch = epoch
            self._log_results_to_tensorboard()
            self.save_checkpoint()
            self._lr_scheduler.step()

            epoch += 1

    def evaluate(self, dataloader: DataLoader) -> Tuple:
        self._model.eval()

        running_loss = 0.0
        running_spars_loss = 0.0
        running_dmap_loss = 0.0
        # Initialize metric containers for each label key
        running_metrics = {key: {} for key in self._label_keys}

        for batch_i, (data_gt, label_gt) in enumerate(dataloader):
            data_gt = data_gt.type(torch.float32).to(self._device)
            label_gt = label_gt.type(torch.float32).to(self._device)

            with torch.no_grad():
                outs = self._model(data_gt)
                spars_loss = self._loss_func(outs[:,:,0], label_gt[:,:,0])
                dmap_loss = self._loss_func(outs[:,:,1], label_gt[:,:,1])
                loss = spars_loss + dmap_loss

            running_loss += loss.cpu().item()
            # running_spars_loss += spars_loss.cpu().item()
            # running_dmap_loss += dmap_loss.cpu().item()
            running_metrics = self._calculate_metrics(outs, label_gt, running_metrics)

        # Compute average loss and metrics over the dataset
        running_loss /= len(dataloader)

        running_metrics = self._average_metrics(running_metrics, dataloader)     

        return running_loss, running_metrics
    
    
    def predict(self, data_gt: torch.Tensor) -> Tuple:
        """
        Args:
            data_gt: shape (T, 3) — a single input sequence with [spars, dmap, alpha]
        
        Returns:
            pred_spars, pred_dmap: denormalized prediction for the next step (spars_T, dmap_T)
        """
        self._model.eval()

        # Add batch dimension: (1, T, 3)
        if len(data_gt.shape) == 2:
            data_gt = data_gt.unsqueeze(0).type(torch.float32).to(self._device)

        with torch.no_grad():
            outs = self._model(data_gt)  # (1, T, 2)
            
        #print(outs)
        last_pred = outs[:, -1, :]  # shape: (2,)

        pred_spars = denormalize(last_pred[:, 0], value_range=(0, 1))
        pred_dmap = denormalize(last_pred[:, 1], value_range=(0, 1))

        return pred_spars, pred_dmap

    def autoregressive_predict(self, alpha_seq, init_spars, init_dmap):
        """
        Predicts next values autoregressively using Transformer SPN.
        Args:
            alpha_seq: Tensor of shape (T,) → alphas for each step
            init_spars, init_dmap: float values for the first input
        Returns:
            Tensor of shape (T, 2) → predicted [spars, dmap] at each step
        """
        self._model.eval()

        device = self._device
        preds = []
        history = []

        # Initial input
        spars, dmap = init_spars, init_dmap

        for alpha in alpha_seq:
            x = torch.tensor([alpha.item(), spars, dmap], dtype=torch.float32).to(device)
            history.append(x)

            input_seq = torch.stack(history).unsqueeze(0)  # shape (1, T_so_far, 3)

            with torch.no_grad():
                out = self._model(input_seq)  # (1, T_so_far, 2)
                next_vals = out[0, -1].cpu()  # (2,)

            preds.append(next_vals)
            spars, dmap = next_vals.tolist()

        return torch.stack(preds)  # (T, 2)

   

    
    
    def _calculate_metrics(self, outs, label_gt, running_metrics):
        for key, idx in self._label_keys.items():
            metrics_dict = calculate_transformer_metrics(outs[:, idx:idx+1], label_gt[:, idx:idx+1])

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

    
    def _log_results_to_tensorboard(self):

        # Learning rate
        lr = self._optimizer.param_groups[0]['lr']
        self._tb_handler.log_scalar(lr, self._epoch, name="learning_rate", tag_ext="train")

        # Train losses
        self._tb_handler.log_scalar(self._train_loss, self._epoch, name="loss", tag_ext="train")
        self._tb_handler.log_scalar(self._train_spars_loss, self._epoch, name="spars_loss", tag_ext="train")
        self._tb_handler.log_scalar(self._train_dmap_loss, self._epoch, name="dmap_loss", tag_ext="train")

        # Val loss
        self._tb_handler.log_scalar(self._val_loss, self._epoch, name="loss", tag_ext="val")
        
        # Train & val metrics
        self._tb_handler.log_dict_as_scalars(self._train_metrics, self._epoch, tag_ext="train")
        self._tb_handler.log_dict_as_scalars(self._val_metrics, self._epoch, tag_ext="val")


    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> str:
        return self._device