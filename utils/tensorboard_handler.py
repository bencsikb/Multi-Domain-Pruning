import os
from typing import Any, Dict
from torch.utils.tensorboard import SummaryWriter



class TensorboardHandler():
    def __init__(self, log_dir) -> None:
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_dict_as_scalars(self, metrics: dict, epoch: int, tag_ext: str) -> None:
        
        for key in metrics:
                for metric_name, value in metrics[key].items():
                    tag = f"{tag_ext}/{key}/{metric_name}"
                    self._writer.add_scalar(tag, value, epoch)

    def log_scalar(self, metric: Any, epoch: int, name: str, tag_ext: str) -> None: # TODO specify Any
        
        tag = f"{tag_ext}/{tag_ext}/{name}"
        self._writer.add_scalar(tag, metric, epoch)

    def log_hparams(self, hparams: Dict[str, Any], metric_dict) -> None:
        """
        Log hyperparameters using TensorBoard's add_hparams function.

        Args:
            hparams (dict): Dictionary of hyperparameters to log.
        """

        self._writer.add_hparams(hparam_dict=hparams, metric_dict=metric_dict)

    def close(self) -> None:
        self._writer.close()