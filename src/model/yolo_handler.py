import torch.nn as nn
import torch_pruning as tp
import torch
import numpy as np
import gc
import sys
import os
from thop import profile
# from fvcore.nn import FlopCountAnalysis
import torchvision.transforms as T
import copy
from typing import List, Optional

from ultralytics.nn.modules import Detect
from ultralytics.utils.loss import v8DetectionLoss 
from src.model.tp_utils import replace_c2f_with_c2f_v2
from ultralytics.models.yolo.detect import DetectionTrainer


class YoloHandler:
    def __init__(self, model_conf) -> None:

        # self._flattened_layers = []
        self._model_conf = model_conf
        self._device = model_conf.device
        self._example_input = torch.randn(1, 3, 224, 224) # TODO where should this come from?

        self._init_model = self._load_pretrained()
        self._model = None
        self._detmodel = None 

        self.reset_model()

    def _load_pretrained(self) -> nn.Module:
        from ultralytics import YOLOv10, YOLO

        if self._model_conf.pretrained_type == "yolov10":
            if self._model_conf.model_path is not None: 
                from ultralytics.nn.tasks import attempt_load_one_weight
                from ultralytics.models.yolov10.model import YOLOv10DetectionModel

                model = YOLOv10('yolov10n.yaml') #TODO why n? check!
                weights, ckpt = attempt_load_one_weight(self._model_conf.model_path)
                cfg = ckpt["model"].yaml    
                detmodel = YOLOv10DetectionModel(cfg)
                detmodel.load(weights)
                model.model = copy.deepcopy(detmodel)
            else:
                model = YOLOv10.from_pretrained('jameslahm/yolov10x')

        elif self._model_conf.pretrained_type == "yolov8":

            assert self._model_conf.model_path is not None, "Model path cannot be None"
            model = YOLO(self._model_conf.model_path) 

        elif self._model_conf.pretrained_type == "yolov5":
            model = YOLO('yolov5n.pt') 

        else:
            raise ValueError(f"Model type '{type}' is not supported.")

        return model.to(self._device)

    def reset_model(self) -> None:

        if self._model is not None :
            del self._model

        self._model = copy.deepcopy(self._init_model)
        self._detmodel = self._model.model.train()
        replace_c2f_with_c2f_v2(self._detmodel)
        for _, param in self._detmodel.named_parameters():
            param.requires_grad = True 
        self._model.model = copy.deepcopy(self._detmodel)

        self.determine_prunable_layers()

    def determine_prunable_layers(self) -> None: 

        flattened_layers = [(name, module) for name, module in self._detmodel.named_modules() if isinstance(module, nn.Conv2d)]    
        ignored_modules = []
        unwrapped_parameters = []
        for m in self._detmodel.modules():
            if isinstance(m, (Detect,)):
                ignored_modules.append(m)

        ignored_layers = [layer for module in ignored_modules for layer in module.modules() if isinstance(layer, nn.Conv2d)]

        self._prunable_layers = [layer for i, layer in enumerate(flattened_layers) if layer[1] not in ignored_layers]

    def train(self):
        pass

    def evaluate(self) -> list:

        prec_metrics = self._model.val(data=self._model_conf.data, batch=self._model_conf.batch_size, plots=None)

        M_params = self.get_n_model_params()

        metrics = [float(np.around(m,4)) for m in list(prec_metrics.results_dict.values())[:4]] 
        metrics.append(M_params)

        return metrics # [precision, recall, map50, map95, M_paramns]

    def get_n_model_params(self) -> float:
        """Used when evaluation is not needed (map is already 0).
        """
        M_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        M_params = float(np.around(M_params, 2))

        return M_params

    def prune(self, all_indices, layer_i):

        self._detmodel.train()

        device = next(self._model.parameters()).device.type
        DG = tp.DependencyGraph().build_dependency(self._detmodel, self._example_input.to(device))

        def prune_conv_layer(layer: nn.Conv2d, indices: list) -> None:
            pruning_group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=indices)
            pruning_group.prune()

        indices = all_indices[layer_i]     
        layer = self._prunable_layers[layer_i][1] #0:name, 1:layer

        if indices is not None: # and len(indices):   
            prune_conv_layer(layer, indices)

        del layer
        gc.collect()

        for name, param in self._detmodel.named_parameters():
            param.requires_grad = True 

        self._model.model = copy.deepcopy(self._detmodel)
    
    def fine_tune(
        self,
        data_yaml: str = None,
        lr: float = 1e-4,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 8,
        **kwargs
    ) -> None:
        """
        Fine-tune the *current* model (pruned or not) using Ultralytics' .train().
        After training, updates self._model with the trained weights.
        """
        pruned_model = self._detmodel 

        args = dict(
            model="",            # unused because we override get_model
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr,
            device=self._device,
            project="runs/pruned",
            name="yolo_pruned_finetune",
        )

        trainer = PrunedDetectionTrainer(pruned_model=pruned_model, overrides=args)
        # Make sure the trainer knows nc, names, args
        trainer.model.nc = trainer.data["nc"]
        trainer.model.names = trainer.data["names"]
        trainer.model.args = trainer.args

        if self._model_conf.pretrained_type == "yolov8":
            trainer.model.criterion = v8DetectionLoss(trainer.model)
        else:
            raise NotImplementedError(
                f"Fine-tuning is not supported for model type {self._model_conf.pretrained_type}. Criterion is missing."
            )
        print(type(trainer.model))  # should be DetectionModel (or similar nn.Module)
        print(hasattr(trainer.model, "criterion"))  # should be True
        print(trainer.model.criterion)  
        print("trainer model params:", sum(p.numel() for p in trainer.model.parameters()))

        # Make sure the trainer knows nc, names, args
        trainer.model.nc = trainer.data["nc"]
        trainer.model.names = trainer.data["names"]
        trainer.model.args = trainer.args

        results = trainer.train()
        self._model.model = copy.deepcopy(trainer.model)

    
    def save_pruned_model(self, path: Optional[str] = None) -> str:
        """
        Save the current pruned (and possibly fine-tuned) detection model.
        """
        if path is None:
            path = os.path.join("runs", "pruned", "pruned_model.pt")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        ckpt = {
            "detmodel": self._detmodel,        # full pruned DetectionModel
            "nc": getattr(self._detmodel, "nc", None),
            "names": getattr(self._detmodel, "names", None),
            "pretrained_type": self._model_conf.pretrained_type,
        }
        torch.save(ckpt, path)
        return path

    def load_pruned_model(self, path: str) -> None:
        """
        Load a previously saved pruned detection model from disk
        and attach it to self._model / self._detmodel.
        """
        ckpt = torch.load(path, map_location=self._device)

        detmodel = ckpt["detmodel"].to(self._device).train()
        # Restore nc/names if present
        if "nc" in ckpt and ckpt["nc"] is not None:
            detmodel.nc = ckpt["nc"]
        if "names" in ckpt and ckpt["names"] is not None:
            detmodel.names = ckpt["names"]

        self._detmodel = detmodel

        # Rebuild wrapper model around this pruned detmodel
        # (mirrors what reset_model() does, but without re-pruning)
        self._model = copy.deepcopy(self._init_model)
        self._model.model = copy.deepcopy(self._detmodel)

        # Recompute any indices / masks that depend on the current model
        self.determine_prunable_layers()

    def save_metrics():
        pass

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def prunable_layers(self) -> list:
        return self._prunable_layers

    @property
    def n_prunable_layers(self) -> int:
        return len(self._prunable_layers)

    @property
    def device(self) -> str:
        return self._device

    @property
    def prunable_in_channels(self) -> List[int]:
        return [layer[1].in_channels for layer in self._prunable_layers]

    @property
    def prunable_out_channels(self) -> List[int]:
        return [layer[1].out_channels for layer in self._prunable_layers]

    @property
    def prunable_kernel_sizes(self) -> List[int]:
        return [layer[1].kernel_size[0] for layer in self._prunable_layers]

    @property
    def prunable_strides(self) -> List[int]:
        return [layer[1].stride[0] for layer in self._prunable_layers]

    @property
    def prunable_paddings(self) -> List[int]:
        return [layer[1].padding[0] for layer in self._prunable_layers]



class PrunedDetectionTrainer(DetectionTrainer):
    def __init__(self, pruned_model, overrides=None, _callbacks=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)

        # override the model with (pruned) DetectionModel
        self.model = pruned_model.to(self.device)

        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

        self.set_model_attributes()
